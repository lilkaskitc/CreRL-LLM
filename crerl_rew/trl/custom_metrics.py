from aenum import extend_enum
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.models.model_output import ModelResponse

from collections import Counter
from itertools import groupby
import re
import logging
from typing import Callable, Literal, Sequence
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.language import Language
from lighteval.utils.timeout import timeout
from lighteval.metrics.utils.extractive_match_utils import (  # noqa: F401
    ExprExtractionConfig,
    ExtractionTarget,
    extract_target_from_pred,
    extract_match,
    get_extraction_regexes,
)
import numpy as np


logger = logging.getLogger(__name__)


class MajorityAtK(SampleLevelComputation):
    def __init__(self, k: int, threshold: float, base_metric: SampleLevelComputation):
        self.k = k
        self.threshold = threshold
        self.base_metric = base_metric

    def compute(self, doc: Doc, model_response: ModelResponse) -> float:
        # Normalize predictions (same logic as before)
        preds = []
        if model_response.text is not None:
            preds = [model_response.text]
      

        if not preds:
            return 0.0

        # Take up to k predictions
        preds = preds[: self.k]

        votes = 0
        for pred in preds:
            single_response = ModelResponse(
                text=pred,
                logprobs=None,
              
            )
            votes += float(self.base_metric.compute(doc, single_response))

        return float(votes >= self.threshold)

def make_majority_at_k(k: int, threshold: float, base_metric: SampleLevelMetric):
    return SampleLevelMetric(
        metric_name=f"maj@{k}",
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=MajorityAtK(
            k=k,
            threshold=threshold,
            base_metric=base_metric.sample_level_fn,
        ),
        corpus_level_fn=np.mean,
    )

class PassAtK(SampleLevelComputation):
    def __init__(self, k: int, threshold: float, base_metric: SampleLevelComputation):
        self.k = k
        self.threshold = threshold
        self.base_metric = base_metric

    def compute(self, doc: Doc, model_response: ModelResponse) -> float:
        # Extract predictions
        if model_response.text is not None:
            preds = [model_response.text]
        else:
            return 0.0

        preds = preds[: self.k]

        for pred in preds:
            single_response = ModelResponse(
                text=pred,
                logprobs=None,
               
            )

            score = self.base_metric.compute(doc, single_response)
            if score >= self.threshold:
                return 1.0

        return 0.0

def make_pass_at_k(k: int, threshold: float, base_metric: SampleLevelMetric):
    return SampleLevelMetric(
        metric_name=f"pass@{k}",
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=PassAtK(
            k=k,
            threshold=threshold,
            base_metric=base_metric.sample_level_fn,
        ),
        corpus_level_fn=np.mean,
    )

class AccuracyAtK(SampleLevelComputation):
    def __init__(self, k: int, threshold: float, base_metric: SampleLevelComputation):
        self.k = k
        self.threshold = threshold
        self.base_metric = base_metric

    def compute(self, doc: Doc, model_response: ModelResponse) -> float:
        if model_response.text is not None:
            preds = [model_response.text]
        else:
            return 0.0

        preds = preds[: self.k]
        if not preds:
            return 0.0

        correct = 0
        for pred in preds:
            single_response = ModelResponse(
                text=pred,
                logprobs=None,
              
            )
            score = self.base_metric.compute(doc, single_response)
            if score >= self.threshold:
                correct += 1

        return correct / self.k

def make_accuracy_at_k(k: int, threshold: float, base_metric: SampleLevelMetric):
    return SampleLevelMetric(
        metric_name=f"acc@{k}",
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=AccuracyAtK(
            k=k,
            threshold=threshold,
            base_metric=base_metric.sample_level_fn,
        ),
        corpus_level_fn=np.mean,
    )



def make_metrics_at_k(k_values, base_sample_metric):
    return [
        make_majority_at_k(k, 0.99, base_sample_metric) for k in k_values
    ] + [
        make_pass_at_k(k, 0.99, base_sample_metric) for k in k_values
    ] + [
        make_accuracy_at_k(k, 0.99, base_sample_metric) for k in k_values
    ]


class MultilingualExtractiveMatchRobust(SampleLevelComputation):
    def __init__(
        self,
        language,
        gold_extraction_target,
        pred_extraction_target,
        aggregation_function,
        fallback_mode,
        extraction_mode,
        precision,
        timeout_seconds,
    ):
        self.language = language
        self.gold_extraction_target = gold_extraction_target
        self.pred_extraction_target = pred_extraction_target
        self.aggregation_function = aggregation_function
        self.fallback_mode = fallback_mode
        self.extraction_mode = extraction_mode
        self.precision = precision
        self.timeout_seconds = timeout_seconds

    def compute(self, doc: Doc, model_response: ModelResponse) -> float:
        # ---- GOLD NORMALIZATION ----
        golds = doc.get_golds() or []
        golds = [g for g in golds if isinstance(g, str)]
        # ---- PRED NORMALIZATION ----
        preds: list[str] = []

        # Case 1:  text is a list[str]
        if isinstance(model_response.text, list):
            preds.extend([p for p in model_response.text if isinstance(p, str)])

        # Case 3: text is a single str
        elif isinstance(model_response.text, str):
            preds.append(model_response.text)
        

        if not golds or not preds:
            return 0.0

        gold_regexes = get_extraction_regexes(
            doc, self.gold_extraction_target, self.language
        )
        pred_regexes = get_extraction_regexes(
            doc, self.pred_extraction_target, self.language
        )

        extracted_golds = [
            extract_target_from_pred_robust(
                g,
                gold_regexes,
                self.fallback_mode,
                self.extraction_mode,
                self.timeout_seconds,
            )
            for g in golds
        ]

        extracted_preds = [
            extract_target_from_pred_robust(
                p,
                pred_regexes,
                self.fallback_mode,
                self.extraction_mode,
                self.timeout_seconds,
            )
            for p in preds
        ]

        if not any(extracted_golds):
            return 0.0

        scores = []

        for eg in extracted_golds:
            for ep in extracted_preds:
                # eg and ep are already lists → pass directly
                try:
                    match = compare_gold_target(
                        eg,
                        ep,
                        self.precision,
                        timeout_seconds=self.timeout_seconds,
                    )
                except Exception:
                    match = False

                scores.append(1.0 if match else 0.0)


        return self.aggregation_function(scores)

def multilingual_extractive_match_metric_robust(
    language: Language = Language.ENGLISH,
    gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    aggregation_function: Callable[[list[float]], float] = max,
    fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
    precision: int = 6,
    timeout_seconds: int = 5,
) -> SampleLevelMetric:
    """Creates a language-aware extractive match metric that extracts answers from the model's output.

    Known issues:
    - If the task is to simplify an expression, the metric might overestimate the accuracy. This is because if the model doesn't output any anchor for the extraction (e.g final answer is..),
        it's possible that the the extracted prediction will be the expression to simplify. Because we do simplifications ourselves, it can thus happen that sympy will correctly simplify the expression,
        thus it will match gold, despite model not doing anything. PRs to fix this are welcome.

    - There is currently no StringExtractionConfig, so if the gold is \boxed{\text{Friday}} and model outputs Friday it will not match, because nothing will be extracted.

    Args:
        language: Language
            The language of the samples.
        gold_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for gold answers. Defaults to extracting simple math expressions.
        pred_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for predictions. Defaults to extracting simple math expressions.
        aggregation_function: Callable[[list[float]], float]
            Function to aggregate scores when multiple golds/predictions are present. Defaults to max.
        fallback_mode: Literal["no_fallback", "first_match"]
            How to perform extraction. Defaults to "first_match".
            - "no_fallback": Only use first successfully parsed matches
            - "first_match": Use the first successfully parsed match + first match irregardless the parsing success
        extraction_mode: Literal["first_match", "any_match"]
            - "first_match": Only tries to extract the first regex match if it fails no other matches are tried
            - "any_match": Tries to extract any regex match

        precision: int
            Number of decimal places to use when comparing numerical values. Defaults to 6.
        timeout_seconds: int
            Timeout for the extraction (each attempt) and comparison. Defaults to 5.

    Returns:
        A sample level metric that extracts and compares mathematical expressions.

    """

    @timeout(2)
    def add_to_specifics_with_timeout(
        formatted_doc: Doc, extracted_predictions: list[list[str]], extracted_golds: list[list[str]]
    ) -> None:
        if formatted_doc.specific is None:
            formatted_doc.specific = {}

        formatted_doc.specific["extracted_predictions"] = [
            str(pred) for preds in extracted_predictions for pred in preds
        ]
        formatted_doc.specific["extracted_golds"] = [str(gold) for golds in extracted_golds for gold in golds]

 
    return SampleLevelMetric(
        metric_name="extractive_match",
        sample_level_fn=MultilingualExtractiveMatchRobust(
            language,
            gold_extraction_target,
            pred_extraction_target,
            aggregation_function,
            fallback_mode,
            extraction_mode,
            precision,
            timeout_seconds,
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )


def extract_target_from_pred_robust(
    pred: str,
    target_res: list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]],
    fallback_mode: Literal["no_fallback", "first_match"] = "no_fallback",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
    timeout_seconds: int = 5,
):
    """Extracts targets from a prediction string using regex patterns.
    Returns first sucesffuly extracted match.

    Args:
        pred (str): The prediction string to extract from
        target_res (list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]]): List of regex patterns and their priorities for each target type
        fallback_mode (Literal["no_fallback", "first_match"], optional): How to handle extraction failures. Defaults to "no_fallback".
            - "no_fallback": Return only successfully parsed match
            - "first_match": Additionaly Include the first string match no matter how parsing finished
        extraction_mode (Literal["first_match", "any_match"], optional): How to handle extraction failures. Defaults to "any_match".
            - "first_match": Only tries to extract the first match
            - "any_match": Tries to extract any match
        timeout_seconds (int, optional): Maximum time in seconds to spend parsing each expression. Defaults to 5.

    Returns:
        list: List of extracted predictions, with first fallbac string appended if fallback_mode is "first_match"
    """
    if not isinstance(pred, str):
        logger.warning(f"Non-string pred passed to extractor: {type(pred)}")
        return []

    extracted_predictions = []
    fallbacks = []

    # Get all patterns and sort by priority
    all_patterns = [
        (pattern, target_type, priority)
        for target_patterns, target_type in target_res
        for pattern, priority in target_patterns
    ]
    match_found = False

    # Group patterns by priority using itertools.groupby
    for _, patterns_group in groupby(sorted(all_patterns, key=lambda x: x[2]), key=lambda x: x[2]):
        # Find all matches for each pattern in this priority group
        matches_with_pos = (
            (match, match.start(), match.end(), target_type)
            for pattern, target_type, _ in patterns_group
            for match in pattern.finditer(pred)
        )

        # Sort matches by end position (rightmost first) and then by start position (leftmost first)
        matches_with_pos = sorted(matches_with_pos, key=lambda x: (x[2], -x[1]), reverse=True)

        # Try to extract from each match, starting from rightmost
        for match, _, _, target_type in matches_with_pos:
            try:
                extracted_match, str_fallback = extract_match(match, target_type, timeout_seconds)
            except Exception:
                continue

            match_found = True

            if str_fallback:
                fallbacks.append(str_fallback)

            if extracted_match is not None:
                extracted_predictions.append(extracted_match)
                break

            if extraction_mode == "first_match":
                break

        # If we found something and we're in first_match mode, stop processing other priorities
        if extracted_predictions or (match_found and extraction_mode == "first_match"):
            break

    if fallback_mode == "first_match" and fallbacks:
        extracted_predictions += [fallbacks[0]]

    return extracted_predictions