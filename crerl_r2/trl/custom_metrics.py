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


def ensure_python_types(obj):
    """
    Recursively ensure all values in an object are proper Python types for PyArrow serialization.
    Converts numpy types and other non-standard types to their Python equivalents.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [ensure_python_types(item) for item in obj.tolist()]
    elif hasattr(obj, '__class__'):
        # Handle custom enum types that might have names like One, Two, etc.
        class_name = obj.__class__.__name__
        # Also check the module name to catch enum types from different modules
        module_name = obj.__class__.__module__ if hasattr(obj.__class__, '__module__') else ''
        
        # Check if it's a custom enum type with common names
        if any(name.lower() in class_name.lower() for name in ['integer', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']):
            # If obj is a custom enum type that represents a number, convert to int
            try:
                # Try to get the value directly if it's a simple enum
                if hasattr(obj, 'value'):
                    return int(obj.value)
                elif hasattr(obj, '__int__'):
                    return int(obj)
                elif hasattr(obj, '__index__'):
                    # For enum.IntEnum and similar
                    return int(obj.__index__())
                else:
                    # If it's a custom enum, try to convert by calling int on it directly
                    return int(obj)
            except (ValueError, TypeError, AttributeError):
                # If conversion fails, return 0 as default
                return 0
        # Handle other custom types that might cause issues
        elif 'enum' in class_name.lower() or 'intenum' in class_name.lower() or 'enum' in module_name.lower():
            try:
                return int(obj)
            except (ValueError, TypeError, AttributeError):
                return int(obj.value) if hasattr(obj, 'value') else 0
    elif isinstance(obj, (list, tuple)):
        return [ensure_python_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: ensure_python_types(value) for key, value in obj.items()}
    else:
        return obj


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
                logprobs=None
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
                logprobs=None
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
                logprobs=None
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


class F1ScoreSample(SampleLevelComputation):
    def __init__(self, base_metric: SampleLevelComputation):
        self.base_metric = base_metric

    def compute(self, doc: Doc, model_response: ModelResponse) -> tuple:
        # Get the prediction result from the base metric
        score = self.base_metric.compute(doc, model_response)
        
        # Determine if prediction is correct based on threshold
        is_correct = 1.0 if score >= 0.99 else 0.0
        
        # Get gold labels and predictions for F1 calculation
        gold_labels = doc.get_golds()
        if model_response.text is not None:
            pred_labels = [model_response.text] if isinstance(model_response.text, str) else model_response.text
        else:
            pred_labels = []
            
        # Process gold and predicted labels to create binary classification arrays
        # For simplicity, we'll consider exact matches for positive class
        gold_binary = [1.0] * len(gold_labels) if gold_labels else [0.0]
        pred_binary = [1.0] * len(pred_labels) if pred_labels else [0.0]
        
        # If we have a correct prediction, adjust the arrays accordingly
        if is_correct and gold_labels and pred_labels:
            # At least one prediction matches gold, so we have true positives
            # For F1 calculation, we need to determine TP, FP, FN
            tp = min(len(gold_labels), len(pred_labels)) if is_correct else 0
            fp = max(0, len(pred_labels) - tp)
            fn = max(0, len(gold_labels) - tp)
        else:
            # If prediction is incorrect, all predictions are false positives
            # and all golds are false negatives
            tp = 0
            fp = len(pred_labels) if pred_labels else 0
            fn = len(gold_labels) if gold_labels else 0
            
        return (tp, fp, fn)


def f1_corpus_level(items):
    """Calculate F1 score from TP, FP, FN counts."""
    if not items:
        return 0.0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            tp, fp, fn = item[0], item[1], item[2]
            total_tp += tp
            total_fp += fp
            total_fn += fn
        elif hasattr(item, 'pred') and isinstance(item.pred, (tuple, list)) and len(item.pred) >= 3:
            # Handle LogprobCorpusMetricInput-like objects
            tp, fp, fn = item.pred[0], item.pred[1], item.pred[2]
            total_tp += tp
            total_fp += fp
            total_fn += fn
        else:
            # Default case - treat as no matches
            continue
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def make_f1_score(base_metric: SampleLevelMetric):
    return SampleLevelMetric(
        metric_name="f1_score",
        higher_is_better=True,
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=F1ScoreSample(
            base_metric=base_metric.sample_level_fn,
        ),
        corpus_level_fn=f1_corpus_level,
    )


def make_metrics_at_k(k_values, base_sample_metric):

    return [
        make_majority_at_k(k, 0.99, base_sample_metric) for k in k_values
    ] + [
        make_accuracy_at_k(k, 0.99, base_sample_metric) for k in k_values
    ]
    #make_f1_score(base_sample_metric)

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




# Multiclass Expected Calibration Error (ECE) Implementation
class MulticlassECESample(SampleLevelComputation):
    def __init__(self, n_bins=15, language=Language.ENGLISH, gold_extraction_target=(ExprExtractionConfig(),), 
                 pred_extraction_target=(ExprExtractionConfig(),), extraction_mode="any_match", 
                 fallback_mode="first_match", precision=6, timeout_seconds=5):
        self.n_bins = n_bins
        self.language = language
        self.gold_extraction_target = gold_extraction_target
        self.pred_extraction_target = pred_extraction_target
        self.extraction_mode = extraction_mode
        self.fallback_mode = fallback_mode
        self.precision = precision
        self.timeout_seconds = timeout_seconds

    def compute(self, doc: Doc, model_response: ModelResponse) -> tuple:
        # Extract gold label and prediction
        golds = doc.get_golds() or []
        golds = [g for g in golds if isinstance(g, str)]

        # Normalize predictions
        preds: list[str] = []
        if isinstance(model_response.text, list):
            preds.extend([p for p in model_response.text if isinstance(p, str)])
        elif isinstance(model_response.text, str):
            preds.append(model_response.text)

        # For LOGPROBS tasks, we need to get confidence from the logprobs
        # The structure of ModelResponse for LOGPROBS sampling should have logprobs available
        confidence = 0.5  # Default confidence

        # Try to extract confidence from logprobs - this is the key for ECE
        # The logprobs attribute might have different structures depending on the model and sampling method
        # Try multiple possible attribute names for logprobs
        logprobs_attr = None
        for attr_name in ['logprobs', 'log_probs', 'logprob', 'log_prob']:
            if hasattr(model_response, attr_name):
                attr_value = getattr(model_response, attr_name)
                if attr_value is not None:
                    logprobs_attr = attr_value
                    break

        if logprobs_attr is not None:
            try:
                # For LOGPROBS sampling, logprobs should contain the log probabilities of the generated tokens
                # We need to extract confidence from these logprobs
                if isinstance(logprobs_attr, list):
                    # If logprobs is a list of token logprobs, we can aggregate them
                    # This is typical for generative evaluations with logprobs
                    token_logprobs = []
                    
                    for token_info in logprobs_attr:
                        # Handle different possible structures of token information in logprobs
                        if hasattr(token_info, 'logprob'):
                            # TokenInfo object with logprob attribute
                            token_logprobs.append(float(token_info.logprob))
                        elif hasattr(token_info, 'logprobs'):  # Alternative attribute name (plural)
                            token_logprobs.append(float(token_info.logprobs))
                        elif isinstance(token_info, dict):
                            # Dictionary with logprob information
                            if 'logprob' in token_info:
                                token_logprobs.append(float(token_info['logprob']))
                            elif 'logprobs' in token_info:  # Alternative spelling
                                token_logprobs.append(float(token_info['logprobs']))
                            elif 'log_probs' in token_info:
                                token_logprobs.append(float(token_info['log_probs']))
                            elif 'top_logprobs' in token_info and isinstance(token_info['top_logprobs'], list):
                                # If it's a list of top logprobs, we can take the max or average
                                top_logprobs = [float(lp['logprob']) if 'logprob' in lp else float(lp.get('logprob', 0)) for lp in token_info['top_logprobs'] if isinstance(lp, dict)]
                                if top_logprobs:
                                    token_logprobs.extend(top_logprobs)
                        elif isinstance(token_info, (float, int)):
                            # Direct logprob value
                            token_logprobs.append(float(token_info))
                        elif isinstance(token_info, (list, tuple)) and len(token_info) > 0:
                            # If it's a list/tuple, try to get the first element that looks like a logprob
                            for item in token_info:
                                if isinstance(item, (float, int)):
                                    token_logprobs.append(float(item))
                                    break
                        elif hasattr(token_info, '__getitem__'):
                            # Indexable object (like dict or list accessed via __getitem__)
                            try:
                                if 'logprob' in token_info:
                                    token_logprobs.append(float(token_info['logprob']))
                                elif 'logprobs' in token_info:
                                    token_logprobs.append(float(token_info['logprobs']))
                                elif 'log_probs' in token_info:
                                    token_logprobs.append(float(token_info['log_probs']))
                                elif len(token_info) > 0:
                                    # If it's a sequence, try the first element
                                    first_item = token_info[0]
                                    if isinstance(first_item, (float, int)):
                                        token_logprobs.append(float(first_item))
                            except (TypeError, KeyError, IndexError):
                                pass

                    if token_logprobs:
                        # Aggregate the token logprobs to get a single confidence measure
                        # We can use the mean of the logprobs and convert to probability
                        avg_logprob = sum(token_logprobs) / len(token_logprobs)
                        # Convert logprob to probability (clamping to ensure valid probability)
                        confidence = float(np.exp(min(avg_logprob, 0)))
                elif hasattr(logprobs_attr, 'logprob'):
                    # Single object with logprob attribute
                    confidence = float(np.exp(min(logprobs_attr.logprob, 0)))
                elif hasattr(logprobs_attr, 'logprobs'):  # Alternative attribute name (plural)
                    confidence = float(np.exp(min(logprobs_attr.logprobs, 0)))
                elif isinstance(logprobs_attr, (float, int)):
                    # Direct logprob value
                    confidence = float(np.exp(min(logprobs_attr, 0)))
                elif isinstance(logprobs_attr, dict):
                    # Dictionary-like object with logprob information
                    if 'logprob' in logprobs_attr:
                        confidence = float(np.exp(min(logprobs_attr['logprob'], 0)))
                    elif 'logprobs' in logprobs_attr:
                        confidence = float(np.exp(min(logprobs_attr['logprobs'], 0)))
                    elif 'log_probs' in logprobs_attr:  # Alternative key with underscore
                        confidence = float(np.exp(min(logprobs_attr['log_probs'], 0)))
                elif hasattr(logprobs_attr, '__dict__'):
                    # Object with attributes, check for common logprob attribute names
                    if hasattr(logprobs_attr, 'logprob'):
                        confidence = float(np.exp(min(logprobs_attr.logprob, 0)))
                    elif hasattr(logprobs_attr, 'logprobs'):
                        confidence = float(np.exp(min(logprobs_attr.logprobs, 0)))
                    elif hasattr(logprobs_attr, 'values'):
                        # If it has a values attribute that might contain logprobs
                        values = getattr(logprobs_attr, 'values', [])
                        if isinstance(values, list):
                            avg_val = sum(values) / len(values) if values else 0
                            confidence = float(np.exp(min(avg_val, 0)))
                    elif hasattr(logprobs_attr, '__getitem__'):
                        # If it supports indexing, try common keys
                        try:
                            if 'logprob' in logprobs_attr:
                                confidence = float(np.exp(min(logprobs_attr['logprob'], 0)))
                            elif 'logprobs' in logprobs_attr:
                                confidence = float(np.exp(min(logprobs_attr['logprobs'], 0)))
                        except (TypeError, AttributeError):
                            pass
            except Exception as e:
                logger.warning(f"ECE confidence extraction error: {e}")
                # Keep default confidence of 0.5
        else:
            # If no logprobs are available, this means we're not getting the right response type for ECE
            # This could happen if the metric is being computed on GENERATIVE responses instead of LOGPROBS responses
            logger.warning("No logprobs available for ECE confidence calculation - this indicates the response may be from GENERATIVE sampling instead of LOGPROBS sampling")
            confidence = 0.5

        # Ensure confidence is in valid range [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        # Use the same extractive matching approach as accuracy calculation to determine correctness
        is_correct = 0
        if golds and preds:
            # Use the same extractive matching approach as accuracy calculation
            gold_regexes = get_extraction_regexes(doc, self.gold_extraction_target, self.language)
            pred_regexes = get_extraction_regexes(doc, self.pred_extraction_target, self.language)

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

            # Calculate scores for all combinations of extracted golds and predictions
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
                    except Exception as e:
                        logger.warning(f"ECE sample computation - Compare gold target error: {e}")
                        match = False

                    scores.append(1.0 if match else 0.0)

            # Use the aggregation function (max) to determine if any combination matched
            is_correct = max(scores) if scores else 0
        else:
            # If no golds or preds, default to incorrect
            is_correct = 0

        # Ensure we return valid float values to avoid None issues
        confidence = float(confidence) if confidence is not None else 0.5
        is_correct = float(is_correct) if is_correct is not None else 0.0

        logger.debug(f"ECE sample computation - Confidence: {confidence}, Is Correct: {is_correct}")

        # Return a simple tuple instead of a complex dictionary to avoid PyArrow serialization issues
        return (confidence, is_correct)


def multiclass_ece_corpus_level(items, n_bins=15):
    """Compute Expected Calibration Error for multiclass classification"""
    if not items:
        logger.info("ECE corpus level: No items received")
        return 0.0

    confidences = []
    correct = []

    for i, item in enumerate(items):
        # Handle different input formats depending on what LightEval passes
        if hasattr(item, 'gold') and hasattr(item, 'pred'):  # LogprobCorpusMetricInput-like object
            # Extract confidence and correctness from the item
            # Since this is for generative tasks, we need to process the sample-level results differently
            # The sample level function should return (confidence, is_correct) tuples
            if isinstance(item.pred, (list, tuple)) and len(item.pred) >= 2:
                conf = float(item.pred[0]) if item.pred[0] is not None else 0.5
                corr = float(item.pred[1]) if item.pred[1] is not None else 0.0
                confidences.append(ensure_python_types(conf))
                correct.append(ensure_python_types(corr))
            else:
                # If pred is not in expected format, skip or use defaults
                confidences.append(ensure_python_types(0.5))  # Default confidence
                correct.append(ensure_python_types(0.0))      # Default to incorrect
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Item is directly a (confidence, is_correct) tuple from sample level
            conf = float(item[0]) if item[0] is not None else 0.5
            corr = float(item[1]) if item[1] is not None else 0.0
            confidences.append(ensure_python_types(conf))
            correct.append(ensure_python_types(corr))
        elif isinstance(item, (int, float)):
            # If item is just a single value, treat as correctness and assume default confidence
            confidences.append(ensure_python_types(0.5))  # Default confidence
            correct.append(ensure_python_types(float(item)))  # Treat as correctness
        else:
            # Handle other formats gracefully
            confidences.append(ensure_python_types(0.5))  # Default confidence
            correct.append(ensure_python_types(0.0))      # Default to incorrect

    # Check if we have any valid confidences after processing
    if not confidences or all(c is None for c in confidences):
        logger.info("ECE corpus level: No valid confidences found")
        return 0.0

    # Filter out any None values if they exist
    filtered_data = [(conf, corr) for conf, corr in zip(confidences, correct) if conf is not None and corr is not None]
    
    if not filtered_data:
        logger.info("ECE corpus level: No valid data after filtering")
        return 0.0
    
    # Separate the filtered data
    confidences, correct = zip(*filtered_data)
    
    # Convert to numpy arrays
    confidences = np.array(confidences, dtype=float)
    correct = np.array(correct, dtype=float)

    # Use the official calibration error implementation from HLE
    # Adapted from the calib_err function in hle.py - following the same pattern
    idxs = np.argsort(confidences)
    confidence_sorted = confidences[idxs]
    correct_sorted = correct[idxs]

    # Calculate bin boundaries based on number of samples per bin (following HLE pattern)
    n_samples = len(confidence_sorted)
    if n_samples == 0:
        logger.info("ECE corpus level: No samples after filtering")
        return 0.0
        
    if n_bins > n_samples and n_samples > 0:
        n_bins = n_samples  # Adjust if we have fewer samples than bins

    # Calculate samples per bin (following HLE pattern)
    samples_per_bin = max(1, n_samples // n_bins) if n_samples > 0 else 1

    # Create bins based on sample count (similar to HLE implementation)
    bins = [[i * samples_per_bin, min((i + 1) * samples_per_bin, n_samples)] for i in range(n_samples // samples_per_bin)]
    if n_samples % samples_per_bin != 0 and n_samples > 0:
        # Add the last bin for remaining samples
        bins.append([len(bins) * samples_per_bin, n_samples])

    if len(bins) == 0 or n_samples == 0:
        logger.warning("Error when computing the bins for calibration error")
        return 0.0

    cerr = 0.0
    total_examples = len(confidence_sorted)

    for i in range(len(bins)):
        start_idx, end_idx = bins[i]
        bin_confidence = confidence_sorted[start_idx:end_idx]
        bin_correct = correct_sorted[start_idx:end_idx]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            # Use L1 norm (p=1) for ECE calculation (consistent with HLE implementation)
            cerr += num_examples_in_bin / total_examples * difference

    final_ece = float(cerr)

    # Ensure we return a plain Python float to avoid PyArrow serialization issues
    return ensure_python_types(final_ece)


# Define the wrapper function outside to avoid pickling issues
def ece_corpus_level_wrapper(results, n_bins=15):
    return multiclass_ece_corpus_level(results, n_bins=n_bins)

# Add __name__ attribute to make it compatible with lighteval's stderr calculation
ece_corpus_level_wrapper.__name__ = "multiclass_ece_corpus_level"

# Create static functions for different bin configurations to ensure picklability
def ece_corpus_level_5_bins(results):
    return ece_corpus_level_wrapper(results, n_bins=5)

def ece_corpus_level_10_bins(results):
    return ece_corpus_level_wrapper(results, n_bins=10)

def ece_corpus_level_15_bins(results):
    return ece_corpus_level_wrapper(results, n_bins=15)

def ece_corpus_level_20_bins(results):
    return ece_corpus_level_wrapper(results, n_bins=20)

# Map n_bins to the corresponding function to ensure picklability
ECE_CORPUS_LEVEL_FUNCTIONS = {
    5: ece_corpus_level_5_bins,
    10: ece_corpus_level_10_bins,
    15: ece_corpus_level_15_bins,
    20: ece_corpus_level_20_bins
}

def make_multiclass_ece(n_bins=15):
    # Use a predefined function if available, otherwise default to 15 bins
    corpus_fn = ECE_CORPUS_LEVEL_FUNCTIONS.get(n_bins, ece_corpus_level_15_bins)
    # Update the function's name to reflect the actual n_bins used
    corpus_fn.__name__ = "multiclass_ece_corpus_level"
    
    return SampleLevelMetric(
        metric_name="multiclass_ece",
        higher_is_better=False,  # Lower ECE is better (better calibration)
        category=SamplingMethod.LOGPROBS,  # Use LOGPROBS category to ensure logprobs are available
        sample_level_fn=MulticlassECESample(n_bins=n_bins),
        corpus_level_fn=corpus_fn
    )