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
import torch
from torchmetrics.classification import CalibrationError


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
    ] + [
        make_f1_score(base_sample_metric)

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
                # Check if item.pred[0] is a list/array and extract the confidence properly
                pred_0 = item.pred[0]
                if isinstance(pred_0, (list, tuple, np.ndarray)):
                    # If it's a list/array, take the mean to get a single confidence value
                    conf = float(np.mean(pred_0)) if len(pred_0) > 0 else 0.5
                else:
                    conf = float(pred_0) if pred_0 is not None and not np.isnan(pred_0) else 0.5
                
                pred_1 = item.pred[1]
                if isinstance(pred_1, (list, tuple, np.ndarray)):
                    # If it's a list/array, take the mean to get a single correctness value
                    corr = float(np.mean(pred_1)) if len(pred_1) > 0 else 0.0
                else:
                    corr = float(pred_1) if pred_1 is not None and not np.isnan(pred_1) else 0.0
                    
                confidences.append(ensure_python_types(conf))
                correct.append(ensure_python_types(corr))
            else:
                # If pred is not in expected format, skip or use defaults
                confidences.append(ensure_python_types(0.5))  # Default confidence
                correct.append(ensure_python_types(0.0))      # Default to incorrect
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Item is directly a (confidence, is_correct) tuple from sample level
            # Check if item[0] is a list/array and extract the confidence properly
            item_0 = item[0]
            if isinstance(item_0, (list, tuple, np.ndarray)):
                # If it's a list/array, take the mean to get a single confidence value
                conf = float(np.mean(item_0)) if len(item_0) > 0 else 0.5
            else:
                conf = float(item_0) if item_0 is not None and not np.isnan(item_0) else 0.5
            
            item_1 = item[1]
            if isinstance(item_1, (list, tuple, np.ndarray)):
                # If it's a list/array, take the mean to get a single correctness value
                corr = float(np.mean(item_1)) if len(item_1) > 0 else 0.0
            else:
                corr = float(item_1) if item_1 is not None and not np.isnan(item_1) else 0.0
                
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

    # Convert to numpy arrays
    confidences = np.array(confidences, dtype=float)
    correct = np.array(correct, dtype=float)

    # Remove any NaN values
    mask = ~(np.isnan(confidences) | np.isnan(correct))
    confidences = confidences[mask]
    correct = correct[mask]

    if len(confidences) == 0:
        logger.info("ECE corpus level: No valid data after filtering")
        return 0.0

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


# Implementation of ECE using PyTorch Metrics as per lighteval_torchmetric.md
def compute_ece(items, n_bins=15, norm='l1'):
    """
    Compute Expected Calibration Error using PyTorch Metrics CalibrationError
    """
    if not items:
        logger.info("ECE corpus level: No items received")
        return 0.0

    # Extract probabilities and targets from LightEval items
    # This assumes that items contain model responses with probabilities and gold indices
    probs_list = []
    targets_list = []

    for item in items:
        # Debug: Print the type and attributes of the item to understand the format
        if logger.level <= logging.DEBUG:
            logger.debug(f"ECE item type: {type(item)}, attributes: {dir(item) if hasattr(item, '__dir__') else 'no dir'}")

        # Handle different input formats depending on what LightEval passes
        if hasattr(item, 'model_response') and hasattr(item, 'doc'):
            # Standard LightEval format
            model_response = item.model_response
            doc = item.doc

            # Extract probabilities from model response
            if hasattr(model_response, 'result_probs'):
                # If result_probs is already available
                prob = model_response.result_probs
            elif hasattr(model_response, 'logprobs'):
                # If we have logprobs, convert to probabilities
                logprobs = model_response.logprobs
                if logprobs is not None:
                    # Convert logprobs to probabilities using softmax
                    try:
                        logprobs_tensor = torch.tensor(logprobs, dtype=torch.float32)
                        probs = torch.softmax(logprobs_tensor, dim=-1)
                        prob = probs.tolist()
                    except Exception as e:
                        logger.warning(f"Error converting logprobs to probabilities: {e}")
                        # Default to uniform distribution if conversion fails
                        num_classes = len(doc.choices) if hasattr(doc, 'choices') else 2
                        prob = [1.0 / num_classes] * num_classes
                else:
                    # Default to uniform distribution if no logprobs
                    num_classes = len(doc.choices) if hasattr(doc, 'choices') else 2
                    prob = [1.0 / num_classes] * num_classes
            else:
                # For GENERATIVE tasks, we don't have logprobs, so we need to handle this differently
                # We'll create a dummy probability distribution based on whether the prediction was correct
                logger.warning("No logprobs available for ECE computation - this may indicate the response is from GENERATIVE sampling instead of LOGPROBS sampling")
                
                # For generative tasks, we can't compute meaningful ECE without probabilities
                # So we'll skip this item or use a default approach
                continue
            # Extract target from document
            target = doc.gold_index if hasattr(doc, 'gold_index') else 0

            # Ensure prob is a valid list of floats and doesn't contain None
            if prob is None or not isinstance(prob, (list, tuple)):
                num_classes = len(doc.choices) if hasattr(doc, 'choices') else 2
                prob = [1.0 / num_classes] * num_classes

            # Validate that all elements in prob are numbers
            prob = [float(p) if p is not None else 0.5 for p in prob]

            probs_list.append(prob)
            targets_list.append(target)
        elif hasattr(item, 'pred') and hasattr(item, 'gold'):
            # LogprobCorpusMetricInput-like object - this is the format used by LightEval
            # The pred attribute contains the prediction information
            # The gold attribute contains the gold information
            pred_data = item.pred
            gold_data = item.gold

            # Extract probabilities and target from pred_data
            # The pred_data might be in different formats depending on the sample-level function
            if isinstance(pred_data, (list, tuple)) and len(pred_data) >= 2:
                # If pred_data is (probabilities, target) or (probabilities, target, is_correct)
                prob = pred_data[0]
                target = pred_data[1] if len(pred_data) > 1 else 0

                # Validate prob and target to ensure they are not None
                if prob is not None and target is not None:
                    # Ensure prob is a valid list of floats
                    if not isinstance(prob, (list, tuple)):
                        # If prob is a single value, convert to list
                        prob = [float(prob) if prob is not None else 0.5, 1.0 - float(prob) if prob is not None else 0.5]

                    # Validate that all elements in prob are numbers
                    prob = [float(p) if p is not None else 0.5 for p in prob]

                    probs_list.append(prob)
                    targets_list.append(int(target) if target is not None else 0)
            elif hasattr(pred_data, '__getitem__') and hasattr(pred_data, '__len__'):
                # If pred_data is array-like, try to extract prob and target
                try:
                    prob = pred_data[0] if len(pred_data) > 0 else None
                    target = pred_data[1] if len(pred_data) > 1 else 0

                    if prob is not None and target is not None:
                        if not isinstance(prob, (list, tuple)):
                            prob = [float(prob) if prob is not None else 0.5, 1.0 - float(prob) if prob is not None else 0.5]

                        prob = [float(p) if p is not None else 0.5 for p in prob]

                        probs_list.append(prob)
                        targets_list.append(int(target) if target is not None else 0)
                except (IndexError, TypeError):
                    # If extraction fails, skip this item
                    continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Item is directly a (probabilities, target) tuple
            prob, target = item[0], item[1]

            # Validate prob and target to ensure they are not None
            if prob is not None and target is not None:
                # Ensure prob is a valid list of floats
                if not isinstance(prob, (list, tuple)):
                    # If prob is a single value, convert to list
                    prob = [float(prob) if prob is not None else 0.5, 1.0 - float(prob) if prob is not None else 0.5]

                # Validate that all elements in prob are numbers
                prob = [float(p) if p is not None else 0.5 for p in prob]

                probs_list.append(prob)
                targets_list.append(int(target) if target is not None else 0)
        else:
            # Try to handle the most generic case by checking if item has pred attribute
            # This might be the format that LightEval is actually sending
            if hasattr(item, 'pred'):
                pred_data = item.pred
                # The pred_data might be a tuple of (confidence, is_correct) from our sample functions
                if isinstance(pred_data, (list, tuple)) and len(pred_data) >= 2:
                    # For ECE, we need probabilities, not just confidence
                    # If we have (confidence, is_correct), we need to reconstruct probabilities
                    if len(pred_data) == 2:
                        # This is likely (confidence, is_correct) from our sample functions
                        # We need to convert this back to a probability distribution
                        confidence = pred_data[0]
                        is_correct = pred_data[1]

                        # Create a simple probability distribution based on confidence
                        # For binary classification: [confidence, 1-confidence]
                        if isinstance(confidence, (int, float)):
                            prob = [float(confidence), 1.0 - float(confidence)]
                        else:
                            # If confidence is not a number, use uniform distribution
                            prob = [0.5, 0.5]

                        # For target, we'll use is_correct or derive from it
                        target = int(is_correct) if isinstance(is_correct, (bool, int, float)) else 0

                        probs_list.append(prob)
                        targets_list.append(target)
                    elif len(pred_data) >= 3:
                        # This might be (prob_distribution, target, is_correct) or similar
                        prob = pred_data[0] if isinstance(pred_data[0], (list, tuple)) else [0.5, 0.5]
                        target = pred_data[1] if len(pred_data) > 1 else 0

                        probs_list.append(prob)
                        targets_list.append(int(target) if target is not None else 0)
                elif isinstance(pred_data, (int, float)):
                    # If pred_data is just a single confidence value
                    confidence = float(pred_data)
                    prob = [confidence, 1.0 - confidence]
                    # For target, we'll need to infer or default
                    target = 0  # Default target

                    probs_list.append(prob)
                    targets_list.append(target)
                else:
                    # If we can't parse the pred_data, skip this item
                    continue
            else:
                # If none of the above conditions match, skip this item
                continue

    if not probs_list or not targets_list:
        logger.info("ECE corpus level: No valid probabilities or targets found")
        return 0.0

    # Filter out any None values and ensure all probabilities are valid
    valid_pairs = []
    for prob, target in zip(probs_list, targets_list):
        if prob is not None and target is not None:
            # Ensure all values in prob are valid numbers
            valid_prob = []
            for p in prob:
                if p is not None:
                    try:
                        valid_prob.append(float(p))
                    except (ValueError, TypeError):
                        valid_prob.append(0.5)  # Default value
                else:
                    valid_prob.append(0.5)  # Default value
            valid_pairs.append((valid_prob, int(target)))

    if not valid_pairs:
        logger.info("ECE corpus level: No valid probability-target pairs after filtering")
        return 0.0

    # Separate the filtered probabilities and targets
    probs_list = [pair[0] for pair in valid_pairs]
    targets_list = [pair[1] for pair in valid_pairs]

    # Convert to tensors
    try:
        # Ensure all probabilities have the same length for tensor creation
        if probs_list:
            max_len = max(len(prob) for prob in probs_list)
            # Pad shorter probability lists with zeros
            padded_probs = []
            for prob in probs_list:
                padded = list(prob) + [0.0] * (max_len - len(prob))
                padded_probs.append(padded)

            preds = torch.tensor(padded_probs, dtype=torch.float32)
            targets = torch.tensor(targets_list, dtype=torch.long)
        else:
            logger.info("ECE corpus level: No valid probabilities after filtering")
            return 0.0
    except Exception as e:
        logger.error(f"Error converting to tensors: {e}")
        return 0.0

    # Initialize and compute ECE using PyTorch Metrics
    try:
        # Check if num_classes is valid (> 1) before initializing CalibrationError
        num_classes = preds.shape[1] if len(preds.shape) > 1 else 1
        if num_classes <= 1:
            logger.warning(f"Number of classes is {num_classes}, which is invalid for multiclass ECE. Falling back to manual implementation.")
            return ensure_python_types(multiclass_ece_corpus_level(items, n_bins=n_bins))

        ece_metric = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=n_bins, norm=norm)
        ece_result = ece_metric(preds, targets)
        return ensure_python_types(float(ece_result.item()))
    except Exception as e:
        logger.error(f"Error computing ECE with PyTorch Metrics: {e}")
        # Fallback to our manual implementation
        return ensure_python_types(multiclass_ece_corpus_level(items, n_bins=n_bins))


# Wrapper functions for different ECE norms
def compute_ece_l1(items, n_bins=15):
    return compute_ece(items, n_bins=n_bins, norm='l1')

def compute_ece_l2(items, n_bins=15):
    return compute_ece(items, n_bins=n_bins, norm='l2')

def compute_ece_max(items, n_bins=15):
    return compute_ece(items, n_bins=n_bins, norm='max')


# Create ECE metrics with different norms
class ECESampleLevelComputation(SampleLevelComputation):
    """
    Sample level computation for ECE metrics. This extracts the necessary information
    from each sample that will be aggregated at the corpus level.
    """
    def __init__(self, language=Language.ENGLISH, gold_extraction_target=(ExprExtractionConfig(),),
                 pred_extraction_target=(ExprExtractionConfig(),), extraction_mode="any_match",
                 fallback_mode="first_match", precision=6, timeout_seconds=5):
        self.language = language
        self.gold_extraction_target = gold_extraction_target
        self.pred_extraction_target = pred_extraction_target
        self.extraction_mode = extraction_mode
        self.fallback_mode = fallback_mode
        self.precision = precision
        self.timeout_seconds = timeout_seconds

    def compute(self, doc: Doc, model_response: ModelResponse) -> tuple:
        """
        Extract probabilities and target information from each sample for ECE computation.
        """
        # Extract probabilities from model response
        if hasattr(model_response, 'result_probs'):
            # If result_probs is already available
            prob = model_response.result_probs
        elif hasattr(model_response, 'logprobs'):
            # If we have logprobs, convert to probabilities
            logprobs = model_response.logprobs
            if logprobs is not None and len(logprobs) > 0:
                # Handle different possible structures of logprobs
                if isinstance(logprobs, list) and len(logprobs) == 1:
                    # If logprobs is a single value in a list, this might be the sequence logprob
                    # For ECE, we need to handle this differently - we might need to get probabilities for each class
                    # But since we only have one logprob, we need to understand the context better

                    # For LOGPROBS sampling in classification tasks, we might need to get logprobs for each possible answer
                    # However, if we only have one logprob, it might be the logprob of the generated sequence
                    # For now, let's handle this by assuming it's the confidence of the prediction
                    sequence_logprob = logprobs[0]
                    confidence = float(torch.exp(torch.tensor(sequence_logprob)).item())

                    # For multiclass ECE, we need a probability distribution over classes
                    # If we only have one logprob, we need to infer the full distribution
                    # Consistently use 2 classes as a minimum to avoid issues with PyArrow
                    num_classes = max(2, len(doc.choices)) if hasattr(doc, 'choices') else 2
                    prob = [confidence if i == 0 else (1.0 - confidence)/(num_classes-1) if num_classes > 1 else 0.0
                            for i in range(num_classes)]
                elif isinstance(logprobs, list) and len(logprobs) > 1:
                    # Multiple logprobs - likely logprobs for each class option in a multiple-choice task
                    logprobs_tensor = torch.tensor(logprobs)
                    probs = torch.softmax(logprobs_tensor, dim=-1)
                    prob = probs.tolist()
                else:
                    # Default to uniform distribution if logprobs structure is unexpected
                    num_classes = max(2, len(doc.choices)) if hasattr(doc, 'choices') else 2  # Ensure at least 2 classes
                    prob = [1.0 / num_classes] * num_classes
            else:
                # Default to uniform distribution if no logprobs
                num_classes = max(2, len(doc.choices)) if hasattr(doc, 'choices') else 2  # Ensure at least 2 classes
                prob = [1.0 / num_classes] * num_classes
        else:
            # For GENERATIVE tasks, we don't have logprobs, so we need to handle this differently
            # We'll create a probability distribution based on whether the prediction was correct using extractive matching
            logger.debug("No logprobs available for ECE computation - using extractive matching for GENERATIVE task")

            # Extract gold label and prediction for extractive matching
            golds = doc.get_golds() or []
            golds = [g for g in golds if isinstance(g, str)]

            # Normalize predictions
            preds: list[str] = []
            if isinstance(model_response.text, list):
                preds.extend([p for p in model_response.text if isinstance(p, str)])
            elif isinstance(model_response.text, str):
                preds.append(model_response.text)

            # Use extractive matching to determine if prediction is correct
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

            # Create a simple probability distribution based on correctness
            # For binary classification: [confidence, 1-confidence] where confidence is based on correctness
            # Use a more conservative confidence estimate to avoid overconfidence
            confidence = 0.8 if is_correct else 0.2  # Moderate confidence if correct, moderate uncertainty if incorrect
            num_classes = max(2, len(doc.choices)) if hasattr(doc, 'choices') else 2  # Ensure at least 2 classes
            prob = [confidence if i == 0 else (1.0 - confidence)/(num_classes-1) if num_classes > 1 else 0.0
                    for i in range(num_classes)]

        # Ensure prob is always a list with consistent length
        if not isinstance(prob, list):
            # If prob is a single value, convert to a list with 2 elements
            prob = [float(prob), 1.0 - float(prob)]

        # Extract target from document
        target = doc.gold_index if hasattr(doc, 'gold_index') else 0
        # Ensure target is within bounds of the probability distribution
        target = min(target, len(prob) - 1) if target is not None else 0

        # Return the probability distribution and target for corpus-level ECE calculation
        # Ensure we return consistent Python types to avoid PyArrow serialization issues
        return (ensure_python_types(prob), ensure_python_types(target))


# Define static functions for different ECE configurations to avoid pickling issues
def ece_l1_corpus_level_15bins(items):
    return compute_ece_l1(items, n_bins=15)

def ece_l2_corpus_level_15bins(items):
    return compute_ece_l2(items, n_bins=15)

def ece_max_corpus_level_15bins(items):
    return compute_ece_max(items, n_bins=15)

def ece_l1_corpus_level_5bins(items):
    return compute_ece_l1(items, n_bins=5)

def ece_l2_corpus_level_5bins(items):
    return compute_ece_l2(items, n_bins=5)

def ece_max_corpus_level_5bins(items):
    return compute_ece_max(items, n_bins=5)

def ece_l1_corpus_level_10bins(items):
    return compute_ece_l1(items, n_bins=10)

def ece_l2_corpus_level_10bins(items):
    return compute_ece_l2(items, n_bins=10)

def ece_max_corpus_level_10bins(items):
    return compute_ece_max(items, n_bins=10)

def ece_l1_corpus_level_20bins(items):
    return compute_ece_l1(items, n_bins=20)

def ece_l2_corpus_level_20bins(items):
    return compute_ece_l2(items, n_bins=20)

def ece_max_corpus_level_20bins(items):
    return compute_ece_max(items, n_bins=20)

# Map of corpus functions for different configurations
ECE_CORPUS_FUNCTIONS = {
    ('l1', 5): ece_l1_corpus_level_5bins,
    ('l2', 5): ece_l2_corpus_level_5bins,
    ('max', 5): ece_max_corpus_level_5bins,
    ('l1', 10): ece_l1_corpus_level_10bins,
    ('l2', 10): ece_l2_corpus_level_10bins,
    ('max', 10): ece_max_corpus_level_10bins,
    ('l1', 15): ece_l1_corpus_level_15bins,
    ('l2', 15): ece_l2_corpus_level_15bins,
    ('max', 15): ece_max_corpus_level_15bins,
    ('l1', 20): ece_l1_corpus_level_20bins,
    ('l2', 20): ece_l2_corpus_level_20bins,
    ('max', 20): ece_max_corpus_level_20bins,
}


def make_ece_metric(norm='l1', n_bins=15):
    """
    Create an ECE metric with the specified norm and number of bins
    """
    metric_name = f"ece_{norm}_{n_bins}bins"
    
    # Get the appropriate corpus level function from the map
    corpus_fn = ECE_CORPUS_FUNCTIONS.get((norm, n_bins), ece_l1_corpus_level_15bins)
    
    # Update function name for compatibility
    corpus_fn.__name__ = f"compute_ece_{norm}"
    
    return SampleLevelMetric(
        metric_name=metric_name,
        higher_is_better=False,  # Lower ECE is better
        category=SamplingMethod.GENERATIVE,  # Changed to GENERATIVE to match task sampling method
        sample_level_fn=ECESampleLevelComputation(),
        corpus_level_fn=corpus_fn
    )


# Create default ECE metrics using the SampleLevelMetric approach
ece_l1_metric = make_ece_metric(norm='l1', n_bins=15)


# Custom ECE implementation for comparison
class CustomECESample(SampleLevelComputation):
    """
    Custom ECE sample level computation that extracts confidence and correctness
    """
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
        # Extract probabilities from model response (same as the other ECE implementation)
        if hasattr(model_response, 'result_probs'):
            # If result_probs is already available
            prob = model_response.result_probs
        elif hasattr(model_response, 'logprobs'):
            # If we have logprobs, convert to probabilities
            logprobs = model_response.logprobs
            if logprobs is not None and len(logprobs) > 0:
                # Handle different possible structures of logprobs
                if isinstance(logprobs, list) and len(logprobs) == 1:
                    # If logprobs is a single value in a list, this might be the sequence logprob
                    # For ECE, we need to handle this differently - we might need to get probabilities for each class
                    # But since we only have one logprob, we need to understand the context better

                    # For LOGPROBS sampling in classification tasks, we might need to get logprobs for each possible answer
                    # However, if we only have one logprob, it might be the logprob of the generated sequence
                    # For now, let's handle this by assuming it's the confidence of the prediction
                    sequence_logprob = logprobs[0]
                    confidence = float(torch.exp(torch.tensor(sequence_logprob)).item())

                    # For multiclass ECE, we need a probability distribution over classes
                    # If we only have one logprob, we need to infer the full distribution
                    # Consistently use 2 classes as a minimum to avoid issues with PyArrow
                    num_classes = max(2, len(doc.choices)) if hasattr(doc, 'choices') else 2
                    prob = [confidence if i == 0 else (1.0 - confidence)/(num_classes-1) if num_classes > 1 else 0.0
                            for i in range(num_classes)]
                elif isinstance(logprobs, list) and len(logprobs) > 1:
                    # Multiple logprobs - likely logprobs for each class option in a multiple-choice task
                    logprobs_tensor = torch.tensor(logprobs)
                    probs = torch.softmax(logprobs_tensor, dim=-1)
                    prob = probs.tolist()
                else:
                    # Default to uniform distribution if logprobs structure is unexpected
                    num_classes = max(2, len(doc.choices)) if hasattr(doc, 'choices') else 2  # Ensure at least 2 classes
                    prob = [1.0 / num_classes] * num_classes
            else:
                # Default to uniform distribution if no logprobs
                num_classes = max(2, len(doc.choices)) if hasattr(doc, 'choices') else 2  # Ensure at least 2 classes
                prob = [1.0 / num_classes] * num_classes
        else:
            # Default to uniform distribution
            num_classes = max(2, len(doc.choices)) if hasattr(doc, 'choices') else 2  # Ensure at least 2 classes
            prob = [1.0 / num_classes] * num_classes

        # Ensure prob is always a list with consistent length
        if not isinstance(prob, list):
            # If prob is a single value, convert to a list with 2 elements
            prob = [float(prob), 1.0 - float(prob)]

        # Extract target from document
        target = doc.gold_index if hasattr(doc, 'gold_index') else 0
        # Ensure target is within bounds of the probability distribution
        target = min(target, len(prob) - 1) if target is not None else 0

        # Use the same extractive matching approach as accuracy calculation to determine correctness
        is_correct = 0
        golds = doc.get_golds() or []
        golds = [g for g in golds if isinstance(g, str)]

        # Normalize predictions
        preds: list[str] = []
        if isinstance(model_response.text, list):
            preds.extend([p for p in model_response.text if isinstance(p, str)])
        elif isinstance(model_response.text, str):
            preds.append(model_response.text)

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
                        logger.warning(f"Custom ECE sample computation - Compare gold target error: {e}")
                        match = False

                    scores.append(1.0 if match else 0.0)

            # Use the aggregation function (max) to determine if any combination matched
            is_correct = max(scores) if scores else 0
        else:
            # If no golds or preds, default to incorrect
            is_correct = 0

        # Return the probability distribution and target for corpus-level ECE calculation
        # Ensure we return consistent Python types to avoid PyArrow serialization issues
        return (ensure_python_types(prob), ensure_python_types(target), ensure_python_types(float(is_correct)))


def custom_ece_corpus_level(items, n_bins=15):
    """
    Custom ECE corpus level computation following the standard ECE formula:
    ECE = sum(|B_m| / n * |acc(B_m) - conf(B_m)|)
    where B_m is bin m, acc(B_m) is accuracy in bin m, conf(B_m) is average confidence in bin m
    """
    if not items:
        return 0.0

    # Extract confidences and accuracies from items
    confidences = []
    accuracies = []  # 1.0 if correct, 0.0 if incorrect

    for item in items:
        if hasattr(item, 'pred') and isinstance(item.pred, (list, tuple)) and len(item.pred) >= 3:
            # Extract confidence and accuracy from the tuple returned by sample level
            # The sample level now returns (prob_distribution, target, is_correct)
            pred_0 = item.pred[0]  # prob_distribution
            pred_2 = item.pred[2]  # is_correct
            
            # Extract confidence from the probability distribution
            # For confidence, we'll use the max probability (confidence in the chosen answer)
            if isinstance(pred_0, (list, tuple, np.ndarray)):
                if len(pred_0) > 0:
                    # Get the maximum probability as confidence
                    conf = float(max(pred_0)) if len(pred_0) > 0 else 0.5
                else:
                    conf = 0.5
            else:
                conf = float(pred_0) if pred_0 is not None and not np.isnan(pred_0) else 0.5
                
            # Extract accuracy from the third element (is_correct)
            if isinstance(pred_2, (list, tuple, np.ndarray)):
                acc = float(np.mean(pred_2)) if len(pred_2) > 0 else 0.0
            else:
                acc = float(pred_2) if pred_2 is not None and not np.isnan(pred_2) else 0.0
            
            confidences.append(conf)
            accuracies.append(acc)
        elif hasattr(item, 'pred') and isinstance(item.pred, (list, tuple)) and len(item.pred) >= 2:
            # Handle the case where only 2 elements are returned
            pred_0 = item.pred[0]
            pred_1 = item.pred[1]
            
            # Handle cases where pred[0] or pred[1] might be arrays/lists
            if isinstance(pred_0, (list, tuple, np.ndarray)):
                conf = float(np.mean(pred_0)) if len(pred_0) > 0 else 0.5
            else:
                conf = float(pred_0) if pred_0 is not None and not np.isnan(pred_0) else 0.5
                
            if isinstance(pred_1, (list, tuple, np.ndarray)):
                acc = float(np.mean(pred_1)) if len(pred_1) > 0 else 0.0
            else:
                acc = float(pred_1) if pred_1 is not None and not np.isnan(pred_1) else 0.0
            
            confidences.append(conf)
            accuracies.append(acc)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            # Item is directly a (confidence, accuracy) tuple
            item_0 = item[0]
            item_1 = item[1]
            
            # Handle cases where item[0] or item[1] might be arrays/lists
            if isinstance(item_0, (list, tuple, np.ndarray)):
                conf = float(np.mean(item_0)) if len(item_0) > 0 else 0.5
            else:
                conf = float(item_0) if item_0 is not None and not np.isnan(item_0) else 0.5
                
            if isinstance(item_1, (list, tuple, np.ndarray)):
                acc = float(np.mean(item_1)) if len(item_1) > 0 else 0.0
            else:
                acc = float(item_1) if item_1 is not None and not np.isnan(item_1) else 0.0
            
            confidences.append(conf)
            accuracies.append(acc)
        else:
            # Default values if format is unexpected
            confidences.append(0.5)
            accuracies.append(0.0)

    if not confidences or not accuracies:
        return 0.0

    # Convert to numpy arrays
    confidences = np.array(confidences, dtype=float)
    accuracies = np.array(accuracies, dtype=float)

    # Remove any NaN values
    mask = ~(np.isnan(confidences) | np.isnan(accuracies))
    confidences = confidences[mask]
    accuracies = accuracies[mask]

    if len(confidences) == 0:
        return 0.0

    # Create bins based on confidence intervals
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(confidences)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_confidences = confidences[in_bin]
        bin_accuracies = accuracies[in_bin]

        if len(bin_confidences) > 0:
            # Calculate average confidence and accuracy in this bin
            avg_confidence_in_bin = np.mean(bin_confidences)
            avg_accuracy_in_bin = np.mean(bin_accuracies)

            # Calculate the ECE contribution from this bin
            bin_count = len(bin_confidences)
            bin_weight = bin_count / total_samples
            bin_ece = bin_weight * abs(avg_accuracy_in_bin - avg_confidence_in_bin)

            ece += bin_ece

    return float(ece)


# Define static wrapper functions outside to avoid pickling issues
def custom_ece_wrapper_5bins_l1(items):
    return custom_ece_corpus_level(items, n_bins=5)

def custom_ece_wrapper_10bins_l1(items):
    return custom_ece_corpus_level(items, n_bins=10)

def custom_ece_wrapper_15bins_l1(items):
    return custom_ece_corpus_level(items, n_bins=15)

def custom_ece_wrapper_20bins_l1(items):
    return custom_ece_corpus_level(items, n_bins=20)

def custom_ece_wrapper_5bins_l2(items):
    return custom_ece_corpus_level(items, n_bins=5)

def custom_ece_wrapper_10bins_l2(items):
    return custom_ece_corpus_level(items, n_bins=10)

def custom_ece_wrapper_15bins_l2(items):
    return custom_ece_corpus_level(items, n_bins=15)

def custom_ece_wrapper_20bins_l2(items):
    return custom_ece_corpus_level(items, n_bins=20)

def custom_ece_wrapper_5bins_max(items):
    return custom_ece_corpus_level(items, n_bins=5)

def custom_ece_wrapper_10bins_max(items):
    return custom_ece_corpus_level(items, n_bins=10)

def custom_ece_wrapper_15bins_max(items):
    return custom_ece_corpus_level(items, n_bins=15)

def custom_ece_wrapper_20bins_max(items):
    return custom_ece_corpus_level(items, n_bins=20)

# Map of corpus functions for different configurations
CUSTOM_ECE_CORPUS_FUNCTIONS = {
    ('l1', 5): custom_ece_wrapper_5bins_l1,
    ('l2', 5): custom_ece_wrapper_5bins_l2,
    ('max', 5): custom_ece_wrapper_5bins_max,
    ('l1', 10): custom_ece_wrapper_10bins_l1,
    ('l2', 10): custom_ece_wrapper_10bins_l2,
    ('max', 10): custom_ece_wrapper_10bins_max,
    ('l1', 15): custom_ece_wrapper_15bins_l1,
    ('l2', 15): custom_ece_wrapper_15bins_l2,
    ('max', 15): custom_ece_wrapper_15bins_max,
    ('l1', 20): custom_ece_wrapper_20bins_l1,
    ('l2', 20): custom_ece_wrapper_20bins_l2,
    ('max', 20): custom_ece_wrapper_20bins_max,
}


def make_custom_ece_metric(norm='l1', n_bins=15):
    """
    Create a custom ECE metric with the specified norm and number of bins
    """
    metric_name = f"custom_ece_{norm}_{n_bins}bins"

    # Get the appropriate corpus level function from the map
    corpus_fn = CUSTOM_ECE_CORPUS_FUNCTIONS.get((norm, n_bins), custom_ece_wrapper_15bins_l1)

    # Update function name for compatibility
    corpus_fn.__name__ = f"custom_compute_ece_{norm}"

    return SampleLevelMetric(
        metric_name=metric_name,
        higher_is_better=False,  # Lower ECE is better
        category=SamplingMethod.GENERATIVE,  # Changed to GENERATIVE to match task sampling method
        sample_level_fn=CustomECESample(n_bins=n_bins),
        corpus_level_fn=corpus_fn
    )