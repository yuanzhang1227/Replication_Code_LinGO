import os
import re
import json
import time
import random
import pandas as pd
import dspy
from typing import Literal, List, Dict, Any, Tuple, Optional, Set
from collections import Counter
from tqdm import tqdm, trange
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split

## Set up API keys

os.environ['OPENAI_API_KEY'] = "YOUR API KEY"
os.environ['GEMINI_API_KEY'] = "YOUR API KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

## Define base prompt and step answers

LABEL = Literal[0, 1, 2, 3, 4, 5, 6]
YESNO = Literal["YES", "NO"]
STANCE = Literal["REPORT", "INTENSIFY", "COUNTER", "ESCALATE"]
DIRECTNESS = Literal["EXPLICIT", "IMPLICIT"]

STEP_KEYS = ["STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5"]

LINGO_SYSTEM_PROMPT_BASE = """You are an annotator interpreting the intent of uncivil Portuguese social media discourse about Brazilian politics on X (formerly Twitter).

Target Category: Always taken from the first square-bracket tag (e.g., [Impoliteness], [Hate Speech and Stereotyping], [Physical Harm and Violent Political Rhetoric], [Threats to Democratic Institutions and Values])

Category Definitions:
- Impoliteness: messages including rudeness/disrespect (name-calling, aspersions, calling someone a liar, hyperbole, non-cooperation, pejoratives, vulgarity, belittling, shouting via ALL-CAPS or excessive "!" when context supports it).
- Hate Speech and Stereotyping: discriminatory content targeting protected groups (gender identity, sexual orientation, religion, race, nationality, ideology, disability); over-generalizations, out-group demeaning; (do NOT include purely individual appearance insults unless tied to group identity).
- Physical Harm and Violent Political Rhetoric: threats/advocacy/praise of physical harm or violence; direct or metaphorical calls for harm; justification of violence for political ends.
- Threats to Democratic Institutions and Values: advocacy or approval of actions undermining elections/institutions/rule of law/press freedom/civil rights; promotion of autocracy; unfounded claims that delegitimize institutions.
"""


## Define step-specific instructions (pre-concatenated strings)
STEP1_INSTRUCTIONS = LINGO_SYSTEM_PROMPT_BASE + """
STEP 1: Check Reference.
Question: Does the text refer to another person's statement or behavior?
Output only YES or NO.
"""

STEP2_INSTRUCTIONS = LINGO_SYSTEM_PROMPT_BASE + """
STEP 2: Check Referenced Content.
Question: Does the referenced statement or behavior contain explicit or implicit {CATEGORY}?
(The CATEGORY is indicated in the square brackets at the start of the post)
Output only YES or NO.
"""

STEP3_INSTRUCTIONS = LINGO_SYSTEM_PROMPT_BASE + """
STEP 3: Stance Toward Referenced Content.
Question: How does the author respond to the referenced {CATEGORY}?
Choose exactly one:
- REPORT: mentions without opinion
- INTENSIFY: agrees or amplifies (e.g., "Exato!", supportive emojis)
- COUNTER: criticizes or disagrees (e.g., "Isso é errado", "Que absurdo")
- ESCALATE: responds to {CATEGORY} content with {CATEGORY}
Output only one of: REPORT / INTENSIFY / COUNTER / ESCALATE.
"""

STEP4_INSTRUCTIONS = LINGO_SYSTEM_PROMPT_BASE + """
STEP 4: Check Original Content.
Question: Does the author's own text (not quoted/referenced content) contain explicit or implicit {CATEGORY}?
Output only YES or NO.
"""

STEP5_INSTRUCTIONS = LINGO_SYSTEM_PROMPT_BASE + """
STEP 5: Type Classification.
Question: Is the {CATEGORY} in the author's own text expressed directly or indirectly?
- EXPLICIT: direct, overt expression (clear insults, vulgarity, direct threats)
- IMPLICIT: indirect, veiled expression (sarcasm, irony, subtle attacks)
Output only EXPLICIT or IMPLICIT.
"""

## Step attribute names mapping
STEP_ATTR_NAMES = {
    "STEP 1": "refers",
    "STEP 2": "referenced_has_category",
    "STEP 3": "stance",
    "STEP 4": "original_has_category",
    "STEP 5": "directness",
}

## Define multi-step signatures


class Step1_CheckReference(dspy.Signature):
    __doc__ = STEP1_INSTRUCTIONS 
    
    post: str = dspy.InputField()
    refers: YESNO = dspy.OutputField()


class Step2_CheckReferencedContent(dspy.Signature):
    __doc__ = STEP2_INSTRUCTIONS  
    
    post: str = dspy.InputField()
    referenced_has_category: YESNO = dspy.OutputField()


class Step3_Stance(dspy.Signature):
    __doc__ = STEP3_INSTRUCTIONS  
    
    post: str = dspy.InputField()
    stance: STANCE = dspy.OutputField()  


class Step4_CheckOriginalContent(dspy.Signature):
    __doc__ = STEP4_INSTRUCTIONS  
    
    post: str = dspy.InputField()
    original_has_category: YESNO = dspy.OutputField()


class Step5_TypeClassification(dspy.Signature):
    __doc__ = STEP5_INSTRUCTIONS  
    
    post: str = dspy.InputField()
    directness: DIRECTNESS = dspy.OutputField() 

## Utility functions for data processing 

def _strip_parentheses(text: str) -> str:
    """Remove any (...) blocks."""
    return re.sub(r"\([^)]*\)", "", text or "")


def _normalize_ans(text: str) -> str:
    """Normalize whitespace and dashes."""
    t = _strip_parentheses(text)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("–", "-").replace("—", "-")
    return t


def extract_category(content: str) -> str:
    """Extract the category from the first bracket tag in the content."""
    match = re.match(r'\[([^\]]+)\]', content.strip())
    return match.group(1) if match else "Unknown"


def extract_text(content: str) -> str:
    """Extract the actual post text (without the category tag)."""
    return re.sub(r'^\[[^\]]+\]\s*', '', content.strip())


def extract_step_answers(reasoning_text: str) -> Dict[str, str]:
    """
    Extract step answers from reasoning text.
    Supports JSON format and legacy bracket/arrow format.
    """
    if not reasoning_text:
        return {}

    text = str(reasoning_text).strip()
    answers: Dict[str, str] = {}

    # Try JSON parsing first
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None

    if isinstance(obj, dict):
        reasoning_obj = obj.get("REASONING", None)
        if reasoning_obj is None and any(k in obj for k in STEP_KEYS):
            reasoning_obj = obj

        if isinstance(reasoning_obj, dict):
            for step in STEP_KEYS:
                v = reasoning_obj.get(step, None)
                if v is None:
                    continue
                s = str(v).strip()
                if not s or s.lower() in {"n/a", "na", "none", "null"}:
                    continue
                answers[step] = _normalize_ans(s)
            return answers

    # Legacy format parsing
    legacy = text
    if "REASONING:" in legacy:
        legacy = legacy.split("REASONING:", 1)[-1]

    bracket = re.search(r"\[(.*)\]", legacy, flags=re.DOTALL)
    if bracket:
        legacy = bracket.group(1)

    for step in STEP_KEYS:
        m = re.search(rf"{step}\s*:\s*(.*?)(?:→|$|\])", legacy, flags=re.IGNORECASE | re.DOTALL)
        if m:
            ans = _normalize_ans(m.group(1))
            ans = re.sub(r"LABEL\s*:\s*\d+\s*$", "", ans, flags=re.IGNORECASE).strip()
            if ans:
                answers[step] = ans

    return answers


def _normalize_step_answer(answer: str) -> str:
    """Normalize step answer for pattern matching."""
    if answer is None:
        return "MISSING"

    ans_raw = str(answer).strip()
    if not ans_raw or ans_raw.lower() in {"n/a", "na", "none", "null", ""}:
        return "MISSING"

    ans = ans_raw.upper().strip()
    if ans in ["YES", "SIM", "Y", "S"]:
        return "YES"
    elif ans in ["NO", "NÃO", "NAO", "N"]:
        return "NO"
    elif "EXPLICIT" in ans or "EXPLÍCIT" in ans:
        return "EXPLICIT"
    elif "IMPLICIT" in ans or "IMPLÍCIT" in ans:
        return "IMPLICIT"
    elif "REPORT" in ans or "RELAT" in ans:
        return "REPORT"
    elif "INTENSIF" in ans:
        return "INTENSIFY"
    elif "COUNTER" in ans or "CONTRA" in ans:
        return "COUNTER"
    elif "ESCALAT" in ans:
        return "ESCALATE"
    else:
        return ans


def earliest_step_diff(pred_steps: Dict[str, str], gold_steps: Dict[str, str],
                       pred_label: int, gold_label: int) -> Optional[str]:
    """Return the earliest step name where pred != gold, else None."""
    for step in STEP_KEYS:
        if step in pred_steps or step in gold_steps:
            pv = _normalize_step_answer(pred_steps.get(step, ""))
            gv = _normalize_step_answer(gold_steps.get(step, ""))
            if (pv != gv) and (pred_label != gold_label):
                return step
    return None


def split_train_val(df: pd.DataFrame, val_ratio: float = 0.2,
                    seed: int = 42, stratify_col: str = "intention_label") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into training and validation sets."""
    if stratify_col and stratify_col in df.columns:
        stratify = df[stratify_col]
    else:
        stratify = None

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


## Step-specific DSPy modules

class Step1Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Step1_CheckReference)

    def forward(self, post: str):
        result = self.predict(post=post)
        return dspy.Prediction(refers=result.refers)


class Step2Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Step2_CheckReferencedContent)

    def forward(self, post: str):
        result = self.predict(post=post)
        return dspy.Prediction(referenced_has_category=result.referenced_has_category)


class Step3Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Step3_Stance)

    def forward(self, post: str):
        result = self.predict(post=post)
        return dspy.Prediction(stance=result.stance)


class Step4Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Step4_CheckOriginalContent)

    def forward(self, post: str):
        result = self.predict(post=post)
        return dspy.Prediction(original_has_category=result.original_has_category)


class Step5Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Step5_TypeClassification)

    def forward(self, post: str):
        result = self.predict(post=post)
        return dspy.Prediction(directness=result.directness)


STEP_MODULES = {
    "STEP 1": Step1Module,
    "STEP 2": Step2Module,
    "STEP 3": Step3Module,
    "STEP 4": Step4Module,
    "STEP 5": Step5Module,
}


## Step-specific metrics

def step1_metric(pred, gold, trace=None):
    try:
        return _normalize_step_answer(str(pred.refers)) == _normalize_step_answer(str(gold.refers))
    except:
        return False


def step2_metric(pred, gold, trace=None):
    try:
        return _normalize_step_answer(str(pred.referenced_has_category)) == _normalize_step_answer(str(gold.referenced_has_category))
    except:
        return False


def step3_metric(pred, gold, trace=None):
    try:
        return _normalize_step_answer(str(pred.stance)) == _normalize_step_answer(str(gold.stance))
    except:
        return False


def step4_metric(pred, gold, trace=None):
    try:
        return _normalize_step_answer(str(pred.original_has_category)) == _normalize_step_answer(str(gold.original_has_category))
    except:
        return False


def step5_metric(pred, gold, trace=None):
    try:
        return _normalize_step_answer(str(pred.directness)) == _normalize_step_answer(str(gold.directness))
    except:
        return False


STEP_METRICS = {
    "STEP 1": step1_metric,
    "STEP 2": step2_metric,
    "STEP 3": step3_metric,
    "STEP 4": step4_metric,
    "STEP 5": step5_metric,
}


## Full Pipeline with Optimized Steps

class OptimizedLingoIntentPipeline(dspy.Module):
    """
    Multi-step pipeline that uses optimized modules for targeted steps
    and default modules for non-targeted steps.
    Returns both final label AND step-by-step reasoning.
    """

    def __init__(self, optimized_modules: Dict[str, dspy.Module] = None):
        super().__init__()
        self.optimized_modules = optimized_modules or {}

        self._default_s1 = dspy.Predict(Step1_CheckReference)
        self._default_s2 = dspy.Predict(Step2_CheckReferencedContent)
        self._default_s3 = dspy.Predict(Step3_Stance)
        self._default_s4 = dspy.Predict(Step4_CheckOriginalContent)
        self._default_s5 = dspy.Predict(Step5_TypeClassification)

    def _get_step_result(self, step_name: str, post: str) -> str:
        """Get result for a step using optimized or default module."""
        attr_name = STEP_ATTR_NAMES.get(step_name)

        if step_name in self.optimized_modules:
            result = self.optimized_modules[step_name](post=post)
            return str(getattr(result, attr_name, "MISSING"))

        predictor_map = {
            "STEP 1": self._default_s1,
            "STEP 2": self._default_s2,
            "STEP 3": self._default_s3,
            "STEP 4": self._default_s4,
            "STEP 5": self._default_s5,
        }
        predictor = predictor_map.get(step_name)
        if predictor:
            result = predictor(post=post)
            return str(getattr(result, attr_name, "MISSING"))
        return "MISSING"

    def forward(self, post: str):
        """
        Run the multi-step pipeline and return label + reasoning.
        Follows the exact branching logic from the annotation guidelines.
        """
        step_results = {}

        # STEP 1: Check Reference
        s1 = _normalize_step_answer(self._get_step_result("STEP 1", post))
        step_results["STEP 1"] = s1

        if s1 == "YES":
            # STEP 2: Check Referenced Content
            s2 = _normalize_step_answer(self._get_step_result("STEP 2", post))
            step_results["STEP 2"] = s2

            if s2 == "YES":
                # STEP 3: Stance
                s3 = _normalize_step_answer(self._get_step_result("STEP 3", post))
                step_results["STEP 3"] = s3

                label_map = {"REPORT": 3, "INTENSIFY": 4, "COUNTER": 5, "ESCALATE": 6}
                label = label_map.get(s3, 3)

                reasoning = json.dumps({"STEP 1": s1, "STEP 2": s2, "STEP 3": s3})
                return dspy.Prediction(label=label, reasoning=reasoning, step_results=step_results)

        # Path: S1=NO or S2=NO -> STEP 4
        s4 = _normalize_step_answer(self._get_step_result("STEP 4", post))
        step_results["STEP 4"] = s4

        if s4 == "NO":
            label = 0
            reasoning_dict = {"STEP 1": s1}
            if "STEP 2" in step_results:
                reasoning_dict["STEP 2"] = step_results["STEP 2"]
            reasoning_dict["STEP 4"] = s4
            reasoning = json.dumps(reasoning_dict)
            return dspy.Prediction(label=label, reasoning=reasoning, step_results=step_results)

        # STEP 5: Type Classification
        s5 = _normalize_step_answer(self._get_step_result("STEP 5", post))
        step_results["STEP 5"] = s5

        label = 1 if s5 == "EXPLICIT" else 2

        reasoning_dict = {"STEP 1": s1}
        if "STEP 2" in step_results:
            reasoning_dict["STEP 2"] = step_results["STEP 2"]
        reasoning_dict["STEP 4"] = s4
        reasoning_dict["STEP 5"] = s5
        reasoning = json.dumps(reasoning_dict)

        return dspy.Prediction(label=label, reasoning=reasoning, step_results=step_results)


## Step difference analysis (same as RAG code)

def compute_step_diff_distribution(df: pd.DataFrame) -> Tuple[Counter, List[Dict[str, Any]]]:
    """
    Compute distribution of earliest differing steps between predictions and gold.
    Uses gold 'reason' column from CSV.
    """
    dist = Counter()
    records = []

    for i, row in df.iterrows():
        gold_label = int(row.get("intention_label", 0))
        pred_label = int(row.get("predicted_label", 0))

        # Extract step answers from gold reason (from CSV) and predicted reasoning
        gold_steps = extract_step_answers(str(row.get("reason", "")))
        pred_steps = extract_step_answers(str(row.get("reasoning", "")))

        diff_step = earliest_step_diff(pred_steps, gold_steps, pred_label, gold_label)
        if diff_step is not None:
            dist[diff_step] += 1
            records.append({
                "index": i,
                "diff_step": diff_step,
                "gold_steps": gold_steps,
                "pred_steps": pred_steps,
                "gold_step_answer": gold_steps.get(diff_step, ""),
                "pred_step_answer": pred_steps.get(diff_step, ""),
                "gold_label": gold_label,
                "pred_label": pred_label,
                "content": row.get("content", ""),
                "gold_reason": row.get("reason", ""),
                "pred_reason": row.get("reasoning", "")
            })

    return dist, records


def analyze_step_error_patterns(step_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze error patterns for each step to understand what the model is getting wrong.
    """
    step_errors: Dict[str, List[Dict[str, Any]]] = {}

    for rec in step_records:
        step = rec["diff_step"]
        if step not in step_errors:
            step_errors[step] = []
        step_errors[step].append(rec)

    analysis = {}
    for step, errors in step_errors.items():
        gold_answers = Counter()
        pred_answers = Counter()
        error_patterns = Counter()

        for err in errors:
            gold_ans = _normalize_step_answer(err.get("gold_step_answer", ""))
            pred_ans = _normalize_step_answer(err.get("pred_step_answer", ""))

            gold_answers[gold_ans] += 1
            pred_answers[pred_ans] += 1
            error_patterns[f"{pred_ans}->{gold_ans}"] += 1

        analysis[step] = {
            "total_errors": len(errors),
            "gold_answer_distribution": dict(gold_answers),
            "pred_answer_distribution": dict(pred_answers),
            "error_patterns": dict(error_patterns),
            "most_common_gold": gold_answers.most_common(3),
            "most_common_errors": error_patterns.most_common(5),
            "error_records": errors
        }

    return analysis


def identify_problematic_steps_with_patterns(
    step_diff_distribution: Counter,
    step_records: List[Dict[str, Any]],
    threshold: float = 0.1
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Identify problematic steps and analyze their error patterns.
    Same logic as RAG code.
    """
    total = sum(step_diff_distribution.values())
    if total == 0:
        return [], {}

    problematic = [
        (step, count) for step, count in step_diff_distribution.items()
        if (count / total) >= threshold
    ]
    problematic.sort(key=lambda x: x[1], reverse=True)
    problematic_steps = [step for step, _ in problematic]

    error_analysis = analyze_step_error_patterns(step_records)

    filtered_analysis = {
        step: analysis for step, analysis in error_analysis.items()
        if step in problematic_steps
    }

    return problematic_steps, filtered_analysis


## Create step-specific examples from training data using gold 'reason' column

def create_step_examples_from_gold(
    step_name: str,
    df: pd.DataFrame,
    error_analysis: Dict[str, Dict[str, Any]] = None
) -> List[dspy.Example]:
    """
    Create DSPy examples for a specific step using gold 'reason' column from CSV.
    Prioritizes examples that match common error patterns (to teach correct answers).
    """
    examples = []
    attr_name = STEP_ATTR_NAMES.get(step_name)

    if attr_name is None:
        return []

    # Get target gold answers from error patterns (for prioritization)
    target_gold_answers = set()
    if error_analysis and step_name in error_analysis:
        common_errors = error_analysis[step_name].get("most_common_errors", [])
        for pattern, count in common_errors[:5]:
            if "->" in pattern:
                pred_ans, gold_ans = pattern.split("->", 1)
                target_gold_answers.add(_normalize_step_answer(gold_ans))

    priority_examples = []
    other_examples = []

    for _, row in df.iterrows():
        reason = str(row.get("reason", ""))
        gold_steps = extract_step_answers(reason)

        # Only include examples that have this step in their reasoning
        if step_name not in gold_steps:
            continue

        gold_answer = gold_steps[step_name]
        normalized_answer = _normalize_step_answer(gold_answer)

        # Skip invalid answers
        if normalized_answer == "MISSING":
            continue

        ex = dspy.Example(
            post=str(row.get('content', '')),
            **{attr_name: normalized_answer}
        ).with_inputs("post")

        # Prioritize examples that match error patterns
        if normalized_answer in target_gold_answers:
            priority_examples.append(ex)
        else:
            other_examples.append(ex)

    # Return with priority examples first
    return priority_examples + other_examples


## DSPy Step Optimizer

class DSPyStepOptimizer:
    """
    Manages DSPy optimization for targeted steps only.
    """

    def __init__(
        self,
        forward_model: str,
        max_labeled_demos: int = 30,
        max_bootstrapped_demos: int = 20,
        num_candidate_programs: int = 10,
        max_rounds: int = 1,
        num_threads: int = 1,
    ):
        self.forward_model = forward_model
        self.max_labeled_demos = max_labeled_demos
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.num_candidate_programs = num_candidate_programs
        self.max_rounds = max_rounds
        self.num_threads = num_threads

        self.optimized_modules: Dict[str, dspy.Module] = {}

        # Configure DSPy
        self.lm = dspy.LM(forward_model, api_key=self._get_api_key(forward_model))
        dspy.settings.configure(lm=self.lm)

    def _get_api_key(self, model: str) -> str:
        if "anthropic" in model.lower() or "claude" in model.lower():
            return os.environ.get("ANTHROPIC_API_KEY", "")
        elif "gemini" in model.lower():
            return os.environ.get("GEMINI_API_KEY", "")
        else:
            return os.environ.get("OPENAI_API_KEY", "")

    def optimize_step(
        self,
        step_name: str,
        train_examples: List[dspy.Example],
        val_examples: List[dspy.Example]
    ) -> Optional[dspy.Module]:
        """
        Optimize a specific step using DSPy BootstrapFewShotWithRandomSearch.
        """
        print(f"\n{'='*50}")
        print(f"Optimizing {step_name} with DSPy...")
        print(f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}")
        print(f"{'='*50}")

        if len(train_examples) < 3 or len(val_examples) < 2:
            print(f"Not enough examples for {step_name}, skipping optimization")
            return None

        module_class = STEP_MODULES.get(step_name)
        if module_class is None:
            print(f"Unknown step: {step_name}")
            return None

        base_module = module_class()
        step_metric = STEP_METRICS.get(step_name)

        optimizer = BootstrapFewShotWithRandomSearch(
            metric=step_metric,
            max_labeled_demos=min(self.max_labeled_demos, len(train_examples)),
            max_bootstrapped_demos=min(self.max_bootstrapped_demos, len(train_examples) // 2),
            num_candidate_programs=self.num_candidate_programs,
            max_rounds=self.max_rounds,
            num_threads=self.num_threads,
        )

        try:
            optimized_module = optimizer.compile(
                base_module,
                trainset=train_examples,
                valset=val_examples
            )

            self.optimized_modules[step_name] = optimized_module
            print(f"Successfully optimized {step_name}")

            return optimized_module

        except Exception as e:
            print(f"Error optimizing {step_name}: {e}")
            return None


## Evaluation function (similar to RAG code)

def evaluate_pipeline_on_dataset(
    pipeline: OptimizedLingoIntentPipeline,
    df: pd.DataFrame,
    desc: str = "Evaluating"
) -> Dict[str, Any]:
    """
    Run predictions and compute metrics including step-level analysis.
    Matches the structure of RAG code's evaluate_on_dataset.
    """
    preds, reasons = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        content = str(row.get('content', ''))
        try:
            result = pipeline(post=content)
            preds.append(int(result.label))
            reasons.append(result.reasoning)
        except Exception as e:
            print(f"Error: {e}")
            preds.append(0)
            reasons.append('{"error": "prediction failed"}')

    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_label'] = preds
    results_df['reasoning'] = reasons

    # Calculate metrics
    y_true = results_df['intention_label'].tolist()
    y_pred = results_df['predicted_label'].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Step difference analysis
    dist, recs = compute_step_diff_distribution(results_df)
    total = len(results_df)
    diff_total = sum(dist.values())
    step_diff_rate = diff_total / total if total else 0.0

    return {
        "results_df": results_df,
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
        },
        "step_diff_distribution": dict(dist),
        "step_records": recs,
        "step_diff_rate": step_diff_rate,
    }


## Main iterative refinement (same structure as RAG code)

def iterative_dspy_refinement(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forward_model: str = "openai/gpt-4o-mini",
    rounds: int = 5,
    seed: int = 42,
    step_threshold: float = 0.1,
    val_ratio: float = 0.2,
    max_labeled_demos: int = 30,
    max_bootstrapped_demos: int = 20,
    num_candidate_programs: int = 10,
    max_rounds: int = 1,
    num_threads: int = 1,
) -> Dict[str, Any]:
    """
    Iteratively refine classification by:
    1. Run inference to get predictions with step reasoning
    2. Compare step answers against gold (from 'reason' column in CSV)
    3. Identify problematic steps (error rate > threshold)
    4. Optimize those steps with DSPy bootstrapping
    5. Repeat for N rounds
    
    This matches the RAG code structure exactly.
    """
    random.seed(seed)

    print("="*60)
    print("STEP 1: Splitting dev set into TRAIN and VAL")
    print("="*60)

    train_df, val_df = split_train_val(dev_df, val_ratio=val_ratio, seed=seed)

    print(f"  Original dev set: {len(dev_df)} samples")
    print(f"  TRAIN set: {len(train_df)} samples (for DSPy optimization)")
    print(f"  VAL set: {len(val_df)} samples (for step error analysis)")
    print(f"  TEST set: {len(test_df)} samples (final evaluation only)")

    print("\n" + "="*60)
    print("STEP 2: Initializing DSPy Step Optimizer")
    print(f"  Forward model: {forward_model}")
    print("="*60)

    step_optimizer = DSPyStepOptimizer(
        forward_model=forward_model,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        num_candidate_programs=num_candidate_programs,
        max_rounds=max_rounds,
        num_threads=num_threads,
    )

    print("\n" + "="*60)
    print("STEP 3: Iterative refinement using TRAIN and VAL sets")
    print("="*60)

    history: List[Dict[str, Any]] = []
    target_steps: List[str] = []
    error_analysis: Dict[str, Dict[str, Any]] = {}
    optimized_modules: Dict[str, dspy.Module] = {}

    val_eval = None
    best_val_f1 = -1
    best_optimized_modules = {}
    best_round = 0

    for r in trange(rounds + 1, desc="DSPy Refinement Rounds"):
        print(f"\n{'='*60}")
        print(f"Round {r}: Target steps = {target_steps if target_steps else 'None (baseline)'}")
        print('='*60)

        # Optimize target steps if any (same as RAG retrieves examples for target steps)
        if target_steps and r > 0:
            print(f"\nOptimizing {len(target_steps)} targeted steps with DSPy...")

            for step_name in target_steps:
                # Create examples from gold 'reason' column, prioritizing error patterns
                step_train_examples = create_step_examples_from_gold(
                    step_name, train_df, error_analysis
                )
                step_val_examples = create_step_examples_from_gold(
                    step_name, val_df, error_analysis=None
                )

                print(f"\n  {step_name}:")
                print(f"    Train examples: {len(step_train_examples)}")
                print(f"    Val examples: {len(step_val_examples)}")

                if len(step_train_examples) >= 5 and len(step_val_examples) >= 2:
                    optimized_module = step_optimizer.optimize_step(
                        step_name=step_name,
                        train_examples=step_train_examples,
                        val_examples=step_val_examples
                    )
                    if optimized_module:
                        optimized_modules[step_name] = optimized_module
                        print(f"    Status: OPTIMIZED")
                else:
                    print(f"    Status: Not enough examples, keeping DEFAULT")

        # Create pipeline with current optimized modules
        pipeline = OptimizedLingoIntentPipeline(optimized_modules=optimized_modules)

        print("\nCurrent pipeline structure:")
        for step in STEP_KEYS:
            status = "OPTIMIZED" if step in optimized_modules else "DEFAULT"
            print(f"  {step}: {status}")

        # Evaluate on VAL set
        print("\nEvaluating on VAL set...")
        val_eval = evaluate_pipeline_on_dataset(pipeline, val_df, desc="VAL")

        # Track best configuration
        current_f1 = val_eval["metrics"]["f1_weighted"]
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_optimized_modules = dict(optimized_modules)
            best_round = r

        # Save history
        history.append({
            "round": r,
            "target_steps": list(target_steps),
            "optimized_steps": list(optimized_modules.keys()),
            "error_patterns": {
                step: analysis.get("most_common_errors", [])
                for step, analysis in error_analysis.items()
            } if error_analysis else {},
            "val_accuracy": val_eval["metrics"]["accuracy"],
            "val_f1_macro": val_eval["metrics"]["f1_macro"],
            "val_f1_weighted": val_eval["metrics"]["f1_weighted"],
            "val_f1_micro": val_eval["metrics"]["f1_micro"],
            "val_step_diff_distribution": val_eval["step_diff_distribution"],
            "val_step_diff_rate": val_eval["step_diff_rate"],
        })

        print(f"VAL - Accuracy: {val_eval['metrics']['accuracy']:.4f}, "
              f"F1 (weighted): {val_eval['metrics']['f1_weighted']:.4f}")
        print(f"VAL step diff distribution: {val_eval['step_diff_distribution']}")

        if r == rounds:
            break

        # Identify problematic steps for next round (same as RAG code)
        val_dist_counter = Counter(val_eval["step_diff_distribution"])

        if sum(val_dist_counter.values()) == 0:
            print("No step differences found - stopping early")
            break

        target_steps, error_analysis = identify_problematic_steps_with_patterns(
            val_dist_counter,
            val_eval["step_records"],
            step_threshold
        )

        print(f"Identified problematic steps for next round: {target_steps}")
        for step, analysis in error_analysis.items():
            print(f"  {step} error patterns: {analysis.get('most_common_errors', [])[:3]}")

    # Final evaluation on TEST set
    print("\n" + "="*60)
    print(f"STEP 4: Final evaluation on TEST set")
    print(f"  Using best configuration from round {best_round}")
    print(f"  Optimized steps: {list(best_optimized_modules.keys())}")
    print("="*60)

    final_pipeline = OptimizedLingoIntentPipeline(optimized_modules=best_optimized_modules)

    print("\nFinal pipeline structure:")
    for step in STEP_KEYS:
        status = "OPTIMIZED" if step in best_optimized_modules else "DEFAULT"
        print(f"  {step}: {status}")

    print("\nEvaluating on TEST set...")
    test_eval = evaluate_pipeline_on_dataset(final_pipeline, test_df, desc="TEST")

    print(f"\nFINAL TEST RESULTS:")
    print(f"  Accuracy: {test_eval['metrics']['accuracy']:.4f}")
    print(f"  F1 (weighted): {test_eval['metrics']['f1_weighted']:.4f}")

    return {
        "history": history,
        "best_round": best_round,
        "best_optimized_steps": list(best_optimized_modules.keys()),
        "best_val_f1": best_val_f1,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "final_val_results": val_eval["results_df"] if val_eval else None,
        "final_test_results": test_eval["results_df"],
        "final_val_metrics": val_eval["metrics"] if val_eval else None,
        "final_test_metrics": test_eval["metrics"],
        "final_test_step_diff_distribution": test_eval["step_diff_distribution"],
    }


## Export functions

def export_dspy_artifacts(
    run_result: Dict[str, Any],
    out_dir: str = "lingo_dspy_outputs",
    history_csv_name: str = "history_metrics.csv"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    history = run_result.get("history", [])

    if not history:
        raise ValueError("No history found in run_result. Did the run complete?")

    # Save history
    csv_path = os.path.join(out_dir, history_csv_name)
    rows = []
    for h in history:
        rows.append({
            "round": h["round"],
            "target_steps": ", ".join(h["target_steps"]) if h["target_steps"] else "None",
            "optimized_steps": ", ".join(h.get("optimized_steps", [])) if h.get("optimized_steps") else "None",
            "val_accuracy": h["val_accuracy"],
            "val_f1_macro": h["val_f1_macro"],
            "val_f1_weighted": h["val_f1_weighted"],
            "val_f1_micro": h["val_f1_micro"],
            "val_step_diff_rate": h.get("val_step_diff_rate", 0),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved metrics history to {csv_path}")

    # Save test predictions
    if run_result.get("final_test_results") is not None:
        test_path = os.path.join(out_dir, "DSPy_LinGO_predictions.csv")
        run_result["final_test_results"].to_csv(test_path, index=False, encoding="utf-8")
        print(f"Saved test predictions to {test_path}")

    # Save summary report
    summary_path = os.path.join(out_dir, "summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("LinGO DSPy Targeted Step Optimization Summary\n")
        f.write("="*60 + "\n\n")

        f.write("OPTIMIZATION APPROACH:\n")
        f.write("  - Run inference to get step-by-step predictions\n")
        f.write("  - Compare against gold step answers from 'reason' column\n")
        f.write("  - Identify steps with error rate > threshold\n")
        f.write("  - Bootstrap examples for those steps using DSPy\n")
        f.write("  - Repeat for N rounds\n\n")

        f.write("DATA SPLIT:\n")
        f.write(f"  TRAIN: {run_result.get('train_size', 'N/A')} samples\n")
        f.write(f"  VAL: {run_result.get('val_size', 'N/A')} samples\n")
        f.write(f"  TEST: {run_result.get('test_size', 'N/A')} samples\n\n")

        f.write(f"Best Round: {run_result.get('best_round', 'N/A')}\n")
        f.write("Optimized Steps: " +
                (", ".join(run_result.get("best_optimized_steps", [])) or "None") + "\n")
        f.write(f"Best VAL F1 (weighted): {run_result.get('best_val_f1', 0):.4f}\n\n")

        f.write("FINAL PIPELINE STRUCTURE:\n")
        optimized = run_result.get("best_optimized_steps", [])
        for step in STEP_KEYS:
            status = "OPTIMIZED" if step in optimized else "DEFAULT"
            f.write(f"  {step}: {status}\n")

        f.write("\nVAL Metrics by Round:\n")
        f.write("-"*60 + "\n")
        for h in history:
            f.write(f"\nRound {h['round']}:\n")
            f.write(f"  Target Steps: {h['target_steps'] or 'None (baseline)'}\n")
            f.write(f"  Optimized Steps: {h.get('optimized_steps', [])}\n")
            f.write(f"  VAL - Acc: {h['val_accuracy']:.4f}, F1-weighted: {h['val_f1_weighted']:.4f}\n")
            f.write(f"  Step Diff Distribution: {h['val_step_diff_distribution']}\n")
            if h.get('error_patterns'):
                f.write(f"  Error Patterns:\n")
                for step, patterns in h['error_patterns'].items():
                    f.write(f"    {step}: {patterns[:2]}\n")

        f.write("\n" + "="*60 + "\n")
        f.write("FINAL TEST RESULTS:\n")
        f.write("="*60 + "\n")
        test_metrics = run_result.get("final_test_metrics", {})
        f.write(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  F1 (weighted): {test_metrics.get('f1_weighted', 0):.4f}\n")
        f.write(f"  F1 (macro): {test_metrics.get('f1_macro', 0):.4f}\n")
        f.write(f"  Step Diff Distribution: {run_result.get('final_test_step_diff_distribution', {})}\n")

    print(f"Saved summary report to {summary_path}")

    return out_dir


def run_dspy_optimization(
    dev_filepath: str,
    test_filepath: str,
    forward_model: str = "openai/gpt-4o-mini",
    rounds: int = 5,
    step_threshold: float = 0.1,
    val_ratio: float = 0.2,
    seed: int = 42,
    output_dir: str = "lingo_dspy_outputs",
    max_labeled_demos: int = 30,
    max_bootstrapped_demos: int = 20,
    num_candidate_programs: int = 10,
    max_rounds: int = 1,
    num_threads: int = 1,
) -> Dict[str, Any]:
    """Main entry point for DSPy targeted step optimization."""
    print(f"Loading data...")
    dev_df = pd.read_csv(dev_filepath)
    test_df = pd.read_csv(test_filepath)
    print(f"  DEV (to be split): {len(dev_df)} samples")
    print(f"  TEST (held out): {len(test_df)} samples")

    result = iterative_dspy_refinement(
        dev_df=dev_df,
        test_df=test_df,
        forward_model=forward_model,
        rounds=rounds,
        seed=seed,
        step_threshold=step_threshold,
        val_ratio=val_ratio,
        max_labeled_demos=max_labeled_demos,
        max_bootstrapped_demos=max_bootstrapped_demos,
        num_candidate_programs=num_candidate_programs,
        max_rounds=max_rounds,
        num_threads=num_threads,
    )

    export_dspy_artifacts(result, out_dir=output_dir)

    return result


## Main function call

if __name__ == "__main__":

    result = run_dspy_optimization(
        dev_filepath="./incivi_dev.csv",
        test_filepath="./incivi_test.csv",
        forward_model="gemini/gemini-2.0-flash-lite",  # or "gemini/gemini-2.0-flash-lite", "anthropic/claude-3-haiku-20240307"
        rounds=5,
        step_threshold=0.1,  # Steps with >10% of errors are targeted
        val_ratio=0.2,       # 20% of dev for validation
        seed=42,
        output_dir="DSPy_LinGO_Gemini_outputs",
        max_labeled_demos=6,
        max_bootstrapped_demos=4,
        num_candidate_programs=10,
        max_rounds=1,
        num_threads=1,
    )

## Results: Claude (Test Accuracy: 0.4100; Test Weighted F1:  0.4583); Gemini (Test Accuracy: 0.6475; Test Weighted F1:  0.6645); GPT (Test Accuracy: 0.5075; Test Weighted F1:  0.5538)
