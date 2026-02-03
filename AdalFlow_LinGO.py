import re
import time
import random
import json
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Callable
from tqdm import trange
from litellm import completion
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)
from dataclasses import dataclass, field

import adalflow as adal
from adalflow.optim.types import ParameterType
from adalflow.components.model_client import GoogleGenAIClient, OpenAIClient, AnthropicAPIClient

os.environ["OPENAI_API_KEY"] = "YOUR API KEY"
os.environ["GOOGLE_API_KEY"] = "YOUR API KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

STEP_KEYS = ["STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5"]

LINGO_SYSTEM_PROMPT_TEMPLATE = """You are an annotator interpreting the intent of uncivil Portuguese social media discourse about Brazilian politics on X (formerly Twitter).

Target Category: Always taken from the first square-bracket tag (e.g., [Impoliteness], [Hate Speech and Stereotyping], [Physical Harm and Violent Political Rhetoric], [Threats to Democratic Institutions and Values])

Category Definitions:
- Impoliteness: messages including rudeness/disrespect (name-calling, aspersions, calling someone a liar, hyperbole, non-cooperation, pejoratives, vulgarity, belittling, shouting via ALL-CAPS or excessive "!" when context supports it).
- Hate Speech and Stereotyping: discriminatory content targeting protected groups (gender identity, sexual orientation, religion, race, nationality, ideology, disability); over-generalizations, out-group demeaning; (do NOT include purely individual appearance insults unless tied to group identity).
- Physical Harm and Violent Political Rhetoric: threats/advocacy/praise of physical harm or violence; direct or metaphorical calls for harm; justification of violence for political ends.
- Threats to Democratic Institutions and Values: advocacy or approval of actions undermining elections/institutions/rule of law/press freedom/civil rights; promotion of autocracy; unfounded claims that delegitimize institutions.

Decision Process:
{step_1_definition}

{step_2_definition}

{step_3_definition}

{step_4_definition}

{step_5_definition}

{output_format}

{step_specific_examples}

{corrective_examples}
"""

DEFAULT_STEP_DEFINITIONS = {
    "STEP 1": """STEP 1: Check Reference.
Question: Does the text refer to another person's statement or behavior?
If NO → go to STEP 4 (Original Content).
If YES → go to STEP 2 (Referenced Content).""",

    "STEP 2": """STEP 2: Check Referenced Content.
Question: Does the referenced statement or behavior contain explicit or implicit {{CATEGORY}}?
If NO → go to STEP 4 (Analyze Original Content).
If YES → go to STEP 3 (Stance Analysis).""",

    "STEP 3": """STEP 3: Stance Toward Referenced Content.
Question: How does the author respond to the referenced {{CATEGORY}}?
Report (3): mentions without opinion.
Intensify (4): agrees or amplifies (e.g., "Exato!", supportive emojis).
Counter (5): criticizes or disagrees (e.g., "Isso é errado", "Que absurdo").
Escalate (6): responds to {{CATEGORY}} content with {{CATEGORY}}.""",

    "STEP 4": """STEP 4: Check Original Content.
Question: Does the author's own text contain explicit or implicit {{CATEGORY}}?
If NO → Label 0 (No {{CATEGORY}}).
If YES → go to STEP 5 (Type Classification).""",

    "STEP 5": """STEP 5: Type Classification (Direct Content).
Question: Is the {{CATEGORY}} expressed directly or indirectly?
Explicit (1): direct, overt {{CATEGORY}} (e.g., clear insults, vulgarity, direct threats).
Implicit (2): indirect, veiled {{CATEGORY}} (e.g., sarcasm, irony, subtle attacks).""",
}

DEFAULT_OUTPUT_FORMAT = """Return ONLY valid JSON in this format:
{
  "LABEL": <integer 0-6>,
  "REASONING": {
    "STEP 1": "YES" or "NO",
    <then include ONLY the steps you actually followed based on the branching logic>
  }

Examples of correct output format:

Example 1 (Path: STEP 1=NO → STEP 4=YES → STEP 5):
{"LABEL": 1, "REASONING": {"STEP 1": "NO", "STEP 4": "YES", "STEP 5": "Explicit"}}

Example 2 (Path: STEP 1=YES → STEP 2=YES → STEP 3):
{"LABEL": 5, "REASONING": {"STEP 1": "YES", "STEP 2": "YES", "STEP 3": "Counter"}}

Example 3 (Path: STEP 1=YES → STEP 2=NO → STEP 4=NO):
{"LABEL": 0, "REASONING": {"STEP 1": "YES", "STEP 2": "NO", "STEP 4": "NO"}}

Example 4 (Path: STEP 1=NO → STEP 4=NO):
{"LABEL": 0, "REASONING": {"STEP 1": "NO", "STEP 4": "NO"}}
}"""

LABEL_DESCRIPTIONS = {
    0: "Other - does not fit any pattern",
    1: "Explicit - direct, overt expression",
    2: "Implicit - indirect, veiled expression",
    3: "Report - quotes/refers without opinion",
    4: "Intensify - quotes and agrees/amplifies",
    5: "Counter - quotes and criticizes/disagrees",
    6: "Escalate - responds with more of the same",
}


def _strip_parentheses(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", text or "")


def _normalize_ans(text: str) -> str:
    t = _strip_parentheses(text)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("–", "-").replace("—", "-")
    return t


def extract_category(content: str) -> str:
    match = re.match(r"\[([^\]]+)\]", content.strip())
    return match.group(1) if match else "Unknown"


def extract_text(content: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s*", "", content.strip())


def extract_step_answers(reasoning_text: str) -> Dict[str, str]:
    if not reasoning_text:
        return {}

    text = str(reasoning_text).strip()

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
            answers: Dict[str, str] = {}
            for step in STEP_KEYS:
                v = reasoning_obj.get(step, None)
                if v is None:
                    continue
                s = str(v).strip()
                if not s or s.lower() in {"n/a", "na", "none", "null"}:
                    continue
                answers[step] = _normalize_ans(s)
            return answers

    legacy = text
    if "REASONING:" in legacy:
        legacy = legacy.split("REASONING:", 1)[-1]

    bracket = re.search(r"\[(.*)\]", legacy, flags=re.DOTALL)
    if bracket:
        legacy = bracket.group(1)

    answers: Dict[str, str] = {}
    for step in STEP_KEYS:
        m = re.search(rf"{step}\s*:\s*(.*?)(?:→|$|\])", legacy, flags=re.IGNORECASE | re.DOTALL)
        if m:
            ans = _normalize_ans(m.group(1))
            ans = re.sub(r"LABEL\s*:\s*\d+\s*$", "", ans, flags=re.IGNORECASE).strip()
            if ans:
                answers[step] = ans

    return answers


def earliest_step_diff(
    pred_steps: Dict[str, str],
    gold_steps: Dict[str, str],
    pred_label: int,
    gold_label: int,
) -> Optional[str]:
    for step in STEP_KEYS:
        if step in pred_steps or step in gold_steps:
            pv = pred_steps.get(step, "")
            gv = gold_steps.get(step, "")
            if (pv != gv) and (pred_label != gold_label):
                return step
    return None


def _normalize_step_answer(answer: str) -> str:
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


def split_train_val(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify_col: str = "intention_label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if stratify_col and stratify_col in df.columns:
        stratify = df[stratify_col]
    else:
        stratify = None

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def filter_df_to_target_steps(df: pd.DataFrame, target_steps: List[str]) -> pd.DataFrame:
    """
    Restrict example pool to ONLY samples that contain at least one of the target steps
    in their gold reasoning (df['reason']). This ensures k_step_examples / k_error_examples
    are sourced from targeted-step-specific samples only.
    """
    if df is None or df.empty or not target_steps:
        return df.iloc[0:0].copy()

    keep_rows = []
    for _, row in df.iterrows():
        reason = str(row.get("reason", ""))
        step_answers = extract_step_answers(reason)
        keep_rows.append(any(s in step_answers for s in target_steps))

    return df.loc[keep_rows].reset_index(drop=True)


def get_step_targeted_examples(
    target_steps: List[str],
    df: pd.DataFrame,
    k_per_step: int = 3,
) -> str:
    if not target_steps or df is None or df.empty:
        return ""

    selected_examples = []
    seen_indices = set()

    for step in target_steps:
        step_examples = []
        seen_answers = set()

        for idx, row in df.iterrows():
            if idx in seen_indices:
                continue

            content = row.get("content", "")
            reason = str(row.get("reason", ""))
            label = int(row.get("intention_label", 0))
            step_answers = extract_step_answers(reason)

            if step in step_answers:
                answer = step_answers[step]
                normalized = _normalize_step_answer(answer)

                if normalized not in seen_answers:
                    step_examples.append(
                        {
                            "idx": idx,
                            "content": content,
                            "label": label,
                            "reason": reason,
                            "step": step,
                            "step_answer": answer,
                        }
                    )
                    seen_answers.add(normalized)

                    if len(step_examples) >= k_per_step:
                        break

        for ex in step_examples:
            if ex["idx"] not in seen_indices:
                selected_examples.append(ex)
                seen_indices.add(ex["idx"])

    if not selected_examples:
        return ""

    formatted_parts = []
    for ex in selected_examples:
        formatted_parts.append(
            f'[{ex["step"]} Example - Answer: {ex["step_answer"]}]\n'
            f'Post: "{extract_text(ex["content"])}"\n'
            f'Category: [{extract_category(ex["content"])}]\n'
            f'Label: {ex["label"]}\n'
            f'Reasoning: {ex["reason"]}'
        )

    return "\n\n---\n\n".join(formatted_parts)


def get_error_pattern_examples(
    error_analysis: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    k_per_pattern: int = 2,
) -> str:
    if not error_analysis or df is None or df.empty:
        return ""

    all_examples = []
    seen_indices = set()

    for step, analysis in error_analysis.items():
        common_errors = analysis.get("most_common_errors", [])

        for pattern, _count in common_errors[:3]:
            if "->" not in pattern:
                continue
            _pred_ans, gold_ans = pattern.split("->", 1)
            normalized_gold = _normalize_step_answer(gold_ans)

            pattern_examples = []
            for idx, row in df.iterrows():
                if idx in seen_indices:
                    continue

                content = row.get("content", "")
                reason = str(row.get("reason", ""))
                label = int(row.get("intention_label", 0))
                step_answers = extract_step_answers(reason)

                if step in step_answers:
                    answer = step_answers[step]
                    if _normalize_step_answer(answer) == normalized_gold:
                        pattern_examples.append(
                            {
                                "idx": idx,
                                "content": content,
                                "label": label,
                                "reason": reason,
                                "step": step,
                                "gold_answer": gold_ans,
                                "pattern": pattern,
                            }
                        )
                        if len(pattern_examples) >= k_per_pattern:
                            break

            for ex in pattern_examples:
                if ex["idx"] not in seen_indices:
                    all_examples.append(ex)
                    seen_indices.add(ex["idx"])

    if not all_examples:
        return ""

    formatted_parts = []
    for ex in all_examples:
        formatted_parts.append(
            f'[Corrective Example for {ex["step"]} - Common mistake: {ex["pattern"].split("->")[0]} → Correct: {ex["gold_answer"]}]\n'
            f'Post: "{extract_text(ex["content"])}"\n'
            f'Category: [{extract_category(ex["content"])}]\n'
            f'Label: {ex["label"]}\n'
            f'Reasoning: {ex["reason"]}'
        )

    return "\n\n---\n\n".join(formatted_parts)


def build_system_prompt(
    step_definitions: Dict[str, str],
    output_format: str,
    step_specific_examples: str = "",
    corrective_examples: str = "",
) -> str:
    return LINGO_SYSTEM_PROMPT_TEMPLATE.format(
        step_1_definition=step_definitions.get("STEP 1", DEFAULT_STEP_DEFINITIONS["STEP 1"]),
        step_2_definition=step_definitions.get("STEP 2", DEFAULT_STEP_DEFINITIONS["STEP 2"]),
        step_3_definition=step_definitions.get("STEP 3", DEFAULT_STEP_DEFINITIONS["STEP 3"]),
        step_4_definition=step_definitions.get("STEP 4", DEFAULT_STEP_DEFINITIONS["STEP 4"]),
        step_5_definition=step_definitions.get("STEP 5", DEFAULT_STEP_DEFINITIONS["STEP 5"]),
        output_format=output_format,
        step_specific_examples=step_specific_examples,
        corrective_examples=corrective_examples,
    )


@dataclass
class StepOptimizationData(adal.DataClass):
    id: str = field(default="", metadata={"desc": "Unique identifier for the sample"})
    text: str = field(default="", metadata={"desc": "The social media post text"})
    category: str = field(default="", metadata={"desc": "The target category"})
    step_name: str = field(default="", metadata={"desc": "The step being evaluated"})
    step_answer: str = field(default="", metadata={"desc": "The ground truth step answer"})
    full_label: str = field(default="", metadata={"desc": "The full label (0-6)"})
    full_content: str = field(default="", metadata={"desc": "Original content with category tag"})
    full_reason: str = field(default="", metadata={"desc": "Full reasoning from gold data"})

    __input_fields__ = ["text", "category", "step_name", "full_content"]
    __output_fields__ = ["step_answer"]

    def to_demo_str(self) -> str:
        return f"Category: [{self.category}]\nPost: {self.text}\n{self.step_name} Answer: {self.step_answer}"


class MultiComponentStepOptimizer(adal.Component):
    """
    Step-level optimizer:
      - PROMPT: step_definition
      - DEMOS:  step_specific_examples, corrective_examples
    """

    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        step_name: str,
        train_data: List[StepOptimizationData] = None,
        error_analysis: Dict[str, Dict[str, Any]] = None,
        train_df: pd.DataFrame = None,
    ):
        super().__init__()
        self.step_name = step_name
        self.error_analysis = error_analysis or {}
        self.train_df = train_df
        self.train_data = train_data or []

        self.step_definition = adal.Parameter(
            data=DEFAULT_STEP_DEFINITIONS.get(step_name, ""),
            role_desc=(
                f"The definition and instructions for {step_name}. "
                "This should clearly explain what the step checks, the question to answer, and the possible outcomes."
            ),
            requires_opt=True,
            param_type=ParameterType.PROMPT,
        )

        self.step_specific_examples = adal.Parameter(
            data="",
            role_desc=(
                f"Step-specific examples demonstrating how to correctly answer {step_name}. "
                "These examples are selected specifically for this step from training data."
            ),
            requires_opt=True,
            param_type=ParameterType.DEMOS,
        )

        self.corrective_examples = adal.Parameter(
            data="",
            role_desc=(
                f"Corrective examples showing correct answers for common mistakes in {step_name}. "
                "These address specific error patterns observed during evaluation."
            ),
            requires_opt=True,
            param_type=ParameterType.DEMOS,
        )

        self.template = r"""<START_OF_SYSTEM_PROMPT>
You are analyzing social media posts for a specific decision step.

{{step_definition}}

{% if step_specific_examples and step_specific_examples != "" %}
<STEP_SPECIFIC_EXAMPLES>
{{step_specific_examples}}
</STEP_SPECIFIC_EXAMPLES>
{% endif %}
{% if corrective_examples and corrective_examples != "" %}
<CORRECTIVE_EXAMPLES>
{{corrective_examples}}
</CORRECTIVE_EXAMPLES>
{% endif %}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
Category: [{{category}}]
Post: {{text}}

What is the answer for {{step_name}}?
<END_OF_USER>
"""

        self.generator = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=self.template,
            prompt_kwargs={
                "step_definition": self.step_definition,
                "step_specific_examples": self.step_specific_examples,
                "corrective_examples": self.corrective_examples,
            },
        )

    def bicall(self, text: str, category: str, step_name: str, id: str) -> adal.Parameter:
        return self.generator(
            prompt_kwargs={"text": text, "category": category, "step_name": step_name},
            id=id,
        )

    def call(self, text: str, category: str, step_name: str, id: Optional[str] = None) -> str:
        out = self.generator(
            prompt_kwargs={"text": text, "category": category, "step_name": step_name},
            id=id,
        )
        if out and out.data is not None:
            return self._parse_step_answer(out.data)
        return ""

    def _parse_step_answer(self, response: str) -> str:
        response = str(response).strip().upper()

        if self.step_name in ["STEP 1", "STEP 2", "STEP 4"]:
            if "YES" in response or "SIM" in response:
                return "YES"
            if "NO" in response or "NÃO" in response or "NAO" in response:
                return "NO"
        elif self.step_name == "STEP 3":
            if "REPORT" in response:
                return "REPORT"
            if "INTENSIF" in response:
                return "INTENSIFY"
            if "COUNTER" in response or "CONTRA" in response:
                return "COUNTER"
            if "ESCALAT" in response:
                return "ESCALATE"
        elif self.step_name == "STEP 5":
            if "EXPLICIT" in response:
                return "EXPLICIT"
            if "IMPLICIT" in response:
                return "IMPLICIT"

        return response

    def get_optimized_components(self) -> Dict[str, str]:
        return {
            "step_definition": self.step_definition.data,
            "step_specific_examples": self.step_specific_examples.data,
            "corrective_examples": self.corrective_examples.data,
        }


def step_answer_match_eval(y: Any, y_gt: Any) -> float:
    pred = _normalize_step_answer(str(y) if y is not None else "")
    gt = _normalize_step_answer(str(y_gt) if y_gt is not None else "")
    return 1.0 if pred == gt else 0.0


class MultiComponentStepAdalComponent(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict,
        text_optimizer_model_config: Dict,
        step_name: str,
        train_data: List[StepOptimizationData] = None,
        teacher_model_config: Optional[Dict] = None,
        error_analysis: Dict[str, Dict[str, Any]] = None,
        train_df: pd.DataFrame = None,
    ):
        task = MultiComponentStepOptimizer(
            model_client=model_client,
            model_kwargs=model_kwargs,
            step_name=step_name,
            train_data=train_data,
            error_analysis=error_analysis,
            train_df=train_df,
        )

        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=step_answer_match_eval,
            eval_fn_desc=f"1 if predicted {step_name} answer matches ground truth else 0",
        )

        super().__init__(
            task=task,
            eval_fn=step_answer_match_eval,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            teacher_model_config=teacher_model_config,
        )
        self.step_name = step_name

    def prepare_task(self, sample: StepOptimizationData) -> Tuple[Callable, Dict[str, Any]]:
        return self.task.bicall, {
            "text": sample.text,
            "category": sample.category,
            "step_name": sample.step_name,
            "id": sample.id,
        }

    def prepare_eval(self, sample: StepOptimizationData, y_pred: Any) -> Tuple[Callable[..., Any], Dict[str, Any]]:
        pred_val = getattr(y_pred, "data", y_pred)
        pred_val = pred_val if pred_val is not None else ""
        return self.eval_fn, {"y": str(pred_val), "y_gt": sample.step_answer}

    def prepare_loss(self, sample: StepOptimizationData, y_pred: adal.Parameter) -> Tuple[Callable, Dict[str, Any]]:
        y_gt = adal.Parameter(
            data=sample.step_answer,
            role_desc=f"Ground truth answer for {sample.step_name}",
            requires_opt=False,
            param_type=ParameterType.INPUT,
        )

        return self.loss_fn, {
            "kwargs": {"y": y_pred, "y_gt": y_gt},
            "id": sample.id,
            "input": f"Category: [{sample.category}]\nPost: {sample.text}",
        }


def get_step_training_data(step_name: str, df: pd.DataFrame) -> List[StepOptimizationData]:
    step_data = []
    for idx, row in df.iterrows():
        content = row.get("content", "")
        reason = str(row.get("reason", ""))
        label = int(row.get("intention_label", 0))

        category = extract_category(content)
        text = extract_text(content)
        step_answers = extract_step_answers(reason)

        if step_name in step_answers:
            step_data.append(
                StepOptimizationData(
                    id=f"step_{step_name}_{idx}",
                    text=text,
                    category=category,
                    step_name=step_name,
                    step_answer=step_answers[step_name],
                    full_label=str(label),
                    full_content=content,
                    full_reason=reason,
                )
            )
    return step_data


def get_step_training_data_with_error_focus(
    step_name: str,
    df: pd.DataFrame,
    error_analysis: Dict[str, Dict[str, Any]] = None,
    max_samples: int = 100,
) -> List[StepOptimizationData]:
    """
    Train data for AdalFlow step optimizer is ALWAYS restricted to the requested step_name.
    Error patterns only influence prioritization within that step-specific pool.
    """
    step_data = []
    priority_data = []

    target_gold_answers = set()
    if error_analysis and step_name in error_analysis:
        common_errors = error_analysis[step_name].get("most_common_errors", [])
        for pattern, _count in common_errors[:5]:
            if "->" in pattern:
                _, gold_ans = pattern.split("->", 1)
                target_gold_answers.add(_normalize_step_answer(gold_ans))

    for idx, row in df.iterrows():
        content = row.get("content", "")
        reason = str(row.get("reason", ""))
        label = int(row.get("intention_label", 0))

        category = extract_category(content)
        text = extract_text(content)
        step_answers = extract_step_answers(reason)

        if step_name in step_answers:
            answer = step_answers[step_name]
            normalized_answer = _normalize_step_answer(answer)

            data_item = StepOptimizationData(
                id=f"step_{step_name}_{idx}",
                text=text,
                category=category,
                step_name=step_name,
                step_answer=answer,
                full_label=str(label),
                full_content=content,
                full_reason=reason,
            )

            if target_gold_answers and normalized_answer in target_gold_answers:
                priority_data.append(data_item)
            else:
                step_data.append(data_item)

    combined = priority_data + step_data
    return combined[:max_samples]


class MultiComponentOptimizerManager:
    def __init__(
        self,
        forward_model_client: adal.ModelClient,
        forward_model_kwargs: Dict,
        teacher_model_config: Dict,
        optimizer_model_config: Dict,
        train_batch_size: int = 6,
        max_steps: int = 5,
        raw_shots: int = 30,
        bootstrap_shots: int = 5,
    ):
        self.forward_model_client = forward_model_client
        self.forward_model_kwargs = forward_model_kwargs
        self.teacher_model_config = teacher_model_config
        self.optimizer_model_config = optimizer_model_config
        self.train_batch_size = train_batch_size
        self.max_steps = max_steps
        self.raw_shots = raw_shots
        self.bootstrap_shots = bootstrap_shots

        self.optimized_components: Dict[str, Dict[str, str]] = {}

    def optimize_step(
        self,
        step_name: str,
        train_data: List[StepOptimizationData],
        val_data: List[StepOptimizationData],
        error_analysis: Dict[str, Dict[str, Any]] = None,
        train_df: pd.DataFrame = None,
    ) -> Dict[str, str]:
        print(f"\n{'='*50}")
        print(f"Optimizing {step_name} with AdalFlow...")
        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        print(f"Components: step_definition (PROMPT)")
        print(f"            step_specific_examples (DEMOS), corrective_examples (DEMOS)")
        print(f"{'='*50}")

        if len(train_data) < 3 or len(val_data) < 2:
            print(f"Not enough samples for {step_name}, using default components")
            return {
                "step_definition": DEFAULT_STEP_DEFINITIONS.get(step_name, ""),
                "step_specific_examples": "",
                "corrective_examples": "",
            }

        adal_component = MultiComponentStepAdalComponent(
            model_client=self.forward_model_client,
            model_kwargs=self.forward_model_kwargs,
            backward_engine_model_config=self.teacher_model_config,
            text_optimizer_model_config=self.optimizer_model_config,
            step_name=step_name,
            train_data=train_data,
            teacher_model_config=self.teacher_model_config,
            error_analysis=error_analysis,
            train_df=train_df,
        )

        effective_raw_shots = min(self.raw_shots, len(train_data))
        effective_bootstrap_shots = min(self.bootstrap_shots, len(train_data) // 2, effective_raw_shots)

        trainer = adal.Trainer(
            adaltask=adal_component,
            train_batch_size=self.train_batch_size,
            max_steps=self.max_steps,
            strategy="constrained",
            optimization_order="interleaved",
            max_proposals_per_step=3,
            debug=False,
            raw_shots=effective_raw_shots,
            bootstrap_shots=effective_bootstrap_shots,
            weighted_sampling=True,
            max_error_samples=4,
            max_correct_samples=4,
        )

        try:
            trainer.fit(train_dataset=train_data, val_dataset=val_data)

            optimized = adal_component.task.get_optimized_components()
            self.optimized_components[step_name] = optimized

            print(f"\nOptimized components for {step_name}:")
            print(f"  - step_definition (PROMPT): {len(optimized['step_definition'])} chars")
            print(f"  - step_specific_examples (DEMOS): {len(optimized['step_specific_examples'])} chars")
            print(f"  - corrective_examples (DEMOS): {len(optimized['corrective_examples'])} chars")

            return optimized

        except Exception as e:
            print(f"Error optimizing {step_name}: {e}")
            return {
                "step_definition": DEFAULT_STEP_DEFINITIONS.get(step_name, ""),
                "step_specific_examples": "",
                "corrective_examples": "",
            }

    def get_optimized_step_definition(self, step_name: str) -> str:
        if step_name in self.optimized_components:
            return self.optimized_components[step_name].get(
                "step_definition",
                DEFAULT_STEP_DEFINITIONS.get(step_name, ""),
            )
        return DEFAULT_STEP_DEFINITIONS.get(step_name, "")

    def get_all_step_definitions(self) -> Dict[str, str]:
        return {step: self.get_optimized_step_definition(step) for step in STEP_KEYS}


class SocialMediaAnalyzerAdalFlow:
    def __init__(
        self,
        model: str,
        optimized_step_definitions: Dict[str, str] = None,
        target_steps: List[str] = None,
        error_analysis: Dict[str, Dict[str, Any]] = None,
        train_df: pd.DataFrame = None,
        k_step_examples: int = 3,
        k_error_examples: int = 2,
    ):
        self.model = model
        self.target_steps = target_steps or []
        self.error_analysis = error_analysis or {}
        self.train_df = train_df
        self.k_step_examples = k_step_examples
        self.k_error_examples = k_error_examples

        self.step_definitions = {}
        for step in STEP_KEYS:
            if optimized_step_definitions and step in optimized_step_definitions:
                self.step_definitions[step] = optimized_step_definitions[step]
            else:
                self.step_definitions[step] = DEFAULT_STEP_DEFINITIONS[step]

        self.output_format = DEFAULT_OUTPUT_FORMAT

        self.step_specific_examples = ""
        self.corrective_examples = ""

        if train_df is not None and not train_df.empty and self.target_steps:
            example_pool_df = filter_df_to_target_steps(train_df, self.target_steps)

            self.step_specific_examples = get_step_targeted_examples(
                self.target_steps, example_pool_df, k_per_step=k_step_examples
            )
            if self.step_specific_examples:
                self.step_specific_examples = (
                    f"## STEP-SPECIFIC EXAMPLES (demonstrating {', '.join(self.target_steps)}):\n"
                    f"{self.step_specific_examples}"
                )

            if self.error_analysis:
                self.corrective_examples = get_error_pattern_examples(
                    self.error_analysis, example_pool_df, k_per_pattern=k_error_examples
                )
                if self.corrective_examples:
                    self.corrective_examples = (
                        "## CORRECTIVE EXAMPLES (showing correct answers for common mistakes):\n"
                        f"{self.corrective_examples}"
                    )

        self.system_prompt = build_system_prompt(
            step_definitions=self.step_definitions,
            output_format=self.output_format,
            step_specific_examples=self.step_specific_examples,
            corrective_examples=self.corrective_examples,
        )

    def get_prediction(self, content: str, max_retries: int = 3) -> Tuple[int, str]:
        for attempt in range(max_retries):
            try:
                resp = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content},
                    ],
                )
                text = resp.choices[0].message.content.strip()

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
                    label = int(obj.get("LABEL", -1))
                    reasoning_obj = obj.get("REASONING", {})

                    if "STEP 1" not in reasoning_obj:
                        print(f"  Retry {attempt + 1}/{max_retries}: Missing STEP 1")
                        continue

                    if label < 0 or label > 6:
                        print(f"  Retry {attempt + 1}/{max_retries}: Invalid label {label}")
                        continue

                    reasoning = json.dumps(reasoning_obj, ensure_ascii=False)
                    return label, reasoning

                print(f"  Retry {attempt + 1}/{max_retries}: Failed to parse JSON")
                continue

            except Exception as e:
                print(f"  Retry {attempt + 1}/{max_retries}: Error - {str(e)}")
                continue

        print(f"  All {max_retries} retries failed, returning default")
        return 0, '{"error": "All retries failed"}'

    def process_dataframe(
        self,
        df: pd.DataFrame,
        delay: float = 0.0,
        desc: str = "Processing",
        max_retries: int = 3,
    ) -> pd.DataFrame:
        preds, reasons = [], []

        for _, row in df.iterrows():
            if "content" not in row:
                raise ValueError("DataFrame must include a 'content' column.")
            label, r = self.get_prediction(row["content"], max_retries=max_retries)
            preds.append(label)
            reasons.append(r)
            if delay:
                time.sleep(delay)

        out = df.copy()
        out["predicted_label"] = preds
        out["reasoning"] = reasons
        return out

    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        accuracy = accuracy_score(y_true, y_pred)

        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

        p_pc, r_pc, f_pc, s_pc = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        cls_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        conf = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_per_class": p_pc,
            "recall_per_class": r_pc,
            "f1_per_class": f_pc,
            "support_per_class": s_pc,
            "classification_report": cls_report,
            "confusion_matrix": conf,
        }

    @staticmethod
    def create_results_table(metrics: Dict[str, Any], labels: List[str] = None) -> pd.DataFrame:
        if labels is None:
            labels = ["No Category", "Explicit", "Implicit", "Report", "Intensify", "Counter", "Escalate"]
        rows = []
        for i, lab in enumerate(labels):
            if i < len(metrics["precision_per_class"]):
                rows.append(
                    {
                        "Class": lab,
                        "Precision": f"{metrics['precision_per_class'][i]:.3f}",
                        "Recall": f"{metrics['recall_per_class'][i]:.3f}",
                        "F1-Score": f"{metrics['f1_per_class'][i]:.3f}",
                        "Support": int(metrics["support_per_class"][i]),
                    }
                )
        total_support = int(sum(metrics["support_per_class"]))
        rows += [
            {
                "Class": "Macro Avg",
                "Precision": f"{metrics['precision_macro']:.3f}",
                "Recall": f"{metrics['recall_macro']:.3f}",
                "F1-Score": f"{metrics['f1_macro']:.3f}",
                "Support": total_support,
            },
            {
                "Class": "Weighted Avg",
                "Precision": f"{metrics['precision_weighted']:.3f}",
                "Recall": f"{metrics['recall_weighted']:.3f}",
                "F1-Score": f"{metrics['f1_weighted']:.3f}",
                "Support": total_support,
            },
            {
                "Class": "Micro Avg",
                "Precision": f"{metrics['precision_micro']:.3f}",
                "Recall": f"{metrics['recall_micro']:.3f}",
                "F1-Score": f"{metrics['f1_micro']:.3f}",
                "Support": total_support,
            },
        ]
        return pd.DataFrame(rows)


def compute_step_diff_distribution(df: pd.DataFrame) -> Tuple[Counter, List[Dict[str, Any]]]:
    dist = Counter()
    records = []

    for i, row in df.iterrows():
        gold_label = row.get("intention_label", 0)
        pred_label = row.get("predicted_label", 0)

        gold = extract_step_answers(str(row.get("reason", "")))
        pred = extract_step_answers(str(row.get("reasoning", "")))

        diff_step = earliest_step_diff(pred, gold, pred_label, gold_label)
        if diff_step is not None:
            dist[diff_step] += 1
            records.append(
                {
                    "index": i,
                    "diff_step": diff_step,
                    "gold_steps": gold,
                    "pred_steps": pred,
                    "gold_step_answer": gold.get(diff_step, ""),
                    "pred_step_answer": pred.get(diff_step, ""),
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "content": row.get("content", ""),
                    "gold_reason": row.get("reason", ""),
                    "pred_reason": row.get("reasoning", ""),
                }
            )

    return dist, records


def analyze_step_error_patterns(step_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    step_errors: Dict[str, List[Dict[str, Any]]] = {}
    for rec in step_records:
        step = rec["diff_step"]
        step_errors.setdefault(step, []).append(rec)

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
            "error_records": errors,
        }

    return analysis


def identify_problematic_steps_with_patterns(
    step_diff_distribution: Counter,
    step_records: List[Dict[str, Any]],
    threshold: float = 0.1,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    total = sum(step_diff_distribution.values())
    if total == 0:
        return [], {}

    problematic = [(step, count) for step, count in step_diff_distribution.items() if (count / total) >= threshold]
    problematic.sort(key=lambda x: x[1], reverse=True)
    problematic_steps = [step for step, _ in problematic]

    error_analysis = analyze_step_error_patterns(step_records)
    filtered_analysis = {step: a for step, a in error_analysis.items() if step in problematic_steps}
    return problematic_steps, filtered_analysis


def evaluate_on_dataset(
    df: pd.DataFrame,
    analyzer: SocialMediaAnalyzerAdalFlow,
    gold_reason_col: str = "reason",
    delay: float = 0.0,
    desc: str = "Evaluating",
    max_retries: int = 3,
) -> Dict[str, Any]:
    results_df = analyzer.process_dataframe(df, delay=delay, desc=desc, max_retries=max_retries)
    print(results_df["reasoning"].head(20).tolist())

    metrics = analyzer.calculate_metrics(
        y_true=results_df["intention_label"].tolist(),
        y_pred=results_df["predicted_label"].tolist(),
    )

    merged = results_df.copy()
    merged["reason"] = df[gold_reason_col].astype(str) if gold_reason_col in df.columns else ""

    dist, recs = compute_step_diff_distribution(merged)
    total = len(merged)
    diff_total = sum(dist.values())
    step_diff_rate = diff_total / total if total else 0.0

    return {
        "results_df": results_df,
        "metrics": metrics,
        "step_diff_distribution": dict(dist),
        "step_records": recs,
        "step_diff_rate": step_diff_rate,
    }


def iterative_adalflow_refinement(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forward_model: str = "gemini/gemini-2.5-flash-lite",
    teacher_model: str = "gpt-4o-mini",
    rounds: int = 3,
    delay: float = 0.0,
    seed: int = 42,
    step_threshold: float = 0.1,
    val_ratio: float = 0.2,
    adalflow_train_batch_size: int = 6,
    adalflow_max_steps: int = 5,
    adalflow_raw_shots: int = 30,
    adalflow_bootstrap_shots: int = 5,
    k_step_examples: int = 3,
    k_error_examples: int = 2,
) -> Dict[str, Any]:
    random.seed(seed)

    print("=" * 60)
    print("STEP 1: Splitting dev set into TRAIN and VAL")
    print("=" * 60)

    train_df, val_df = split_train_val(dev_df, val_ratio=val_ratio, seed=seed)

    print(f"  Original dev set: {len(dev_df)} samples")
    print(f"  TRAIN set: {len(train_df)} samples (for AdalFlow optimization)")
    print(f"  VAL set: {len(val_df)} samples (for step error analysis)")
    print(f"  TEST set: {len(test_df)} samples (final evaluation only)")

    print("\n" + "=" * 60)
    print("STEP 2: Initializing AdalFlow Multi-Component Optimizer")
    print(f"  Forward model: {forward_model}")
    print(f"  Teacher model: {teacher_model}")
    print("  Optimizing: PROMPT (step_definition) + DEMOS (step_specific_examples, corrective_examples)")
    print(f"  raw_shots: {adalflow_raw_shots}, bootstrap_shots: {adalflow_bootstrap_shots}")
    print("=" * 60)

    forward_client = OpenAIClient() #forward_client = AnthropicAPIClient(), forward_client = OpenAIClient(), GoogleGenAIClient()
    adalflow_forward_model = forward_model.split("/")[-1] if "/" in forward_model else forward_model

    forward_model_kwargs = {"model": adalflow_forward_model, "max_tokens": 1024}

    teacher_config = {
        "model_client": OpenAIClient(),
        "model_kwargs": {"model": teacher_model, "max_tokens": 1024},
    }
    optimizer_config = {
        "model_client": OpenAIClient(),
        "model_kwargs": {"model": teacher_model, "max_tokens": 1024},
    }

    step_optimizer = MultiComponentOptimizerManager(
        forward_model_client=forward_client,
        forward_model_kwargs=forward_model_kwargs,
        teacher_model_config=teacher_config,
        optimizer_model_config=optimizer_config,
        train_batch_size=adalflow_train_batch_size,
        max_steps=adalflow_max_steps,
        raw_shots=adalflow_raw_shots,
        bootstrap_shots=adalflow_bootstrap_shots,
    )

    print("\n" + "=" * 60)
    print("STEP 3: Iterative refinement using VAL set")
    print("=" * 60)

    history: List[Dict[str, Any]] = []
    target_steps: List[str] = []
    error_analysis: Dict[str, Dict[str, Any]] = {}

    optimized_step_definitions: Dict[str, str] = {}

    val_eval = None
    best_val_f1 = -1
    best_target_steps = []
    best_error_analysis = {}
    best_optimized_definitions = {}
    best_round = 0

    for r in trange(rounds + 1, desc="AdalFlow Refinement Rounds"):
        print(f"\n{'='*60}")
        print(f"Round {r}: Target steps = {target_steps if target_steps else 'None (baseline)'}")
        print("=" * 60)

        if target_steps and r > 0:
            print(f"\nOptimizing {len(target_steps)} targeted steps with AdalFlow...")
            print("  Optimizing PROMPT + DEMOS for each targeted step")

            for step_name in target_steps:
                step_train_data = get_step_training_data_with_error_focus(
                    step_name, train_df, error_analysis, max_samples=100
                )
                step_val_data = get_step_training_data(step_name, val_df)

                if step_train_data and step_val_data:
                    if len(step_train_data) > 5:
                        optimized_components = step_optimizer.optimize_step(
                            step_name=step_name,
                            train_data=step_train_data,
                            val_data=step_val_data,
                            error_analysis=error_analysis,
                            train_df=train_df,
                        )
                        optimized_step_definitions[step_name] = optimized_components["step_definition"]
                        print(f"  {step_name}: Optimized (PROMPT + DEMOS)")
                    else:
                        print(f"  {step_name}: Not enough data, keeping default")

        analyzer = SocialMediaAnalyzerAdalFlow(
            model=forward_model,
            optimized_step_definitions=optimized_step_definitions,
            target_steps=target_steps,
            error_analysis=error_analysis,
            train_df=train_df,
            k_step_examples=k_step_examples,
            k_error_examples=k_error_examples,
        )

        print("\nCurrent prompt structure:")
        for step in STEP_KEYS:
            status = "OPTIMIZED (PROMPT+DEMOS)" if step in optimized_step_definitions else "DEFAULT"
            print(f"  {step}: {status}")

        if analyzer.step_specific_examples:
            print("  Step-specific examples: included (targeted-step pool only)")
        if analyzer.corrective_examples:
            print("  Corrective examples: included (targeted-step pool only)")

        print("\nEvaluating on VAL set...")
        val_eval = evaluate_on_dataset(val_df, analyzer, delay=delay, desc="VAL")

        current_f1 = val_eval["metrics"]["f1_weighted"]
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_target_steps = list(target_steps)
            best_error_analysis = dict(error_analysis)
            best_optimized_definitions = dict(optimized_step_definitions)
            best_round = r

        history.append(
            {
                "round": r,
                "target_steps": list(target_steps),
                "optimized_steps": list(optimized_step_definitions.keys()),
                "error_patterns": {
                    step: analysis.get("most_common_errors", [])
                    for step, analysis in error_analysis.items()
                }
                if error_analysis
                else {},
                "val_accuracy": val_eval["metrics"]["accuracy"],
                "val_f1_macro": val_eval["metrics"]["f1_macro"],
                "val_f1_weighted": val_eval["metrics"]["f1_weighted"],
                "val_f1_micro": val_eval["metrics"]["f1_micro"],
                "val_step_diff_distribution": val_eval["step_diff_distribution"],
                "val_step_diff_rate": val_eval["step_diff_rate"],
            }
        )

        print(
            f"VAL - Accuracy: {val_eval['metrics']['accuracy']:.4f}, "
            f"F1 (weighted): {val_eval['metrics']['f1_weighted']:.4f}"
        )
        print(f"VAL step diff distribution: {val_eval['step_diff_distribution']}")

        if r == rounds:
            break

        val_dist_counter = Counter(val_eval["step_diff_distribution"])
        if sum(val_dist_counter.values()) == 0:
            print("No step differences found - stopping early")
            break

        target_steps, error_analysis = identify_problematic_steps_with_patterns(
            val_dist_counter, val_eval["step_records"], step_threshold
        )

        print(f"Identified problematic steps for next round: {target_steps}")
        for step, analysis in error_analysis.items():
            print(f"  {step} error patterns: {analysis.get('most_common_errors', [])[:3]}")

    print("\n" + "=" * 60)
    print(f"STEP 4: Final evaluation on TEST set")
    print(f"  Using best configuration from round {best_round}")
    print(f"  Steps with optimized PROMPT+DEMOS: {list(best_optimized_definitions.keys())}")
    print(f"  Target steps with examples: {best_target_steps}")
    print("=" * 60)

    final_analyzer = SocialMediaAnalyzerAdalFlow(
        model=forward_model,
        optimized_step_definitions=best_optimized_definitions,
        target_steps=best_target_steps,
        error_analysis=best_error_analysis,
        train_df=train_df,
        k_step_examples=k_step_examples,
        k_error_examples=k_error_examples,
    )

    print("\nFinal prompt structure:")
    for step in STEP_KEYS:
        status = "OPTIMIZED (PROMPT+DEMOS)" if step in best_optimized_definitions else "DEFAULT"
        print(f"  {step}: {status}")

    if final_analyzer.step_specific_examples:
        print("  Step-specific examples: included (targeted-step pool only)")
    if final_analyzer.corrective_examples:
        print("  Corrective examples: included (targeted-step pool only)")

    print("\nEvaluating on TEST set...")
    test_eval = evaluate_on_dataset(test_df, final_analyzer, delay=delay, desc="TEST")

    print("\nFINAL TEST RESULTS:")
    print(f"  Accuracy: {test_eval['metrics']['accuracy']:.4f}")
    print(f"  F1 (weighted): {test_eval['metrics']['f1_weighted']:.4f}")

    return {
        "history": history,
        "best_round": best_round,
        "best_target_steps": best_target_steps,
        "best_error_analysis": best_error_analysis,
        "best_optimized_definitions": best_optimized_definitions,
        "best_val_f1": best_val_f1,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "final_val_results": val_eval["results_df"] if val_eval else None,
        "final_test_results": test_eval["results_df"],
        "final_val_metrics": val_eval["metrics"] if val_eval else None,
        "final_test_metrics": test_eval["metrics"],
        "final_test_step_diff_distribution": test_eval["step_diff_distribution"],
        "forward_model": forward_model,
        "teacher_model": teacher_model,
        "final_system_prompt": final_analyzer.system_prompt,
    }


def export_adalflow_artifacts(
    run_result: Dict[str, Any],
    out_dir: str = "lingo_adalflow_outputs",
    history_csv_name: str = "history_metrics.csv",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    history = run_result.get("history", [])

    if not history:
        raise ValueError("No history found in run_result. Did the run complete?")

    csv_path = os.path.join(out_dir, history_csv_name)
    rows = []
    for h in history:
        rows.append(
            {
                "round": h["round"],
                "target_steps": ", ".join(h["target_steps"]) if h["target_steps"] else "None",
                "optimized_steps": ", ".join(h.get("optimized_steps", [])) if h.get("optimized_steps") else "None",
                "val_accuracy": h["val_accuracy"],
                "val_f1_macro": h["val_f1_macro"],
                "val_f1_weighted": h["val_f1_weighted"],
                "val_f1_micro": h["val_f1_micro"],
                "val_step_diff_rate": h["val_step_diff_rate"],
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved metrics history to {csv_path}")

    if run_result.get("final_test_results") is not None:
        test_path = os.path.join(out_dir, "final_test_predictions.csv")
        run_result["final_test_results"].to_csv(test_path, index=False, encoding="utf-8")
        print(f"Saved test predictions to {test_path}")

    if run_result.get("best_optimized_definitions"):
        definitions_path = os.path.join(out_dir, "optimized_step_definitions.json")
        with open(definitions_path, "w", encoding="utf-8") as f:
            json.dump(run_result["best_optimized_definitions"], f, indent=2, ensure_ascii=False)
        print(f"Saved optimized step definitions to {definitions_path}")

    if run_result.get("final_system_prompt"):
        prompt_path = os.path.join(out_dir, "final_system_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(run_result["final_system_prompt"])
        print(f"Saved final system prompt to {prompt_path}")

    summary_path = os.path.join(out_dir, "summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("LinGO AdalFlow Multi-Component Optimization Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Forward model: {run_result.get('forward_model', 'N/A')}\n")
        f.write(f"  Teacher model: {run_result.get('teacher_model', 'N/A')}\n\n")

        f.write("OPTIMIZATION APPROACH:\n")
        f.write("  - Targeted steps optimized with PROMPT + DEMOS\n")
        f.write("  - PROMPT: step_definition\n")
        f.write("  - DEMOS: step_specific_examples, corrective_examples\n")
        f.write("  - Examples for evaluation prompt are sourced ONLY from targeted-step-specific TRAIN samples\n\n")

        f.write("DATA SPLIT:\n")
        f.write(f"  TRAIN: {run_result.get('train_size', 'N/A')} samples\n")
        f.write(f"  VAL: {run_result.get('val_size', 'N/A')} samples\n")
        f.write(f"  TEST: {run_result.get('test_size', 'N/A')} samples\n\n")

        f.write(f"Best Round: {run_result.get('best_round', 'N/A')}\n")
        f.write(
            "Steps with optimized PROMPT+DEMOS: "
            + (", ".join(run_result.get("best_optimized_definitions", {}).keys()) or "None")
            + "\n"
        )
        f.write(
            "Target steps (with examples): "
            + (", ".join(run_result.get("best_target_steps", [])) or "None")
            + "\n"
        )
        f.write(f"Best VAL F1 (weighted): {run_result.get('best_val_f1', 0):.4f}\n\n")

        f.write("FINAL PROMPT STRUCTURE:\n")
        optimized = run_result.get("best_optimized_definitions", {})
        for step in STEP_KEYS:
            status = "OPTIMIZED (PROMPT+DEMOS)" if step in optimized else "DEFAULT"
            f.write(f"  {step}: {status}\n")

        f.write("\nERROR PATTERNS ADDRESSED:\n")
        best_error_analysis = run_result.get("best_error_analysis", {})
        for step, analysis in best_error_analysis.items():
            f.write(f"  {step}:\n")
            for pattern, count in analysis.get("most_common_errors", [])[:3]:
                f.write(f"    - {pattern}: {count} occurrences\n")

        f.write("\nVAL Metrics by Round:\n")
        f.write("-" * 60 + "\n")
        for h in history:
            f.write(f"\nRound {h['round']}:\n")
            f.write(f"  Target Steps: {h['target_steps'] or 'None (baseline)'}\n")
            f.write(f"  Optimized (PROMPT+DEMOS) Steps: {h.get('optimized_steps', [])}\n")
            f.write(f"  VAL - Acc: {h['val_accuracy']:.4f}, F1-weighted: {h['val_f1_weighted']:.4f}\n")
            f.write(f"  Step Diff Distribution: {h['val_step_diff_distribution']}\n")
            if h.get("error_patterns"):
                f.write("  Error Patterns:\n")
                for step, patterns in h["error_patterns"].items():
                    f.write(f"    {step}: {patterns[:2]}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("FINAL TEST RESULTS:\n")
        f.write("=" * 60 + "\n")
        test_metrics = run_result.get("final_test_metrics", {})
        f.write(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  F1 (weighted): {test_metrics.get('f1_weighted', 0):.4f}\n")
        f.write(f"  Step Diff Distribution: {run_result.get('final_test_step_diff_distribution', {})}\n")

        if run_result.get("best_optimized_definitions"):
            f.write("\n" + "=" * 60 + "\n")
            f.write("OPTIMIZED STEP DEFINITIONS:\n")
            f.write("=" * 60 + "\n")
            for step, definition in run_result["best_optimized_definitions"].items():
                f.write(f"\n{step}:\n")
                f.write("-" * 40 + "\n")
                f.write(definition)
                f.write("\n")

    print(f"Saved summary report to {summary_path}")
    return out_dir


def run_adalflow_optimization(
    dev_filepath: str,
    test_filepath: str,
    forward_model: str = "gemini/gemini-2.5-flash-lite",
    teacher_model: str = "gpt-4o-mini",
    rounds: int = 3,
    delay: float = 0.0,
    step_threshold: float = 0.1,
    val_ratio: float = 0.2,
    seed: int = 42,
    output_dir: str = "lingo_adalflow_outputs",
    adalflow_train_batch_size: int = 6,
    adalflow_max_steps: int = 5,
    adalflow_raw_shots: int = 30,
    adalflow_bootstrap_shots: int = 5,
    k_step_examples: int = 3,
    k_error_examples: int = 2,
) -> Dict[str, Any]:
    print("Loading data...")
    dev_df = pd.read_csv(dev_filepath)
    test_df = pd.read_csv(test_filepath)
    print(f"  DEV (to be split): {len(dev_df)} samples")
    print(f"  TEST (held out): {len(test_df)} samples")

    result = iterative_adalflow_refinement(
        dev_df=dev_df,
        test_df=test_df,
        forward_model=forward_model,
        teacher_model=teacher_model,
        rounds=rounds,
        delay=delay,
        seed=seed,
        step_threshold=step_threshold,
        val_ratio=val_ratio,
        adalflow_train_batch_size=adalflow_train_batch_size,
        adalflow_max_steps=adalflow_max_steps,
        adalflow_raw_shots=adalflow_raw_shots,
        adalflow_bootstrap_shots=adalflow_bootstrap_shots,
        k_step_examples=k_step_examples,
        k_error_examples=k_error_examples,
    )

    export_adalflow_artifacts(result, out_dir=output_dir)
    return result


if __name__ == "__main__":
    result = run_adalflow_optimization(
        dev_filepath="./incivi_dev.csv",
        test_filepath="./incivi_test.csv",
        forward_model="gpt-5-mini",
        teacher_model="gpt-5-mini",
        rounds=5,
        delay=0.0,
        step_threshold=0.1,
        val_ratio=0.2,
        seed=42,
        output_dir="Adalflow_LinGO_outputs_GPT",
        adalflow_train_batch_size=6,
        adalflow_max_steps=5,
        adalflow_raw_shots=30,
        adalflow_bootstrap_shots=5,
        k_step_examples=4,
        k_error_examples=4,
    )

## Results: Claude (Test Accuracy: 0.3650; Test Weighted F1: 0.3795); Gemini (Test Accuracy: 0.5300; Test Weighted F1: 0.5505); GPT (Test Accuracy: 0.6200; Test Weighted F1: 0.6373)