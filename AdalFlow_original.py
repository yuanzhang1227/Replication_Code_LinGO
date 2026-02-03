import adalflow as adal
from adalflow.optim.types import ParameterType
from adalflow.components.model_client import GoogleGenAIClient, OpenAIClient, AnthropicAPIClient
from typing import Dict, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import os

os.environ['OPENAI_API_KEY'] = "YOUR API KEY"
os.environ['GOOGLE_API_KEY'] = "YOUR API KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

TASK_DESC = """You are an annotator interpreting the intent of uncivil Portuguese social media discourse about Brazilian politics on X (formerly Twitter).
Target Category: Always taken from the first square-bracket tag (e.g., [Impoliteness], [Hate Speech and Stereotyping], [Physical Harm and Violent Political Rhetoric], [Threats to Democratic Institutions and Values])
Category Definitions:
- Impoliteness: messages including rudeness/disrespect (name-calling, aspersions, calling someone a liar, hyperbole, non-cooperation, pejoratives, vulgarity, belittling, shouting via ALL-CAPS or excessive "!" when context supports it).
- Hate Speech and Stereotyping: harmful or discriminatory content targeting protected groups (gender identity, sexual orientation, religion, race, nationality, ideology, disability); over-generalizations, out-group demeaning; (do NOT include purely individual appearance insults unless tied to group identity).
- Physical Harm and Violent Political Rhetoric: threats/advocacy/praise of physical harm or violence; direct or metaphorical calls for harm; justification of violence for political ends.
- Threats to Democratic Institutions and Values: advocacy or approval of actions undermining elections/institutions/rule of law/press freedom/civil rights; promotion of autocracy; unfounded claims that delegitimize institutions.
Task: analyze the intent behind the given post and assign the most appropriate intention label from the list below (apply ONLY to the target category from the tag):
1 = Explicit {CATEGORY}: direct, overt {CATEGORY}.
2 = Implicit {CATEGORY}: indirect, veiled {CATEGORY}.
3 = Report {CATEGORY}: quotes/refers to {CATEGORY} content without opinion.
4 = Intensify {CATEGORY}: quotes/refers to {CATEGORY} content and agrees/amplifies.
5 = Counter {CATEGORY}: quotes/refers to {CATEGORY} content and criticizes/disagrees.
6 = Escalate {CATEGORY}: responds to {CATEGORY} content with {CATEGORY}.
0 = Other - does not fit any pattern.
Output: ONLY one integer from [0-6]. No other text.
"""


@dataclass
class InciviData(adal.DataClass):
    id: str = field(default="", metadata={"desc": "Unique identifier for the sample"})
    text: str = field(default="", metadata={"desc": "The social media post text"})
    label: str = field(default="", metadata={"desc": "The ground truth intent label (0-6)"})

    __input_fields__ = ["text"]
    __output_fields__ = ["label"]

    def to_demo_str(self) -> str:
        return f"Post: {self.text}\nLabel: {self.label}"


def load_dataset(filepath: str):
    df = pd.read_csv(filepath)
    text_col = "content" if "content" in df.columns else None
    label_col = "intention_label" if "intention_label" in df.columns else None
    if text_col is None or label_col is None:
        raise ValueError(f"Expected columns ['content','intention_label'], got {df.columns.tolist()}")

    samples = []
    for idx, row in df.iterrows():
        samples.append(
            InciviData(
                id=f"{filepath}:{idx}",
                text=str(row[text_col]),
                label=str(int(row[label_col])).strip(),
            )
        )
    return samples


class IntentClassifier(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        self.system_prompt = adal.Parameter(
            data=TASK_DESC,
            role_desc="System prompt for intent classification (0-6).",
            requires_opt=True,
            param_type=ParameterType.PROMPT,
        )

        self.few_shot_demos = adal.Parameter(
            data="",
            role_desc="Few-shot examples showing input-output pairs for intent classification",
            requires_opt=True,
            param_type=ParameterType.DEMOS,
        )

        self.template = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
<END_OF_SYSTEM_PROMPT>
{% if few_shot_demos and few_shot_demos != "" %}
<EXAMPLES>
{{few_shot_demos}}
</EXAMPLES>
{% endif %}
<START_OF_USER>
{{input_str}}
<END_OF_USER>
"""

        self.generator = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=self.template,
            prompt_kwargs={
                "system_prompt": self.system_prompt,
                "few_shot_demos": self.few_shot_demos,
            },
        )

    def bicall(self, question: str, id: str) -> adal.Parameter:
        return self.generator(prompt_kwargs={"input_str": question}, id=id)

    def call(self, question: str, id: Optional[str] = None) -> str:
        out = self.generator(prompt_kwargs={"input_str": question}, id=id)
        if out and out.data is not None:
            return self._parse_intent(out.data)
        return "0"

    @staticmethod
    def _parse_intent(response: str) -> str:
        m = re.search(r"\b([0-6])\b", str(response).strip())
        return m.group(1) if m else "0"


def intent_match_eval(y: Any, y_gt: Any) -> float:
    m1 = re.search(r"[0-6]", str(y).strip()) if y is not None else None
    m2 = re.search(r"[0-6]", str(y_gt).strip()) if y_gt is not None else None
    pred = m1.group() if m1 else "0"
    gt = m2.group() if m2 else "0"
    return 1.0 if pred == gt else 0.0


class IntentAdalComponent(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict,
        text_optimizer_model_config: Dict,
        teacher_model_config: Optional[Dict] = None,
    ):
        task = IntentClassifier(model_client=model_client, model_kwargs=model_kwargs)

        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=intent_match_eval,
            eval_fn_desc="1 if predicted digit 0-6 equals ground truth digit 0-6 else 0",
        )

        super().__init__(
            task=task,
            eval_fn=intent_match_eval,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            teacher_model_config=teacher_model_config,
        )

    def prepare_task(self, sample: InciviData) -> Tuple[Callable, Dict[str, Any]]:
        return self.task.bicall, {"question": sample.text, "id": sample.id}

    def prepare_eval(self, sample: InciviData, y_pred: Any) -> Tuple[Callable[..., Any], Dict[str, Any]]:
        pred_val = getattr(y_pred, "data", y_pred)
        pred_val = pred_val if pred_val is not None else "0"
        return self.eval_fn, {"y": str(pred_val), "y_gt": sample.label}

    def prepare_loss(self, sample: InciviData, y_pred: adal.Parameter) -> Tuple[Callable, Dict[str, Any]]:
        y_gt = adal.Parameter(
            data=sample.label,
            role_desc="Ground truth label 0-6",
            requires_opt=False,
            param_type=ParameterType.INPUT,
        )

        return self.loss_fn, {
            "kwargs": {"y": y_pred, "y_gt": y_gt},
            "id": sample.id,
            "input": f"Post: {sample.text}",
        }


def main():
    forward_client = GoogleGenAIClient() #forward_client = AnthropicAPIClient(), forward_client = OpenAIClient()
    model_kwargs = {
        "model": "gemini-2.5-flash-lite",
        "max_tokens": 1024,
    }

    teacher_config = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-5-mini",
            "max_tokens": 1024,
        },
    }
    optimizer_config = {
        "model_client": OpenAIClient(),
        "model_kwargs": {
            "model": "gpt-5-mini",
            "max_tokens": 1024,
        },
    }

    dev_data = load_dataset("incivi_dev.csv")
    test_data = load_dataset("incivi_test.csv")

    train_df, val_df = train_test_split(
        dev_data,
        test_size=0.2,
        random_state=42,
        stratify=[s.label for s in dev_data]
    )

    adal_component = IntentAdalComponent(
        model_client=forward_client,
        model_kwargs=model_kwargs,
        backward_engine_model_config=teacher_config,
        text_optimizer_model_config=optimizer_config,
        teacher_model_config=teacher_config,
    )

    trainer = adal.Trainer(
        adaltask=adal_component,
        train_batch_size=6,
        max_steps=5,
        strategy="constrained",
        optimization_order="interleaved",
        max_proposals_per_step=3,
        debug=False,
        raw_shots=30, 
        bootstrap_shots=5, 
        weighted_sampling=True,
        max_error_samples=4, 
        max_correct_samples=4, 
    )

    trainer.fit(train_dataset=train_df, val_dataset=val_df)

    adal_component.task.eval()
    y_true, y_pred = [], []
    for s in test_data:
        pred = adal_component.task.call(question=s.text, id=s.id)
        y_true.append(s.label)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Weighted F1: {f1w:.4f}")

    print("\n--- Optimized system prompt ---\n")
    print(adal_component.task.system_prompt.data)

    print("\n--- Optimized few-shot demos ---\n")
    print(adal_component.task.few_shot_demos.data)


if __name__ == "__main__":
    main()

## Results: Claude (Test Accuracy: 0.4875; Test Weighted F1: 0.3852); Gemini (Test Accuracy: 0.4925; Test Weighted F1: 0.4268); GPT (Test Accuracy: 0.5675; Test Weighted F1: 0.4109)