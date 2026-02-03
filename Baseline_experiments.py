import re
import time
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from litellm import completion
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.metrics import precision_recall_fscore_support

# Set up API keys for all the models
# Models: GPT-4o, claude-3-haiku-20240307, gemini-2.5-flash-lite
# os.environ['OPENAI_API_KEY'] = "xxxxxx"
# os.environ['GEMINI_API_KEY'] = "xxxxxx"
# os.environ["ANTHROPIC_API_KEY"] = "xxxxxx"

ZERO_SHOT_SYSTEM_PROMPT = """You are an annotator interpreting the intent of uncivil Portuguese social media discourse about Brazilian politics on X (formerly Twitter).

Target Category: Always taken from the first square-bracket tag (e.g., [Impoliteness], [Hate Speech and Stereotyping], [Physical Harm and Violent Political Rhetoric], [Threats to Democratic Institutions and Values])\

Category Definitions:
- Impoliteness: messages including rudeness/disrespect (name-calling, aspersions, calling someone a liar, hyperbole, non-cooperation, pejoratives, vulgarity, belittling, shouting via ALL-CAPS or excessive “!” when context supports it).
- Hate Speech and Stereotyping: harmful or discriminatory content targeting protected groups (gender identity, sexual orientation, religion, race, nationality, ideology, disability); over-generalizations, out-group demeaning; (do NOT include purely individual appearance insults unless tied to group identity).
- Physical Harm and Violent Political Rhetoric: threats/advocacy/praise of physical harm or violence; direct or metaphorical calls for harm; justification of violence for political ends.
- Threats to Democratic Institutions and Values: advocacy or approval of actions undermining elections/institutions/rule of law/press freedom/civil rights; promotion of autocracy; unfounded claims that delegitimize institutions.\

Task: analyze the intent behind the given post and assign the most appropriate intention label from the list below (apply ONLY to the target category from the tag):
1 = Explicit {{CATEGORY}}: direct, overt {{CATEGORY}}.
2 = Implicit {{CATEGORY}}: indirect, veiled {{CATEGORY}}.
3 = Report {{CATEGORY}}: quotes/refers to {{CATEGORY}} content without opinion.
4 = Intensify {{CATEGORY}}: quotes/refers to {{CATEGORY}} content and agrees/amplifies.
5 = Counter {{CATEGORY}}: quotes/refers to {{CATEGORY}} content and criticizes/disagrees.
6 = Escalate {{CATEGORY}}: responds to {{CATEGORY}} content with {{CATEGORY}}.
0 = No instances of {{CATEGORY}}.

OUTPUT FORMAT: only one integer from [0-6].
"""

COT_SYSTEM_PROMPT = """You are an annotator interpreting the intent of uncivil Portuguese social media discourse about Brazilian politics on X (formerly Twitter).

Target Category: Always taken from the first square-bracket tag (e.g., [Impoliteness], [Hate Speech and Stereotyping], [Physical Harm and Violent Political Rhetoric], [Threats to Democratic Institutions and Values])\

Category Definitions:
- Impoliteness: messages including rudeness/disrespect (name-calling, aspersions, calling someone a liar, hyperbole, non-cooperation, pejoratives, vulgarity, belittling, shouting via ALL-CAPS or excessive “!” when context supports it).
- Hate Speech and Stereotyping: harmful or discriminatory content targeting protected groups (gender identity, sexual orientation, religion, race, nationality, ideology, disability); over-generalizations, out-group demeaning; (do NOT include purely individual appearance insults unless tied to group identity).
- Physical Harm and Violent Political Rhetoric: threats/advocacy/praise of physical harm or violence; direct or metaphorical calls for harm; justification of violence for political ends.
- Threats to Democratic Institutions and Values: advocacy or approval of actions undermining elections/institutions/rule of law/press freedom/civil rights; promotion of autocracy; unfounded claims that delegitimize institutions.\

Task: analyze the intent behind the given post and assign the most appropriate intention label from the list below (apply ONLY to the target category from the tag):
1 = Explicit {{CATEGORY}}: direct, overt {{CATEGORY}}.
2 = Implicit {{CATEGORY}}: indirect, veiled {{CATEGORY}}.
3 = Report {{CATEGORY}}: quotes/refers to {{CATEGORY}} content without opinion.
4 = Intensify {{CATEGORY}}: quotes/refers to {{CATEGORY}} content and agrees/amplifies.
5 = Counter {{CATEGORY}}: quotes/refers to {{CATEGORY}} content and criticizes/disagrees.
6 = Escalate {{CATEGORY}}: responds to {{CATEGORY}} content with {{CATEGORY}}.
0 = No instances of {{CATEGORY}}.

OUTPUT FORMAT: only one integer from [0-6].

Please think step by step.
"""

def _to_json_safe(obj):
    """
    Convert numpy types/arrays to JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, (np.bool_, )):
        return bool(obj)
    return obj

class SocialMediaAnalyzerZeroShot:
    def __init__(self, model: str = "model_name", prompt: str = None):
        self.model = model
        self.system_prompt = prompt

    def get_prediction(self, content: str) -> int:
        try:
            resp = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}
                ]
            )
            text = resp.choices[0].message.content.strip()
            nums = re.findall(r'\d', text)
            return int(nums[0]) if nums else 0
        except Exception as e:
            print(f"Error: {e}")
            return 0

    def process_dataframe(self, df: pd.DataFrame, delay: float = 0.0) -> pd.DataFrame:
        preds = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing Zero-Shot"):
            if 'content' not in row:
                raise ValueError("DataFrame must include a 'content' column.")
            preds.append(self.get_prediction(row['content']))
            if delay:
                time.sleep(delay)
        out = df.copy()
        out['predicted_label'] = preds
        return out

    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

        p_pc, r_pc, f_pc, s_pc = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        cls_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        conf = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro,
            'precision_weighted': precision_weighted, 'recall_weighted': recall_weighted, 'f1_weighted': f1_weighted,
            'precision_micro': precision_micro, 'recall_micro': recall_micro, 'f1_micro': f1_micro,
            'precision_per_class': p_pc, 'recall_per_class': r_pc, 'f1_per_class': f_pc, 'support_per_class': s_pc,
            'classification_report': cls_report, 'confusion_matrix': conf
        }

    @staticmethod
    def create_results_table(metrics: Dict[str, Any], labels: List[str] = None) -> pd.DataFrame:
        if labels is None:
            labels = ['No Category', 'Explicit', 'Implicit', 'Report', 'Intensify', 'Counter', 'Escalate']
        rows = []
        for i, lab in enumerate(labels):
            if i < len(metrics['precision_per_class']):
                rows.append({
                    'Class': lab,
                    'Precision': f"{metrics['precision_per_class'][i]:.3f}",
                    'Recall': f"{metrics['recall_per_class'][i]:.3f}",
                    'F1-Score': f"{metrics['f1_per_class'][i]:.3f}",
                    'Support': int(metrics['support_per_class'][i])
                })
        total_support = int(sum(metrics['support_per_class']))
        rows += [
            {'Class': 'Macro Avg', 'Precision': f"{metrics['precision_macro']:.3f}", 'Recall': f"{metrics['recall_macro']:.3f}", 'F1-Score': f"{metrics['f1_macro']:.3f}", 'Support': total_support},
            {'Class': 'Weighted Avg', 'Precision': f"{metrics['precision_weighted']:.3f}", 'Recall': f"{metrics['recall_weighted']:.3f}", 'F1-Score': f"{metrics['f1_weighted']:.3f}", 'Support': total_support},
            {'Class': 'Micro Avg', 'Precision': f"{metrics['precision_micro']:.3f}", 'Recall': f"{metrics['recall_micro']:.3f}", 'F1-Score': f"{metrics['f1_micro']:.3f}", 'Support': total_support},
        ]
        return pd.DataFrame(rows)

def run_zero_shot(sample_df: pd.DataFrame, model: str = "model_name", prompt: str = None, delay: float = 0.0, save: bool = True, out_dir: str = "zero_outputs"):
    analyzer = SocialMediaAnalyzerZeroShot(model=model, prompt=prompt)
    results_df = analyzer.process_dataframe(sample_df, delay=delay)
    metrics = analyzer.calculate_metrics(
        y_true=results_df['intention_label'].tolist(),
        y_pred=results_df['predicted_label'].tolist()
    )
    table = analyzer.create_results_table(metrics)

    if save:
        save_metrics(results_df, metrics, table, out_dir=out_dir)

    return results_df, metrics, table

def save_metrics(results_df: pd.DataFrame, metrics: Dict[str, Any], table: pd.DataFrame, out_dir: str = "zero_outputs") -> None:
    """
    Save metrics and results table:
      - metrics.json: full dictionary including per-class arrays, classification_report, confusion_matrix
      - results_table.csv: summary precision/recall/F1 table
      - confusion_matrix.csv: raw confusion matrix
    """
    os.makedirs(out_dir, exist_ok=True)

    metrics_json_safe = _to_json_safe(metrics)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_json_safe, f, ensure_ascii=False, indent=2)

    table.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    results_df.to_csv(os.path.join(out_dir, "data_with_prediction.csv"), index=False)

    conf = metrics.get("confusion_matrix")
    if conf is not None:
        conf_df = pd.DataFrame(conf)
        conf_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=False)



# Call the function
incivi_dev = pd.read_csv("./incivi_dev.csv")
incivi_test = pd.read_csv("./incivi_test.csv")

# Zero-shot
zero_results, zero_metrics, zero_table = run_zero_shot(incivi_test, model="model_name", prompt=ZERO_SHOT_SYSTEM_PROMPT, delay=0.0, out_dir='model_zero_outputs')

# CoT-zero 
cot_results, cot_metrics, cot_table = run_zero_shot(incivi_test, model="model_name", prompt=COT_SYSTEM_PROMPT, delay=0.0, out_dir='model_cot_outputs')