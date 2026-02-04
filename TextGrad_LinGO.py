import re
import time
import random
import json
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
from tqdm import trange
from litellm import completion
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)

import textgrad as tg

## Set up API keys 

os.environ['OPENAI_API_KEY'] = "YOUR API KEY"
os.environ['GEMINI_API_KEY'] = "YOUR API KEY"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY"

## Provide prompt and definitions

LINGO_SYSTEM_PROMPT_BASE = """You are an annotator interpreting the intent of uncivil Portuguese social media discourse about Brazilian politics on X (formerly Twitter).

Target Category: Always taken from the first square-bracket tag (e.g., [Impoliteness], [Hate Speech and Stereotyping], [Physical Harm and Violent Political Rhetoric], [Threats to Democratic Institutions and Values])

Category Definitions:
- Impoliteness: messages including rudeness/disrespect (name-calling, aspersions, calling someone a liar, hyperbole, non-cooperation, pejoratives, vulgarity, belittling, shouting via ALL-CAPS or excessive "!" when context supports it).
- Hate Speech and Stereotyping: discriminatory content targeting protected groups (gender identity, sexual orientation, religion, race, nationality, ideology, disability); over-generalizations, out-group demeaning; (do NOT include purely individual appearance insults unless tied to group identity).
- Physical Harm and Violent Political Rhetoric: threats/advocacy/praise of physical harm or violence; direct or metaphorical calls for harm; justification of violence for political ends.
- Threats to Democratic Institutions and Values: advocacy or approval of actions undermining elections/institutions/rule of law/press freedom/civil rights; promotion of autocracy; unfounded claims that delegitimize institutions.

Decision Process:
STEP 1: Check Reference.
Question: Does the text refer to another person's statement or behavior?
If NO → go to STEP 4 (Original Content).
If YES → go to STEP 2 (Referenced Content).

STEP 2: Check Referenced Content.
Question: Does the referenced statement or behavior contain explicit or implicit {{CATEGORY}}?
If NO → go to STEP 4 (Analyze Original Content).
If YES → go to STEP 3 (Stance Analysis).

STEP 3: Stance Toward Referenced Content.
Question: How does the author respond to the referenced {{CATEGORY}}?
Report (3): mentions without opinion.
Intensify (4): agrees or amplifies (e.g., "Exato!", supportive emojis).
Counter (5): criticizes or disagrees (e.g., "Isso é errado", "Que absurdo").
Escalate (6): responds to {{CATEGORY}} content with {{CATEGORY}}.

STEP 4: Check Original Content.
Question: Does the author's own text contain explicit or implicit {{CATEGORY}}?
If NO → Label 0 (No {{CATEGORY}}).
If YES → go to STEP 5 (Type Classification).

STEP 5: Type Classification (Direct Content).
Question: Is the {{CATEGORY}} expressed directly or indirectly?
Explicit (1): direct, overt {{CATEGORY}} (e.g., clear insults, vulgarity, direct threats).
Implicit (2): indirect, veiled {{CATEGORY}} (e.g., sarcasm, irony, subtle attacks).

Return ONLY valid JSON in this format:
{
  "LABEL": <integer 0-6>,
  "REASONING": {
    "STEP 1": "YES" or "NO",
    <then include ONLY the steps you actually followed based on the branching logic>
  }
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
"""

STEP_KEYS = ["STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5"]

## Step-specific prompt sections that can be optimized

STEP_PROMPT_SECTIONS = {
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
Implicit (2): indirect, veiled {{CATEGORY}} (e.g., sarcasm, irony, subtle attacks)."""
}

CATEGORY_DEFINITIONS = {
    "Impoliteness": """Messages including rudeness/disrespect (name-calling, aspersions, 
    calling someone a liar, hyperbole, non-cooperation, pejoratives, vulgarity, 
    belittling, shouting via ALL-CAPS or excessive "!" when context supports it).""",
    
    "Hate Speech and Stereotyping": """Harmful or discriminatory content targeting 
    protected groups (gender identity, sexual orientation, religion, race, nationality, 
    ideology, disability); over-generalizations, out-group demeaning.""",
    
    "Physical Harm and Violent Political Rhetoric": """Threats/advocacy/praise of 
    physical harm or violence; direct or metaphorical calls for harm; justification 
    of violence for political ends.""",
    
    "Threats to Democratic Institutions and Values": """Advocacy or approval of 
    actions undermining elections/institutions/rule of law/press freedom/civil rights; 
    promotion of autocracy; unfounded claims that delegitimize institutions."""
}

LABEL_DESCRIPTIONS = {
    0: "Other - does not fit any pattern",
    1: "Explicit - direct, overt expression",
    2: "Implicit - indirect, veiled expression",
    3: "Report - quotes/refers without opinion",
    4: "Intensify - quotes and agrees/amplifies",
    5: "Counter - quotes and criticizes/disagrees",
    6: "Escalate - responds with more of the same"
}

## Define functions for data processing

def _strip_parentheses(text: str) -> str:
    """Remove any (...) blocks."""
    return re.sub(r"\([^)]*\)", "", text or "")


def _normalize_ans(text: str) -> str:
    """Normalize whitespace and dashes, after removing parentheses content."""
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
    Supports:
    (A) New JSON format:
        {"LABEL": 0-6, "REASONING": {"STEP 1": "...", "STEP 2": "...", "STEP 3": null, ...}}
        (steps may be missing or null)
    (B) Legacy bracket/arrow format:
        REASONING: [STEP 1: ... → STEP 2: ... → ...]
    """
    if not reasoning_text:
        return {}

    text = str(reasoning_text).strip()

    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        # salvage first {...} block if extra text wraps JSON
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None

    if isinstance(obj, dict):
        reasoning_obj = obj.get("REASONING", None)

        # allow passing just the reasoning dict
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


def earliest_step_diff(pred_steps: Dict[str, str], gold_steps: Dict[str, str], 
                       pred_label: int, gold_label: int) -> Optional[str]:
    """Return the earliest step name where pred != gold, else None."""
    for step in STEP_KEYS:
        if step in pred_steps or step in gold_steps:
            pv = pred_steps.get(step, "")
            gv = gold_steps.get(step, "")
            if (pv != gv) and (pred_label != gold_label):
                return step
    return None


def _normalize_step_answer(answer: str) -> str:
    """Normalize step answer for pattern matching. Returns 'MISSING' if empty."""
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


def parse_digit_0_6(text: str) -> str:
    """Extract a single label in {0..6}. Fallback to '0'."""
    if text is None:
        return "0"
    m = re.search(r"\b([0-6])\b", str(text).strip())
    return m.group(1) if m else "0"


def split_train_val(df: pd.DataFrame, val_ratio: float = 0.2, 
                    seed: int = 42, stratify_col: str = "intention_label") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into training and validation sets.
    """
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


## TextGrad-based Step Optimizer

class TextGradStepOptimizer:
    """
    Uses TextGrad to optimize specific steps in the prompt that are identified as problematic.
    """
    
    def __init__(self, 
                 forward_model: str = "gpt-5-mini",
                 backward_model: str = "gpt-5-mini",
                 base_prompt: str = LINGO_SYSTEM_PROMPT_BASE):
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.base_prompt = base_prompt
        self.optimized_step_sections: Dict[str, str] = {}
        
        self.forward_engine = tg.get_engine(f"experimental:{forward_model}")
        tg.set_backward_engine(f"experimental:{backward_model}", override=True)
        
    def _build_prompt_with_optimized_steps(self, optimized_sections: Dict[str, str] = None) -> str:
        """
        Build the full prompt, replacing step sections with optimized versions.
        """
        prompt = self.base_prompt
        sections_to_use = optimized_sections or self.optimized_step_sections
        
        for step, optimized_section in sections_to_use.items():
            if step in STEP_PROMPT_SECTIONS:
                original_section = STEP_PROMPT_SECTIONS[step]
                prompt = prompt.replace(original_section, optimized_section)
        
        return prompt
    
    def _create_step_specific_loss_instruction(self, target_steps: List[str], 
                                                error_analysis: Dict[str, Dict[str, Any]]) -> str:
        """
        Create a loss instruction that focuses on the targeted steps.
        """
        step_focus = []
        for step in target_steps:
            if step in error_analysis:
                patterns = error_analysis[step].get("most_common_errors", [])
                if patterns:
                    pattern_str = ", ".join([f"'{p}'" for p, _ in patterns[:3]])
                    step_focus.append(f"- {step}: Common mistakes include {pattern_str}")
        
        step_guidance = "\n".join(step_focus) if step_focus else "Focus on the targeted steps."
        
        return f"""You are a strict evaluator for a Portuguese social media intent classification task.

You will be given:
- POST (the raw post text; the FIRST [bracket tag] indicates the target category)
- GOLD (an integer 0-6 representing the correct label)
- PRED (model output - should be JSON with LABEL and REASONING)

TARGETED STEPS TO EVALUATE:
{step_guidance}

Evaluation criteria:
1. Extract the LABEL from PRED (look for "LABEL": <number> in JSON or a digit 0-6)
2. Compare with GOLD label

Return ONLY:
0 if the predicted label equals GOLD exactly;
1 otherwise.

No other text."""

    def optimize_steps(self, 
                       train_data: List[Tuple[str, int]],
                       target_steps: List[str],
                       error_analysis: Dict[str, Dict[str, Any]],
                       epochs: int = 3,
                       batch_size: int = 6,
                       max_batches_per_epoch: int = 30) -> Dict[str, str]:
        """
        Optimize only the targeted step sections using TextGrad.
        """
        if not target_steps:
            print("No target steps specified, skipping optimization")
            return {}
        
        print(f"\n{'='*60}")
        print(f"TextGrad Optimization for steps: {target_steps}")
        print(f"{'='*60}")
        
        step_variables: Dict[str, tg.Variable] = {}
        for step in target_steps:
            if step in STEP_PROMPT_SECTIONS:
                step_variables[step] = tg.Variable(
                    STEP_PROMPT_SECTIONS[step],
                    requires_grad=True,
                    role_description=f"instruction for {step} in the classification decision process"
                )
        
        if not step_variables:
            print("No valid step sections to optimize")
            return {}
        
        def build_current_prompt() -> str:
            current_sections = {step: var.value for step, var in step_variables.items()}
            return self._build_prompt_with_optimized_steps(current_sections)
        
        current_prompt_var = tg.Variable(
            build_current_prompt(),
            requires_grad=True,
            role_description="system prompt to guide the LLM to output the correct intent label (0-6)"
        )
        
        model = tg.BlackboxLLM(self.forward_engine, system_prompt=current_prompt_var)
        
        optimizer = tg.TGD(parameters=list(step_variables.values()))
        
        loss_instruction = self._create_step_specific_loss_instruction(target_steps, error_analysis)
        loss_fn = tg.TextLoss(loss_instruction)
        
        random.seed(42)
        
        for epoch in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
            batches = batches[:min(len(batches), max_batches_per_epoch)]
            
            batch_errs = []
            for batch_idx, batch in enumerate(batches):
                losses = []
                errs = 0
                
                current_prompt_var.value = build_current_prompt()
                
                for post, gold in batch:
                    q = tg.Variable(
                        f"POST:\n{post}\n\nProvide your analysis:",
                        role_description="input post",
                        requires_grad=False,
                    )
                    
                    pred = model(q)
                    pred_text = getattr(pred, "value", str(pred))
                    
                    pred_label = parse_digit_0_6(pred_text)
                    
                    eval_pack = tg.Variable(
                        f"POST:\n{post}\n\nGOLD: {gold}\nPRED: {pred_label}",
                        role_description="evaluation pack",
                        requires_grad=False,
                    )
                    
                    loss = loss_fn(eval_pack)
                    losses.append(loss)
                    
                    errs += 1 if str(getattr(loss, "value", loss)).strip().startswith("1") else 0
                
                for loss in losses:
                    loss.backward()
                
                optimizer.step()
                
                batch_errs.append(errs / max(1, len(batch)))
            
            mean_err = sum(batch_errs) / len(batch_errs) if batch_errs else 0
            print(f"  [epoch {epoch+1}/{epochs}] mean batch error: {mean_err:.3f}")
        
        optimized_sections = {step: var.value for step, var in step_variables.items()}
        self.optimized_step_sections.update(optimized_sections)
        
        print(f"\nOptimized {len(optimized_sections)} step sections")
        for step, section in optimized_sections.items():
            print(f"\n--- Optimized {step} ---")
            print(section[:200] + "..." if len(section) > 200 else section)
        
        return optimized_sections
    
    def get_current_prompt(self) -> str:
        """Get the current prompt with any optimized sections."""
        return self._build_prompt_with_optimized_steps()


## Define class module for analysis

class SocialMediaAnalyzerTextGrad:
    """
    Social media analyzer that uses TextGrad-optimized prompts for targeted steps.
    """
    
    def __init__(self, model: str, 
                 system_prompt: str = None,
                 optimized_step_sections: Dict[str, str] = None):
        self.model = model
        self.base_system_prompt = system_prompt or LINGO_SYSTEM_PROMPT_BASE
        self.optimized_step_sections = optimized_step_sections or {}
        
    def _build_prompt(self) -> str:
        """Build the prompt with optimized step sections."""
        prompt = self.base_system_prompt
        
        for step, optimized_section in self.optimized_step_sections.items():
            if step in STEP_PROMPT_SECTIONS:
                original_section = STEP_PROMPT_SECTIONS[step]
                prompt = prompt.replace(original_section, optimized_section)
        
        return prompt
    
    def get_prediction(self, content: str, max_retries: int = 3) -> Tuple[int, str]:
        """
        Get prediction with retry if output is invalid or missing required fields.
        """
        prompt = self._build_prompt()
        
        for attempt in range(max_retries):
            try:
                resp = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content}
                    ]
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
                        except:
                            obj = None
                
                if isinstance(obj, dict):
                    label = int(obj.get("LABEL", -1))
                    reasoning_obj = obj.get("REASONING", {})
                    
                    # Check if STEP 1 exists (required for all responses)
                    if "STEP 1" not in reasoning_obj:
                        print(f"  Retry {attempt + 1}/{max_retries}: Missing STEP 1")
                        continue
                    
                    # Check if label is valid
                    if label < 0 or label > 6:
                        print(f"  Retry {attempt + 1}/{max_retries}: Invalid label {label}")
                        continue
                    
                    # Valid response
                    reasoning = json.dumps(reasoning_obj, ensure_ascii=False)
                    return label, reasoning
                else:
                    print(f"  Retry {attempt + 1}/{max_retries}: Failed to parse JSON")
                    continue
                    
            except Exception as e:
                print(f"  Retry {attempt + 1}/{max_retries}: Error - {str(e)}")
                continue
        
        # All retries failed
        print(f"  All {max_retries} retries failed, returning default")
        return 0, '{"error": "All retries failed"}'

    def process_dataframe_cot(self, df: pd.DataFrame, delay: float = 0.0,
                              desc: str = "Processing", max_retries: int = 3) -> pd.DataFrame:
        preds, reasons = [], []
        
        for idx, row in df.iterrows():
            if 'content' not in row:
                raise ValueError("DataFrame must include a 'content' column.")
            label, r = self.get_prediction(row['content'], max_retries=max_retries)
            preds.append(label)
            reasons.append(r)
            if delay:
                time.sleep(delay)
                
        out = df.copy()
        out['predicted_label'] = preds
        out['reasoning'] = reasons
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

        p_pc, r_pc, f_pc, s_pc = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
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
            {'Class': 'Macro Avg', 'Precision': f"{metrics['precision_macro']:.3f}", 
             'Recall': f"{metrics['recall_macro']:.3f}", 'F1-Score': f"{metrics['f1_macro']:.3f}", 
             'Support': total_support},
            {'Class': 'Weighted Avg', 'Precision': f"{metrics['precision_weighted']:.3f}", 
             'Recall': f"{metrics['recall_weighted']:.3f}", 'F1-Score': f"{metrics['f1_weighted']:.3f}", 
             'Support': total_support},
            {'Class': 'Micro Avg', 'Precision': f"{metrics['precision_micro']:.3f}", 
             'Recall': f"{metrics['recall_micro']:.3f}", 'F1-Score': f"{metrics['f1_micro']:.3f}", 
             'Support': total_support},
        ]
        return pd.DataFrame(rows)


## Step difference analysis

def compute_step_diff_distribution(df: pd.DataFrame) -> Tuple[Counter, List[Dict[str, Any]]]:
    """
    Compute distribution of earliest differing steps between predictions and gold.
    """
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
            records.append({
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


## Evaluation function

def evaluate_on_dataset(df: pd.DataFrame,
                        analyzer: SocialMediaAnalyzerTextGrad,
                        gold_reason_col: str = "reason",
                        delay: float = 0.0,
                        desc: str = "Evaluating",
                        max_retries: int = 3) -> Dict[str, Any]:
    """
    Run predictions and compute metrics including step-level analysis.
    """
    results_df = analyzer.process_dataframe_cot(df, delay=delay, desc=desc, max_retries=max_retries)
    print(results_df["reasoning"].head(20).tolist())
    
    metrics = analyzer.calculate_metrics(
        y_true=results_df['intention_label'].tolist(),
        y_pred=results_df['predicted_label'].tolist()
    )
    
    merged = results_df.copy()
    if gold_reason_col in df.columns:
        merged['reason'] = df[gold_reason_col].astype(str)
    else:
        merged['reason'] = ""
    
    dist, recs = compute_step_diff_distribution(merged)
    total = len(merged)
    diff_total = sum(dist.values())
    step_diff_rate = diff_total / total if total else 0.0
    
    return {
        "results_df": results_df,
        "metrics": metrics,
        "step_diff_distribution": dict(dist),
        "step_records": recs,
        "step_diff_rate": step_diff_rate
    }


## Iterative TextGrad refinement

def iterative_textgrad_refinement(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forward_model: str = "gpt-5-mini",
    backward_model: str = "gpt-5-mini",
    rounds: int = 3,
    delay: float = 0.0,
    seed: int = 42,
    step_threshold: float = 0.1,
    val_ratio: float = 0.2,
    textgrad_epochs: int = 3,
    textgrad_batch_size: int = 6,
    textgrad_max_batches: int = 30
) -> Dict[str, Any]:
    """
    Iteratively refine classification using TextGrad to optimize problematic steps.
    
    Args:
        forward_model: Model used for making predictions 
        backward_model: Model used for generating gradients/feedback
    """
    random.seed(seed)
    
    print("="*60)
    print("STEP 1: Splitting dev set into TRAIN and VAL")
    print("="*60)
    
    train_df, val_df = split_train_val(dev_df, val_ratio=val_ratio, seed=seed)
    
    print(f"  Original dev set: {len(dev_df)} samples")
    print(f"  TRAIN set: {len(train_df)} samples (for TextGrad optimization)")
    print(f"  VAL set: {len(val_df)} samples (for step error analysis)")
    print(f"  TEST set: {len(test_df)} samples (final evaluation only)")
    print(f"  Forward model: {forward_model}")
    print(f"  Backward model: {backward_model}")
    
    # Prepare training data for TextGrad
    train_data = list(zip(
        train_df["content"].astype(str).tolist(),
        train_df["intention_label"].astype(int).tolist()
    ))
    
    print("\n" + "="*60)
    print("STEP 2: Initializing TextGrad optimizer")
    print("="*60)
    
    textgrad_optimizer = TextGradStepOptimizer(
        forward_model=forward_model,
        backward_model=backward_model,
        base_prompt=LINGO_SYSTEM_PROMPT_BASE
    )
    
    print("\n" + "="*60)
    print("STEP 3: Iterative refinement using VAL set")
    print("="*60)
    
    history: List[Dict[str, Any]] = []
    target_steps: List[str] = [] 
    error_analysis: Dict[str, Dict[str, Any]] = {}
    cumulative_optimized_sections: Dict[str, str] = {}
    
    val_eval = None
    best_val_f1 = -1
    best_optimized_sections = {}
    best_round = 0
    
    for r in trange(rounds + 1, desc="TextGrad Refinement Rounds"):
        print(f"\n{'='*60}")
        print(f"Round {r}: Target steps = {target_steps if target_steps else 'None (baseline)'}")
        print('='*60)
        
        # If we have target steps, optimize them using TextGrad
        if target_steps and r > 0:
            print(f"\nOptimizing steps {target_steps} with TextGrad...")
            new_optimized_sections = textgrad_optimizer.optimize_steps(
                train_data=train_data,
                target_steps=target_steps,
                error_analysis=error_analysis,
                epochs=textgrad_epochs,
                batch_size=textgrad_batch_size,
                max_batches_per_epoch=textgrad_max_batches
            )
            # Accumulate optimized sections
            cumulative_optimized_sections.update(new_optimized_sections)
        
        # Create analyzer with current optimized sections
        # Use forward_model for inference
        analyzer = SocialMediaAnalyzerTextGrad(
            model=forward_model,
            system_prompt=LINGO_SYSTEM_PROMPT_BASE,
            optimized_step_sections=cumulative_optimized_sections
        )
        
        print("Evaluating on VAL set...")
        val_eval = evaluate_on_dataset(val_df, analyzer, delay=delay, desc="VAL")
        
        current_f1 = val_eval["metrics"]["f1_weighted"]
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_optimized_sections = dict(cumulative_optimized_sections)
            best_round = r
        
        history.append({
            "round": r,
            "target_steps": list(target_steps),
            "optimized_steps": list(cumulative_optimized_sections.keys()),
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
        
        val_dist_counter = Counter(val_eval["step_diff_distribution"])
        
        if sum(val_dist_counter.values()) == 0:
            print("No step differences found - stopping early")
            break
        
        # Identify problematic steps for next round
        target_steps, error_analysis = identify_problematic_steps_with_patterns(
            val_dist_counter, 
            val_eval["step_records"],
            step_threshold
        )
        
        print(f"Identified problematic steps for next round: {target_steps}")
        for step, analysis in error_analysis.items():
            print(f"  {step} error patterns: {analysis.get('most_common_errors', [])[:3]}")
    
    print("\n" + "="*60)
    print(f"STEP 4: Final evaluation on TEST set")
    print(f"  Using best configuration from round {best_round}")
    print(f"  Best optimized steps: {list(best_optimized_sections.keys()) if best_optimized_sections else 'None (baseline)'}")
    print("="*60)
    
    final_analyzer = SocialMediaAnalyzerTextGrad(
        model=forward_model,
        system_prompt=LINGO_SYSTEM_PROMPT_BASE,
        optimized_step_sections=best_optimized_sections
    )
    
    print("Evaluating on TEST set...")
    test_eval = evaluate_on_dataset(test_df, final_analyzer, delay=delay, desc="TEST")
    
    print(f"\nFINAL TEST RESULTS:")
    print(f"  Accuracy: {test_eval['metrics']['accuracy']:.4f}")
    print(f"  F1 (weighted): {test_eval['metrics']['f1_weighted']:.4f}")
    
    return {
        "history": history,
        "best_round": best_round,
        "best_optimized_sections": best_optimized_sections,
        "best_val_f1": best_val_f1,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "forward_model": forward_model,
        "backward_model": backward_model,
        "final_val_results": val_eval["results_df"] if val_eval else None,
        "final_test_results": test_eval["results_df"],
        "final_val_metrics": val_eval["metrics"] if val_eval else None,
        "final_test_metrics": test_eval["metrics"],
        "final_test_step_diff_distribution": test_eval["step_diff_distribution"],
        "final_optimized_prompt": final_analyzer._build_prompt()
    }


## Export functions

def export_textgrad_refinement_artifacts(
    run_result: Dict[str, Any],
    out_dir: str = "lingo_textgrad_outputs",
    history_csv_name: str = "history_metrics.csv"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    history = run_result.get("history", [])
    
    if not history:
        raise ValueError("No history found in run_result.")
    
    csv_path = os.path.join(out_dir, history_csv_name)
    rows = []
    for h in history:
        rows.append({
            "round": h["round"],
            "target_steps": ", ".join(h["target_steps"]) if h["target_steps"] else "None",
            "optimized_steps": ", ".join(h["optimized_steps"]) if h["optimized_steps"] else "None",
            "val_accuracy": h["val_accuracy"],
            "val_f1_macro": h["val_f1_macro"],
            "val_f1_weighted": h["val_f1_weighted"],
            "val_f1_micro": h["val_f1_micro"],
            "val_step_diff_rate": h["val_step_diff_rate"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved metrics history to {csv_path}")
    
    if run_result.get("final_test_results") is not None:
        test_path = os.path.join(out_dir, "final_test_predictions.csv")
        run_result["final_test_results"].to_csv(test_path, index=False, encoding="utf-8")
        print(f"Saved test predictions to {test_path}")
    
    # Save optimized prompt
    if run_result.get("final_optimized_prompt"):
        prompt_path = os.path.join(out_dir, "optimized_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(run_result["final_optimized_prompt"])
        print(f"Saved optimized prompt to {prompt_path}")
    
    # Save optimized step sections
    if run_result.get("best_optimized_sections"):
        sections_path = os.path.join(out_dir, "optimized_step_sections.json")
        with open(sections_path, "w", encoding="utf-8") as f:
            json.dump(run_result["best_optimized_sections"], f, indent=2, ensure_ascii=False)
        print(f"Saved optimized step sections to {sections_path}")
    
    summary_path = os.path.join(out_dir, "summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("LinGO TextGrad Refinement Summary (Train/Val/Test Split)\n")
        f.write("="*60 + "\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Forward Model: {run_result.get('forward_model', 'N/A')}\n")
        f.write(f"  Backward Model: {run_result.get('backward_model', 'N/A')}\n\n")
        
        f.write("DATA SPLIT:\n")
        f.write(f"  TRAIN: {run_result.get('train_size', 'N/A')} samples (for TextGrad optimization)\n")
        f.write(f"  VAL: {run_result.get('val_size', 'N/A')} samples (for step error analysis)\n")
        f.write(f"  TEST: {run_result.get('test_size', 'N/A')} samples (final evaluation)\n\n")
        
        f.write(f"Best Round: {run_result.get('best_round', 'N/A')}\n")
        f.write("Best Optimized Steps: " + 
                (", ".join(run_result.get("best_optimized_sections", {}).keys()) or "None (baseline)") + "\n")
        f.write(f"Best VAL F1 (weighted): {run_result.get('best_val_f1', 0):.4f}\n\n")
        
        f.write("VAL Metrics by Round:\n")
        f.write("-"*60 + "\n")
        for h in history:
            f.write(f"\nRound {h['round']}:\n")
            f.write(f"  Target Steps: {h['target_steps'] or 'None (baseline)'}\n")
            f.write(f"  Optimized Steps: {h['optimized_steps'] or 'None'}\n")
            f.write(f"  VAL - Acc: {h['val_accuracy']:.4f}, F1-weighted: {h['val_f1_weighted']:.4f}\n")
            f.write(f"  Step Diff Distribution: {h['val_step_diff_distribution']}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("FINAL TEST RESULTS:\n")
        f.write("="*60 + "\n")
        test_metrics = run_result.get("final_test_metrics", {})
        f.write(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  F1 (weighted): {test_metrics.get('f1_weighted', 0):.4f}\n")
        f.write(f"  Step Diff Distribution: {run_result.get('final_test_step_diff_distribution', {})}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("OPTIMIZED STEP SECTIONS:\n")
        f.write("="*60 + "\n")
        for step, section in run_result.get("best_optimized_sections", {}).items():
            f.write(f"\n--- {step} ---\n")
            f.write(section + "\n")
    
    print(f"Saved summary report to {summary_path}")
    
    return out_dir


def run_textgrad_optimization(
    dev_filepath: str,
    test_filepath: str,
    forward_model: str = "gpt-5-mini",
    backward_model: str = "gpt-5-mini",
    rounds: int = 3,
    delay: float = 0.0,
    step_threshold: float = 0.1,
    val_ratio: float = 0.2,
    seed: int = 42,
    output_dir: str = "lingo_textgrad_outputs",
    textgrad_epochs: int = 3,
    textgrad_batch_size: int = 6,
    textgrad_max_batches: int = 30
) -> Dict[str, Any]:
    print(f"Loading data...")
    dev_df = pd.read_csv(dev_filepath)
    test_df = pd.read_csv(test_filepath)
    print(f"  DEV (to be split): {len(dev_df)} samples")
    print(f"  TEST (held out): {len(test_df)} samples")
    print(f"  Forward model: {forward_model}")
    print(f"  Backward model: {backward_model}")
    
    result = iterative_textgrad_refinement(
        dev_df=dev_df,
        test_df=test_df,
        forward_model=forward_model,
        backward_model=backward_model,
        rounds=rounds,
        delay=delay,
        seed=seed,
        step_threshold=step_threshold,
        val_ratio=val_ratio,
        textgrad_epochs=textgrad_epochs,
        textgrad_batch_size=textgrad_batch_size,
        textgrad_max_batches=textgrad_max_batches
    )
    
    export_textgrad_refinement_artifacts(result, out_dir=output_dir)
    
    return result


## Final function call

if __name__ == "__main__":
    
    result = run_textgrad_optimization(
        dev_filepath="./incivi_dev.csv",
        test_filepath="./incivi_test.csv",
        forward_model="gemini/gemini-2.5-flash-lite",  # gpt-5-mini, anthropic/claude-3-haiku-20240307
 
        backward_model="gpt-5-mini", 
        rounds=5,
        delay=0.0,
        step_threshold=0.1,  # Steps with >10% of errors are targeted
        val_ratio=0.2,       # 20% of dev for validation   
        seed=42,
        output_dir="textgrad_lingo_outputs",
        textgrad_epochs=3,       # TextGrad optimization epochs per round
        textgrad_batch_size=6,   # Batch size for TextGrad
        textgrad_max_batches=30  # Max batches per epoch
    )
