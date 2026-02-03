import re
import time
import random
import json
import os
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set

from tqdm import trange, tqdm
from litellm import completion

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)

from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY" 
os.environ["OPENAI_API_KEY"] = "YOUR API KEY" 
os.environ["GEMINI_API_KEY"] = "YOUR API KEY"
os.environ["LANGSMITH_API_KEY"] = "YOUR API KEY" 

## PROMPT + CONSTANTS

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

STEP 5: Type Classification.
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


# Define formating functions

def _strip_parentheses(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", text or "")

def _normalize_ans(text: str) -> str:
    t = _strip_parentheses(text)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("–", "-").replace("—", "-")
    return t

def extract_category(content: str) -> str:
    match = re.match(r'\[([^\]]+)\]', content.strip())
    return match.group(1) if match else "Unknown"

def extract_text(content: str) -> str:
    return re.sub(r'^\[[^\]]+\]\s*', '', content.strip())

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

def _normalize_step_answer(answer: str) -> str:
    if answer is None:
        return "MISSING"
    ans_raw = str(answer).strip()
    if not ans_raw or ans_raw.lower() in {"n/a", "na", "none", "null", ""}:
        return "MISSING"

    ans = ans_raw.upper().strip()
    if ans in ["YES", "SIM", "Y", "S"]:
        return "YES"
    if ans in ["NO", "NÃO", "NAO", "N"]:
        return "NO"
    if "EXPLICIT" in ans or "EXPLÍCIT" in ans:
        return "EXPLICIT"
    if "IMPLICIT" in ans or "IMPLÍCIT" in ans:
        return "IMPLICIT"
    if "REPORT" in ans or "RELAT" in ans:
        return "REPORT"
    if "INTENSIF" in ans:
        return "INTENSIFY"
    if "COUNTER" in ans or "CONTRA" in ans:
        return "COUNTER"
    if "ESCALAT" in ans:
        return "ESCALATE"
    return ans

def earliest_step_diff(
    pred_steps: Dict[str, str],
    gold_steps: Dict[str, str],
    pred_label: int,
    gold_label: int
) -> Optional[str]:
    for step in STEP_KEYS:
        if step in pred_steps or step in gold_steps:
            pv = _normalize_step_answer(pred_steps.get(step, ""))
            gv = _normalize_step_answer(gold_steps.get(step, ""))
            if (pv != gv) and (pred_label != gold_label):
                return step
    return None

def split_train_val(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify_col: str = "intention_label"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=seed, stratify=stratify
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# RAG store

class RAGExampleStore:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model_name = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.documents: List[Document] = []
        self.step_index: Dict[str, List[int]] = {step: [] for step in STEP_KEYS}
        self.train_df: Optional[pd.DataFrame] = None

    def index_training_data(self, train_df: pd.DataFrame) -> int:
        self.train_df = train_df.copy()
        self.documents = []
        self.step_index = {step: [] for step in STEP_KEYS}
        self.vector_store = InMemoryVectorStore(self.embeddings)

        for idx, row in train_df.iterrows():
            content = row.get("content", "")
            label = int(row.get("intention_label", 0))
            reason = str(row.get("reason", ""))

            category = extract_category(content)
            text = extract_text(content)
            step_answers = extract_step_answers(reason)

            doc = Document(
                page_content=text,
                metadata={
                    "index": int(idx),
                    "doc_id": int(len(self.documents)),
                    "category": category,
                    "label": label,
                    "original_content": content,
                    "reason": reason,
                    "step_answers": step_answers,
                    "formatted": self._format_example(content, label, reason),
                },
            )
            self.documents.append(doc)

            for step in step_answers.keys():
                if step in self.step_index:
                    self.step_index[step].append(doc.metadata["doc_id"])

        document_ids = self.vector_store.add_documents(documents=self.documents)
        print(f"Indexed {len(document_ids)} TRAIN documents in vector store")
        return len(document_ids)

    def _format_example(self, content: str, label: int, reason: str) -> str:
        return f"""Post: "{extract_text(content)}"
Category: [{extract_category(content)}]
Label: {label}
Reasoning: {reason}"""

    def _diversify_and_trim(self, docs: List[Document], k: int) -> List[Document]:
        seen_labels = set()
        diverse, rest = [], []
        for d in docs:
            lab = d.metadata.get("label")
            if lab not in seen_labels:
                diverse.append(d)
                seen_labels.add(lab)
            else:
                rest.append(d)
        return (diverse + rest)[:k]

    def retrieve_from_targeted_steps(
        self,
        query: str,
        category: str,
        target_steps: List[str],
        k_final: int = 5,
        k_candidates: int = 20,
        seen_train_row_ids: Optional[Set[int]] = None,
    ) -> Tuple[str, List[Document], Set[int]]:
        """
        Retrieve examples that have answers for target steps, 
        ranked by similarity to query.
        """
        if seen_train_row_ids is None:
            seen_train_row_ids = set()

        if not target_steps:
            return "", [], seen_train_row_ids

        allowed_doc_ids: Set[int] = set()
        for step in target_steps:
            allowed_doc_ids.update(self.step_index.get(step, []))

        if not allowed_doc_ids:
            return "", [], seen_train_row_ids

        raw = self.vector_store.similarity_search(query, k=k_candidates)

        filtered = [
            d for d in raw
            if d.metadata.get("doc_id") in allowed_doc_ids
            and int(d.metadata.get("index")) not in seen_train_row_ids
        ]

        cat_docs = [d for d in filtered if d.metadata.get("category") == category]
        if cat_docs:
            filtered = cat_docs

        final_docs = self._diversify_and_trim(filtered, k_final)

        for d in final_docs:
            seen_train_row_ids.add(int(d.metadata.get("index")))

        serialized = "\n\n---\n\n".join(
            d.metadata.get("formatted", d.page_content) for d in final_docs
        )
        return serialized, final_docs, seen_train_row_ids

    def retrieve_from_error_patterns(
        self,
        query: str,
        category: str,
        error_analysis: Dict[str, Dict[str, Any]],
        k_final: int = 5,
        k_candidates: int = 20,
        seen_train_row_ids: Optional[Set[int]] = None,
    ) -> Tuple[str, List[Document], Set[int]]:
        """
        Retrieve examples that demonstrate correct answers for common error patterns,
        ranked by similarity to query.
        """
        if seen_train_row_ids is None:
            seen_train_row_ids = set()

        if not error_analysis:
            return "", [], seen_train_row_ids

        step_targets: Dict[str, Set[str]] = {}
        for step, analysis in error_analysis.items():
            for pattern, _cnt in analysis.get("most_common_errors", [])[:5]:
                if "->" not in pattern:
                    continue
                _pred, gold = pattern.split("->", 1)
                step_targets.setdefault(step, set()).add(_normalize_step_answer(gold))

        allowed_doc_ids: Set[int] = set()
        for step, gold_targets in step_targets.items():
            for did in self.step_index.get(step, []):
                doc = self.documents[did]
                step_answers = doc.metadata.get("step_answers", {})
                gold_ans = _normalize_step_answer(step_answers.get(step, ""))
                if gold_ans in gold_targets:
                    allowed_doc_ids.add(did)

        if not allowed_doc_ids:
            return "", [], seen_train_row_ids

        raw = self.vector_store.similarity_search(query, k=k_candidates)

        filtered = [
            d for d in raw
            if d.metadata.get("doc_id") in allowed_doc_ids
            and int(d.metadata.get("index")) not in seen_train_row_ids
        ]

        cat_docs = [d for d in filtered if d.metadata.get("category") == category]
        if cat_docs:
            filtered = cat_docs

        final_docs = self._diversify_and_trim(filtered, k_final)

        for d in final_docs:
            seen_train_row_ids.add(int(d.metadata.get("index")))

        serialized = "\n\n---\n\n".join(
            d.metadata.get("formatted", d.page_content) for d in final_docs
        )
        return serialized, final_docs, seen_train_row_ids

# Analyzer function (retrieves 5 from targeted steps + 5 from error patterns = 10 total)

class SocialMediaAnalyzerRAG:
    def __init__(
        self,
        model: str,
        rag_store: RAGExampleStore,
        target_steps: Optional[List[str]] = None,
        error_analysis: Optional[Dict[str, Dict[str, Any]]] = None,
        num_examples_per_source: int = 5,
        system_prompt: Optional[str] = None,
        retrieval_candidates: int = 20,
    ):
        self.model = model
        self.rag_store = rag_store
        self.base_system_prompt = system_prompt or LINGO_SYSTEM_PROMPT_BASE
        self.target_steps = target_steps or []
        self.error_analysis = error_analysis or {}
        self.num_examples_per_source = num_examples_per_source
        self.retrieval_candidates = retrieval_candidates

    def _build_prompt(self, content: str) -> str:
        category = extract_category(content)
        text = extract_text(content)

        seen_train_row_ids: Set[int] = set()

        prompt = self.base_system_prompt

        if self.target_steps:
            step_guidance = "\n\n## ATTENTION: The following steps frequently cause errors. Pay careful attention:\n"
            for step in self.target_steps:
                step_guidance += f"- {step}: "
                if step in self.error_analysis:
                    patterns = self.error_analysis[step].get("most_common_errors", [])
                    if patterns:
                        patt = ", ".join([f"'{p}'" for p, _ in patterns[:2]])
                        step_guidance += f"Common mistakes: {patt}. "
                step_guidance += "Use the retrieved examples to answer the step correctly.\n"
            prompt += step_guidance

        targeted_examples_str = ""
        if self.target_steps:
            targeted_examples_str, _, seen_train_row_ids = self.rag_store.retrieve_from_targeted_steps(
                query=text,
                category=category,
                target_steps=self.target_steps,
                k_final=self.num_examples_per_source,
                k_candidates=self.retrieval_candidates,
                seen_train_row_ids=seen_train_row_ids,
            )

        # Retrieve 5 examples from error patterns (similarity-based)
        error_examples_str = ""
        if self.error_analysis:
            error_examples_str, _, seen_train_row_ids = self.rag_store.retrieve_from_error_patterns(
                query=text,
                category=category,
                error_analysis=self.error_analysis,
                k_final=self.num_examples_per_source,
                k_candidates=self.retrieval_candidates,
                seen_train_row_ids=seen_train_row_ids,
            )

        if targeted_examples_str:
            prompt += f"\n\n## EXAMPLES FROM TARGETED STEPS (demonstrating {', '.join(self.target_steps)}):\n{targeted_examples_str}\n"

        if error_examples_str:
            prompt += f"\n\n## EXAMPLES FROM ERROR PATTERNS (showing correct answers for common mistakes):\n{error_examples_str}\n"

        return prompt

    @staticmethod
    def _try_parse_label_and_reasoning(text: str) -> Optional[Tuple[int, str]]:
        if not text:
            return None

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
            try:
                label = int(obj.get("LABEL", -1))
            except Exception:
                label = -1
            reasoning_obj = obj.get("REASONING", {})

            if 0 <= label <= 6 and isinstance(reasoning_obj, dict) and ("STEP 1" in reasoning_obj):
                reasoning = json.dumps(reasoning_obj, ensure_ascii=False)
                return label, reasoning

        m1 = re.search(r'"LABEL"\s*:\s*([0-6])', text)
        if m1:
            label = int(m1.group(1))
            return label, json.dumps({"PARSE": "FALLBACK_LABEL_ONLY"}, ensure_ascii=False)

        m2 = re.search(r"^\s*([0-6])\s*$", text.strip())
        if m2:
            label = int(m2.group(1))
            return label, json.dumps({"PARSE": "FALLBACK_SINGLE_DIGIT"}, ensure_ascii=False)

        return None

    def get_prediction(self, content: str, max_retries: int = 3, max_api_retries: int = 5) -> Optional[Tuple[int, str]]:
        """
        Get prediction with retry logic for both API errors and parsing failures.
        
        Args:
            content: The text content to analyze
            max_retries: Maximum retries for parsing failures
            max_api_retries: Maximum retries for API errors (overload, rate limit, etc.)
        
        Returns:
            Tuple of (label, reasoning) or None if the model refuses/fails
        """
        last_text = ""
        last_error = None
        
        for _attempt in range(max_retries):
            enhanced_prompt = self._build_prompt(content)
            resp = None

            for api_attempt in range(max_api_retries):
                try:
                    resp = completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": enhanced_prompt},
                            {"role": "user", "content": content},
                        ],
                    )
                    break  
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    is_retryable = any(term in error_str for term in [
                        "overloaded", "overload_error", "rate_limit", "rate limit", "429", 
                        "503", "502", "500", "timeout", "connection", "temporarily unavailable",
                        "internal server error", "service unavailable"
                    ])
                    
                    if is_retryable and api_attempt < max_api_retries - 1:
                        wait_time = 2 ** (api_attempt + 1) + random.uniform(0, 1)
                        print(f"API error (attempt {api_attempt + 1}/{max_api_retries}): {type(e).__name__}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"Skipping sample due to API error: {type(e).__name__}: {str(e)[:100]}")
                        return None
            
            if resp is None:
                print(f"Skipping sample: Max API retries reached")
                return None

            last_text = (resp.choices[0].message.content or "").strip()
            
            parsed = self._try_parse_label_and_reasoning(last_text)
            if parsed is not None:
                return parsed[0], parsed[1]

        print(f"Skipping sample: Failed to parse after {max_retries} retries. Last output: {last_text[:100]}...")
        return None

    def process_dataframe(self, df: pd.DataFrame, delay: float = 0.0, max_retries: int = 3) -> pd.DataFrame:
        preds, reasons, skipped_indices = [], [], []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            if "content" not in row:
                raise ValueError("DataFrame must include a 'content' column.")
            
            result = self.get_prediction(str(row["content"]), max_retries=max_retries)
            
            if result is None:
                preds.append(-1)
                reasons.append("SKIPPED")
                skipped_indices.append(idx)
            else:
                label, r = result
                preds.append(label)
                reasons.append(r)
            
            if delay:
                time.sleep(delay)

        out = df.copy()
        out["predicted_label"] = preds
        out["reasoning"] = reasons
        
        if skipped_indices:
            print(f"\nWarning: {len(skipped_indices)} samples were skipped due to refusals/errors")
            print(f"Skipped indices: {skipped_indices[:20]}{'...' if len(skipped_indices) > 20 else ''}")
        
        return out

    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        filtered_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
        
        if not filtered_pairs:
            print("Warning: No valid predictions to evaluate!")
            return {
                "accuracy": 0.0,
                "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0,
                "precision_weighted": 0.0, "recall_weighted": 0.0, "f1_weighted": 0.0,
                "precision_micro": 0.0, "recall_micro": 0.0, "f1_micro": 0.0,
                "skipped_count": len(y_pred) - len(filtered_pairs),
                "evaluated_count": 0,
            }
        
        y_true_filtered = [t for t, _ in filtered_pairs]
        y_pred_filtered = [p for _, p in filtered_pairs]
        
        skipped_count = len(y_pred) - len(filtered_pairs)
        if skipped_count > 0:
            print(f"Note: {skipped_count} samples were skipped, evaluating on {len(filtered_pairs)} samples")
        
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)

        precision_macro = precision_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
        recall_macro = recall_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
        f1_macro = f1_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)

        precision_weighted = precision_score(y_true_filtered, y_pred_filtered, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_true_filtered, y_pred_filtered, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true_filtered, y_pred_filtered, average="weighted", zero_division=0)

        precision_micro = precision_score(y_true_filtered, y_pred_filtered, average="micro", zero_division=0)
        recall_micro = recall_score(y_true_filtered, y_pred_filtered, average="micro", zero_division=0)
        f1_micro = f1_score(y_true_filtered, y_pred_filtered, average="micro", zero_division=0)

        p_pc, r_pc, f_pc, s_pc = precision_recall_fscore_support(
            y_true_filtered, y_pred_filtered, average=None, zero_division=0
        )
        cls_report = classification_report(y_true_filtered, y_pred_filtered, zero_division=0, output_dict=True)
        conf = confusion_matrix(y_true_filtered, y_pred_filtered)

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro, "recall_macro": recall_macro, "f1_macro": f1_macro,
            "precision_weighted": precision_weighted, "recall_weighted": recall_weighted, "f1_weighted": f1_weighted,
            "precision_micro": precision_micro, "recall_micro": recall_micro, "f1_micro": f1_micro,
            "precision_per_class": p_pc, "recall_per_class": r_pc, "f1_per_class": f_pc, "support_per_class": s_pc,
            "classification_report": cls_report,
            "confusion_matrix": conf,
            "skipped_count": skipped_count,
            "evaluated_count": len(filtered_pairs),
        }

# Step error analysis

def compute_step_diff_distribution(df: pd.DataFrame) -> Tuple[Counter, List[Dict[str, Any]]]:
    dist = Counter()
    records = []

    for i, row in df.iterrows():
        gold_label = int(row.get("intention_label", 0))
        pred_label = int(row.get("predicted_label", 0))

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

    problematic = [
        (step, count) for step, count in step_diff_distribution.items()
        if (count / total) >= threshold
    ]
    problematic.sort(key=lambda x: x[1], reverse=True)
    problematic_steps = [step for step, _ in problematic]

    error_analysis = analyze_step_error_patterns(step_records)
    filtered = {s: a for s, a in error_analysis.items() if s in problematic_steps}
    return problematic_steps, filtered

# Evaluation

def evaluate_on_dataset(
    df: pd.DataFrame,
    analyzer: SocialMediaAnalyzerRAG,
    gold_reason_col: str = "reason",
    delay: float = 0.0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    results_df = analyzer.process_dataframe(df, delay=delay, max_retries=max_retries)

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


def export_results(
    result: Dict[str, Any],
    output_dir: str = "lingo_outputs",
    model_name: str = "model"
) -> str:
    """
    Export predictions and summary to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    clean_model_name = model_name.replace("/", "_").replace(":", "_")
    
    # Save test predictions CSV
    if result.get("final_test_results") is not None:
        predictions_path = os.path.join(output_dir, f"LinGO_RAG_{clean_model_name}_predictions.csv")
        result["final_test_results"].to_csv(predictions_path, index=False, encoding="utf-8")
        print(f"Saved test predictions to {predictions_path}")
    
    # Save history CSV
    if result.get("history"):
        history_path = os.path.join(output_dir, f"LinGO_RAG_{clean_model_name}_history.csv")
        history_rows = []
        for h in result["history"]:
            history_rows.append({
                "round": h["round"],
                "target_steps": ", ".join(h["target_steps"]) if h["target_steps"] else "None",
                "val_accuracy": h["val_accuracy"],
                "val_f1_macro": h["val_f1_macro"],
                "val_f1_weighted": h["val_f1_weighted"],
                "val_f1_micro": h["val_f1_micro"],
                "val_step_diff_rate": h["val_step_diff_rate"],
            })
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        print(f"Saved history to {history_path}")
    
    # Save summary report
    summary_path = os.path.join(output_dir, f"LinGO_RAG_{clean_model_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"LinGO RAG Results - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATA SPLIT:\n")
        f.write(f"  TRAIN: {result.get('train_size', 'N/A')} samples\n")
        f.write(f"  VAL: {result.get('val_size', 'N/A')} samples\n")
        f.write(f"  TEST: {result.get('test_size', 'N/A')} samples\n\n")
        
        f.write(f"Best Round: {result.get('best_round', 'N/A')}\n")
        f.write(f"Best Target Steps: {', '.join(result.get('best_target_steps', [])) or 'None'}\n")
        f.write(f"Best VAL F1 (weighted): {result.get('best_val_f1', 0):.4f}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("FINAL TEST RESULTS:\n")
        f.write("=" * 60 + "\n")
        test_metrics = result.get("final_test_metrics", {})
        f.write(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"  F1 (macro): {test_metrics.get('f1_macro', 0):.4f}\n")
        f.write(f"  F1 (weighted): {test_metrics.get('f1_weighted', 0):.4f}\n")
        f.write(f"  F1 (micro): {test_metrics.get('f1_micro', 0):.4f}\n")
        f.write(f"  Step Diff Distribution: {result.get('final_test_step_diff_distribution', {})}\n")
    
    print(f"Saved summary to {summary_path}")
    
    return output_dir

# Main interative loop 

def iterative_rag_refinement(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: str = "gemini/gemini-2.5-flash-lite",
    rounds: int = 3,
    delay: float = 0.0,
    seed: int = 42,
    step_threshold: float = 0.1,
    val_ratio: float = 0.2,
    num_examples_per_source: int = 5,
    retrieval_candidates: int = 20,
) -> Dict[str, Any]:
    random.seed(seed)

    print("="*60)
    print("STEP 1: Split dev into TRAIN and VAL")
    print("="*60)

    train_df, val_df = split_train_val(dev_df, val_ratio=val_ratio, seed=seed)
    print(f"DEV: {len(dev_df)} | TRAIN: {len(train_df)} | VAL: {len(val_df)} | TEST: {len(test_df)}")

    print("\n" + "="*60)
    print("STEP 2: Build FULL RAG store from TRAIN only")
    print("="*60)

    rag_store = RAGExampleStore()
    rag_store.index_training_data(train_df)

    print("\n" + "="*60)
    print("STEP 3: Iterative refinement")
    print(f"       Retrieval: {num_examples_per_source} from targeted steps + {num_examples_per_source} from error patterns = {num_examples_per_source * 2} total")
    print("="*60)

    history: List[Dict[str, Any]] = []

    cumulative_target_steps: Set[str] = set()
    cumulative_error_patterns: Dict[str, Counter] = {}

    best_val_f1 = -1.0
    best_round = 0
    best_target_steps: Set[str] = set()
    best_error_analysis: Dict[str, Dict[str, Any]] = {}

    last_val_eval: Optional[Dict[str, Any]] = None

    def _build_cumulative_error_analysis() -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for step, patt_counter in cumulative_error_patterns.items():
            out[step] = {
                "most_common_errors": patt_counter.most_common(5),
                "error_patterns": dict(patt_counter),
                "total_errors": int(sum(patt_counter.values())),
            }
        return out

    for r in trange(rounds + 1, desc="LinGO Rounds"):
        print(f"\n{'='*60}")
        print(f"Round {r}")
        print(f"  cumulative_target_steps = {sorted(list(cumulative_target_steps)) if cumulative_target_steps else 'None'}")
        print("="*60)

        cumulative_error_analysis = _build_cumulative_error_analysis()

        analyzer = SocialMediaAnalyzerRAG(
            model=model,
            rag_store=rag_store,
            target_steps=sorted(list(cumulative_target_steps)),
            error_analysis=cumulative_error_analysis,
            num_examples_per_source=num_examples_per_source,
            retrieval_candidates=retrieval_candidates,
        )

        print(f"Evaluating on VAL (retrieval: {num_examples_per_source} targeted + {num_examples_per_source} error patterns)...")
        val_eval = evaluate_on_dataset(val_df, analyzer, delay=delay, max_retries=3)
        last_val_eval = val_eval

        f1w = val_eval["metrics"]["f1_weighted"]
        acc = val_eval["metrics"]["accuracy"]

        history.append({
            "round": r,
            "target_steps": sorted(list(cumulative_target_steps)),
            "val_accuracy": acc,
            "val_f1_macro": val_eval["metrics"]["f1_macro"],
            "val_f1_weighted": f1w,
            "val_f1_micro": val_eval["metrics"]["f1_micro"],
            "val_step_diff_distribution": val_eval["step_diff_distribution"],
            "val_step_diff_rate": val_eval["step_diff_rate"],
        })

        print(f"VAL - Acc={acc:.4f}, F1w={f1w:.4f}")
        print(f"VAL step diff distribution: {val_eval['step_diff_distribution']}")

        if f1w > best_val_f1:
            best_val_f1 = f1w
            best_round = r
            best_target_steps = set(cumulative_target_steps)
            best_error_analysis = dict(cumulative_error_analysis)

        if r == rounds:
            break

        val_dist_counter = Counter(val_eval["step_diff_distribution"])
        if sum(val_dist_counter.values()) == 0:
            print("No step differences found. Early stop.")
            break

        new_target_steps, new_error_analysis = identify_problematic_steps_with_patterns(
            val_dist_counter, val_eval["step_records"], threshold=step_threshold
        )

        cumulative_target_steps.update(new_target_steps)

        for step, analysis in new_error_analysis.items():
            patt = Counter(dict(analysis.get("error_patterns", {})))
            cumulative_error_patterns.setdefault(step, Counter())
            cumulative_error_patterns[step].update(patt)

        print(f"Next-round new targeted steps: {new_target_steps}")
        for step, analysis in new_error_analysis.items():
            print(f"  {step} top patterns: {analysis.get('most_common_errors', [])[:3]}")

    print("\n" + "="*60)
    print(f"STEP 4: Final evaluation on TEST using best configuration from round {best_round}")
    print(f"Best target steps: {sorted(list(best_target_steps)) if best_target_steps else 'None'}")
    print("="*60)

    final_analyzer = SocialMediaAnalyzerRAG(
        model=model,
        rag_store=rag_store,
        target_steps=sorted(list(best_target_steps)),
        error_analysis=best_error_analysis,
        num_examples_per_source=num_examples_per_source,
        retrieval_candidates=retrieval_candidates,
    )

    test_eval = evaluate_on_dataset(test_df, final_analyzer, delay=delay, max_retries=3)

    print("\nFINAL TEST RESULTS")
    print(f"Accuracy: {test_eval['metrics']['accuracy']:.4f}")
    print(f"F1 weighted: {test_eval['metrics']['f1_weighted']:.4f}")

    return {
        "history": history,
        "best_round": best_round,
        "best_target_steps": sorted(list(best_target_steps)),
        "best_error_analysis": best_error_analysis,
        "best_val_f1": best_val_f1,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "final_val_metrics": (last_val_eval["metrics"] if last_val_eval else None),
        "final_test_metrics": test_eval["metrics"],
        "final_test_results": test_eval["results_df"],  # Added: test predictions DataFrame
        "final_test_step_diff_distribution": test_eval["step_diff_distribution"],
    }

# Final run

def run_rag_optimization(
    dev_filepath: str,
    test_filepath: str,
    model: str = "gemini/gemini-2.5-flash-lite",
    rounds: int = 3,
    delay: float = 0.0,
    step_threshold: float = 0.1,
    val_ratio: float = 0.2,
    num_examples_per_source: int = 5,
    retrieval_candidates: int = 20,
    seed: int = 42,
    output_dir: str = "lingo_outputs",
) -> Dict[str, Any]:
    dev_df = pd.read_csv(dev_filepath)
    test_df = pd.read_csv(test_filepath)

    result = iterative_rag_refinement(
        dev_df=dev_df,
        test_df=test_df,
        model=model,
        rounds=rounds,
        delay=delay,
        seed=seed,
        step_threshold=step_threshold,
        val_ratio=val_ratio,
        num_examples_per_source=num_examples_per_source,
        retrieval_candidates=retrieval_candidates,
    )
    
    export_results(result, output_dir=output_dir, model_name=model)
    
    return result


if __name__ == "__main__":
    result = run_rag_optimization(
        dev_filepath="./incivi_dev.csv",
        test_filepath="./incivi_test.csv",
        model="claude-3-haiku-20240307", # or "gemini/gemini-2.0-flash-lite", "gpt-5-mini"
        rounds=5,
        delay=1.0,
        step_threshold=0.1,
        val_ratio=0.2,
        num_examples_per_source=5,  # 5 from targeted steps + 5 from error patterns = 10 total
        retrieval_candidates=20,
        seed=42,
        output_dir="RAG_LinGO_Claude_outputs_0201",  
    )

## Results: Claude (Test Accuracy: 0.4787; Test F1 weighted: 0.4887), Gemini ( Test Accuracy: 0.6900; Test F1 weighted: 0.6992); GPT (Accuracy: 0.6550; F1 (weighted): 0.6765)