import pandas as pd
import dspy
from typing import Literal
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import random

## Define signature, prompt structure, and evaluation metric

LABEL = Literal[0, 1, 2, 3, 4, 5, 6]

class IntentLabel(dspy.Signature):
    """You are an annotator interpreting the intent of uncivil Portuguese social media discourse about Brazilian politics on X (formerly Twitter).
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
0 = Does not fit any of the above patterns.
Output: ONLY one integer from [0-6]. No other text.
    """
    post: str = dspy.InputField()
    label: LABEL = dspy.OutputField()

def acc_metric(pred, gold, trace=None):
    try:
        return int(pred.label) == int(gold.label)
    except Exception:
        return False

## Load datasets and convert to the Dpsy format

dev_df  = pd.read_csv("incivi_dev.csv")
test_df = pd.read_csv("incivi_test.csv")

dev_df["intention_label"]  = dev_df["intention_label"].astype(int)
test_df["intention_label"] = test_df["intention_label"].astype(int)

def df_to_examples(df):
    return [
        dspy.Example(post=str(r["content"]), label=int(r["intention_label"])).with_inputs("post")
        for _, r in df.iterrows()
    ]

train_df, val_df = train_test_split(
    dev_df,
    test_size=0.2,                
    random_state=42,              
    stratify=dev_df["intention_label"]  
)

trainset = df_to_examples(train_df)
valset   = df_to_examples(val_df)
testset  = df_to_examples(test_df)

## Choose large language model and do optimization

lm = dspy.LM("anthropic/claude-3-haiku-20240307", api_key="YOUR API KEY") #gemini/gemini-2.5-flash-lite; openai/gpt-5-mini
dspy.configure(lm=lm)

base = dspy.Predict(IntentLabel)

optimizer = BootstrapFewShotWithRandomSearch(
    metric=acc_metric,
    max_labeled_demos=30,
    max_bootstrapped_demos=20,
    num_candidate_programs=10,
    max_rounds=1,
    num_threads=1
)

optimized = optimizer.compile(base, trainset=trainset, valset=valset)

## Predict on test set and compute Accuracy and weighted F1

y_true, y_pred = [], []

for ex in testset:
    pred = optimized(post=ex.post)
    try:
        pred_label = int(str(pred.label).strip())
    except Exception:
        pred_label = 0

    y_true.append(int(ex.label))
    y_pred.append(pred_label)

acc = accuracy_score(y_true, y_pred)
f1w = f1_score(y_true, y_pred, average="weighted")

print(f"Test Accuracy:     {acc:.4f}")
print(f"Test Weighted F1:  {f1w:.4f}")

## Results: Claude (Test Accuracy: 0.5150; Test Weighted F1:  0.5398); Gemini (Test Accuracy: 0.6300; Test Weighted F1:  0.6060); GPT (Test Accuracy: 0.5400; Test Weighted F1:  0.5247)
