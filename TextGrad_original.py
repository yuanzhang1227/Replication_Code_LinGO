import os
import re
import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import textgrad as tg


## Load the datasets

def load_csv(path: str):
    df = pd.read_csv(path)
    df["intention_label"] = df["intention_label"].astype(int)
    return list(zip(df["content"].astype(str).tolist(), df["intention_label"].tolist()))

train_data = load_csv("incivi_dev.csv")
test_data  = load_csv("incivi_test.csv")


## Define helper function

def parse_digit_0_6(text: str) -> str:
    """Extract a single label in {0..6}. Fallback to '0'."""
    if text is None:
        return "0"
    m = re.search(r"\b([0-6])\b", str(text).strip())
    return m.group(1) if m else "0"


## Define initial prompt

INITIAL_SYSTEM_PROMPT = """You are an annotator interpreting the intent of uncivil Portuguese social media discourse about Brazilian politics on X (formerly Twitter).
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
0 = Other.
Output: ONLY one integer from [0-6]. No other text.
"""

## Optimization engines (forward model + backward/feedback model)

os.environ["GEMINI_API_KEY"] = "YOUR API KEY" 
os.environ["OPENAI_API_KEY"] = "YOUR API KEY"

forward_engine = tg.get_engine("experimental:gemini/gemini-2.5-flash-lite")   #experimental:gpt-5-mini, experimental:anthropic/claude-3-haiku-20240307

tg.set_backward_engine("experimental:gpt-5-mini", override=True)

system_prompt = tg.Variable(
    INITIAL_SYSTEM_PROMPT,
    requires_grad=True,
    role_description="system prompt to guide the LLM to output the correct intent label (0-6)"
)

model = tg.BlackboxLLM(forward_engine, system_prompt=system_prompt)

optimizer = tg.TGD(parameters=list(model.parameters())) 

## Loss: evaluator that returns 0/1 only

LOSS_INSTRUCTION = """You are a strict evaluator.

You will be given:
- POST (the raw post text; the FIRST [bracket tag] indicates the target category)
- GOLD (an integer 0-6)
- PRED (model output text)

Return ONLY:
0 if the first integer label in PRED equals GOLD exactly;
1 otherwise.

No other text.
"""
loss_fn = tg.TextLoss(LOSS_INSTRUCTION)  


def make_question(post: str) -> tg.Variable:
    return tg.Variable(
        f"POST:\n{post}\n\nFINAL ANSWER (0-6):",
        role_description="input post",
        requires_grad=False,
    )


def make_eval_input(post: str, gold: int, pred_text: str) -> tg.Variable:
    pred = parse_digit_0_6(pred_text)
    return tg.Variable(
        f"POST:\n{post}\n\nGOLD: {gold}\nPRED: {pred}",
        role_description="evaluation pack",
        requires_grad=False,
    )

## Training loop 
random.seed(42)

EPOCHS = 5
BATCH_SIZE = 6
MAX_BATCHES_PER_EPOCH = 60  

for epoch in range(EPOCHS):
    random.shuffle(train_data)
    batches = [train_data[i:i + BATCH_SIZE] for i in range(0, len(train_data), BATCH_SIZE)]
    batches = batches[: min(len(batches), MAX_BATCHES_PER_EPOCH)]

    batch_errs = []
    for batch in batches:
        losses = []
        errs = 0

        for post, gold in batch:
            q = make_question(post)
            pred = model(q)  
            pred_text = getattr(pred, "value", str(pred))

            eval_pack = make_eval_input(post, gold, pred_text)
            loss = loss_fn(eval_pack)  
            losses.append(loss)

            errs += 1 if str(getattr(loss, "value", loss)).strip().startswith("1") else 0

        for loss in losses:
            loss.backward()

        optimizer.step()
        batch_errs.append(errs / max(1, len(batch)))

    print(f"[epoch {epoch+1}/{EPOCHS}] mean batch error: {sum(batch_errs)/len(batch_errs):.3f}")

print("\n--- Optimized system prompt ---\n")
print(system_prompt.value)

## Evaluate on test set
y_true, y_pred = [], []
for post, gold in test_data:
    q = make_question(post)
    pred = model(q)
    pred_text = getattr(pred, "value", str(pred))
    y_true.append(int(gold))
    y_pred.append(int(parse_digit_0_6(pred_text)))

acc = accuracy_score(y_true, y_pred)
f1w = f1_score(y_true, y_pred, average="weighted")
print(f"\nTest Accuracy:    {acc:.4f}")
print(f"Test Weighted F1: {f1w:.4f}")
