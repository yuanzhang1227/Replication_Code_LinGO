import os
import re
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.metrics import accuracy_score, f1_score

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["ANTHROPIC_API_KEY"] = "YOUR API KEY" 
os.environ["OPENAI_API_KEY"] = "YOUR API KEY"
os.environ["LANGSMITH_API_KEY"] = "YOUR API KEY"  


## Provide definitions of incivility categories and output labels

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

def extract_category(content: str) -> str:
    """
    Extract the category from the first bracket tag in the content.
    
    Example:
        "[Impoliteness] Some text..." -> "Impoliteness"
    """
    match = re.match(r'\[([^\]]+)\]', content.strip())
    if match:
        return match.group(1)
    return "Unknown"


def extract_text(content: str) -> str:
    """
    Extract the actual post text (without the category tag).
    
    Example:
        "[Impoliteness] Some text..." -> "Some text..."
    """
    text = re.sub(r'^\[[^\]]+\]\s*', '', content.strip())
    return text


def load_and_process_data(filepath: str) -> pd.DataFrame:
    """
    Load data and extract category and text from content.
    """
    df = pd.read_csv(filepath)
    df['category'] = df['content'].apply(extract_category)
    df['text'] = df['content'].apply(extract_text)
    return df


def create_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert DataFrame to LangChain Documents for indexing.
    
    Each document stores:
    - page_content: The post text (for embedding)
    - metadata: category, label, original_content, formatted example
    """
    documents = []
    
    for idx, row in df.iterrows():
        doc = Document(
            page_content=row['text'],  
            metadata={
                "category": row['category'],
                "label": int(row['intention_label']),
                "original_content": row['content'],
                "formatted": f"Post: \"{row['text']}\"\nCategory: [{row['category']}]\nLabel: {row['intention_label']}"
            }
        )
        documents.append(doc)
    
    return documents

## Initializing langchain components and indexing data

print("Initializing LangChain components...")

# 1. Chat model
# GPT
model = init_chat_model("gpt-5-mini")

# # Gemini
# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",
#     google_api_key="YOUR API KEY"  
# )

# # Claude
# model = ChatAnthropic(model="claude-3-haiku-20240307")

# 2. Embeddings model (multilingual)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Vector store
vector_store = InMemoryVectorStore(embeddings)

# 4. Indexing data
def initialize_vector_store(train_filepath: str):
    """Load training data and add to vector store."""
    global train_df
    
    train_df = load_and_process_data(train_filepath)
    
    documents = create_documents(train_df)
    
    document_ids = vector_store.add_documents(documents=documents)
    print(f"Indexed {len(document_ids)} documents in vector store")
    
    return train_df


## Define example retrieval function

def retrieve_similar_examples(query: str, category: str):
    """
    Retrieve similar training examples for classification.
    """
    retrieved_docs = vector_store.similarity_search(query, k=20)
    
    category_docs = [
        doc for doc in retrieved_docs 
        if doc.metadata.get("category") == category
    ]
    
    if not category_docs:
        category_docs = retrieved_docs
    
    seen_labels = set()
    diverse_docs = []
    remaining_docs = []
    
    for doc in category_docs:
        label = doc.metadata.get("label")
        if label not in seen_labels:
            diverse_docs.append(doc)
            seen_labels.add(label)
        else:
            remaining_docs.append(doc)
    
    final_docs = (diverse_docs + remaining_docs)[:10]

    serialized = "\n\n".join(
        doc.metadata.get("formatted", doc.page_content)
        for doc in final_docs
    )
    
    return serialized, final_docs


## Define classification prompt and function

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert annotator interpreting the intent of uncivil Portuguese social media discourse about Brazilian politics on X (formerly Twitter).

## Target Category: [{category}]
{category_definition}

## Intent Labels (apply ONLY to the target category):
1 = Explicit {category}: direct, overt {category}.
2 = Implicit {category}: indirect, veiled {category}.
3 = Report {category}: quotes/refers to {category} content without opinion.
4 = Intensify {category}: quotes/refers to {category} content and agrees/amplifies.
5 = Counter {category}: quotes/refers to {category} content and criticizes/disagrees.
6 = Escalate {category}: responds to {category} content with {category}.
0 = Other: Does not fit any of the above patterns.

## Similar Training Examples:
{examples}

Output: ONLY one integer from [0-6]. No other text."""),
    
    ("human", """Classify this post:

Post: "{post}"
Category: [{category}]

Your classification (single integer 0-6):""")
])

classification_chain = classification_prompt | model | StrOutputParser()

def classify_post(content: str) -> int:
    category = extract_category(content)
    text = extract_text(content)

    category_def = CATEGORY_DEFINITIONS.get(category, f"Content classified under {category}.")

    examples, _ = retrieve_similar_examples(query=text, category=category)

    result = classification_chain.invoke({
        "category": category,
        "category_definition": category_def,
        "examples": examples,
        "post": text
    })

    m = re.search(r"^[0-6]$", result.strip())  
    return int(m.group()) if m else 0
    
def classify_batch(test_df: pd.DataFrame, verbose: bool = True) -> list[int]:
    """
    Classify all posts in a test DataFrame.
    """
    predictions = []
    
    for idx, row in test_df.iterrows():
        try:
            pred = classify_post(row['content'])
            predictions.append(pred)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            predictions.append(0)
        
        if verbose and (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(test_df)}")
    
    return predictions


def evaluate(test_df: pd.DataFrame, predictions: list[int]) -> dict:
    """
    Evaluate classification performance.
    """
    
    true_labels = test_df['intention_label'].tolist()
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    weighted_f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1
    }

## The main function call

def main():

    train_df = initialize_vector_store("incivi_dev.csv")
    test_df = load_and_process_data("incivi_test.csv")
        
    predictions = classify_batch(test_df, verbose=True)

    test_df['predicted_label'] = predictions
    test_df.to_csv("RAG_original_predictions.csv", index=False)
    print(f"\nPredictions saved.")
        
    results = evaluate(test_df, predictions)

if __name__ == "__main__":
    main()
