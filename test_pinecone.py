import os
from dotenv import load_dotenv
import pinecone
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ------------------ Pinecone Setup ------------------ #
from pinecone import Pinecone

# Initialize Pinecone and connect to the index
index_name = "medigraphai"  # Ensure this is the correct index name
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# ------------------ Embedding Function ------------------ #
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> list:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return pooled.squeeze().tolist()

# ------------------ Pinecone Retriever ------------------ #
from langchain_pinecone import PineconeVectorStore

class CustomHFEmbedding(Embeddings):
    def embed_query(self, text: str) -> list:
        return get_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [get_embedding(t) for t in texts]

# Connect to the Pinecone VectorStore
vectorstore = PineconeVectorStore(index=index, embedding=CustomHFEmbedding())

# ------------------ Query Function ------------------ #
def pinecone_retriever(query: str) -> str:
    # Generate the query vector embedding
    query_vector = get_embedding(query)

    # Retrieve the top 3 most similar documents from Pinecone
    docs = vectorstore.similarity_search(query, k=3)
    
    # Format the response
    if not docs:
        return "No relevant documents found in Pinecone."
    
    combined_text = ""
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        combined_text += f"[Result {i} from Pinecone]\nText: {doc.page_content}\nSource: {source}\n\n"
    
    return combined_text

# ------------------ Testing Pinecone Retrieval ------------------ #
def test_pinecone(query: str):
    # Get the result from Pinecone Retriever
    result = pinecone_retriever(query)
    print("\nPinecone Retrieval Result:\n")
    print(result)

# ------------------ Main ------------------ #
if __name__ == "__main__":
    # Test the Pinecone retrieval
    user_query = input("Enter the query to search in Pinecone: ")
    test_pinecone(user_query)
