import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ------------------ 1. Load Preprocessed Data ------------------ #

with open("D:/Multi-Agents/rag_structured_chunks.json", "r", encoding="utf-8") as f:
    structured_chunks = json.load(f)

print(f"✅ Loaded {len(structured_chunks)} chunks")

# ------------------ 2. Initialize Pinecone ------------------ #

api_key = "pcsk_39dCQD_E2BgPziuKwtAGMBHnVmG6bJYJsgqi1WB6ajxTAefLSzFAHh8e8s2qgLitnsHt5K"
index_name = "rag-minimilist-data"
dimension = 384  # This must match your embedding model's output size

pc = Pinecone(api_key=api_key)

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created index: {index_name}")
else:
    print(f"⚡ Index already exists: {index_name}")

index = pc.Index(index_name)

# ------------------ 3. Embed and Upload ------------------ #

model = SentenceTransformer("all-MiniLM-L6-v2")  # Output dim = 768

batch_size = 100
for i in tqdm(range(0, len(structured_chunks), batch_size), desc="🔁 Uploading to Pinecone"):
    batch = structured_chunks[i:i + batch_size]

    ids = [f"{chunk['product_id']}_{i+j}" for j, chunk in enumerate(batch)]
    texts = [chunk["content"] for chunk in batch]
    embeddings = model.encode(texts).tolist()
    metas = [
        {
            "product_id": chunk["product_id"],
            "title": chunk["title"],
            "section": chunk["section"]
        }
        for chunk in batch
    ]

    to_upsert = zip(ids, embeddings, metas)
    index.upsert(vectors=to_upsert)

print("✅ All chunks embedded and uploaded to Pinecone!")

# ------------------ 4. Sample Query ------------------ #

query = "Which cream contains resveratrol and retinal?"
query_embedding = model.encode([query])[0].tolist()

results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

print("\n🔍 Top Matches:")
for match in results["matches"]:
    print(f"\nScore: {match['score']}")
    print(f"Title: {match['metadata']['title']}")
    print(f"Section: {match['metadata']['section']}")
