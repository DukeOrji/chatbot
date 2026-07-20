#emb.py

from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np
import torch
import faiss


embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

MEMORY_PATH = "./memory"
embeddings = []

def load_memory():
    # ----------------------------
    # Load Memory
    # ----------------------------

    memory_store = []
    

    for filename in os.listdir(MEMORY_PATH):

        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(MEMORY_PATH, filename)

        with open(file_path, "r", encoding="utf-8") as file:

            data = json.load(file)

            conversation = []

            for message in data["messages"]:
                conversation.append(
                    f'{message["speaker"]}: {message["text"]}'
                )

            conversation = "\n".join(conversation)

            memory_store.append(
                {
                    "conversation": conversation,
                    "participant": data.get("participant", "unknown"),
                    "intent": data.get("intent", "unknown"),
                    "timestamp": data.get("timestamp", "unknown"),
                }
            )


    # ----------------------------
    # Build Embeddings
    # ----------------------------

    with torch.no_grad():

        for memory in memory_store:

            embedding = embedder.encode(
                memory["conversation"],
                convert_to_numpy=True
            )

            embeddings.append(embedding)

    return memory_store

    
def build_index(embeddings):
    embedding_matrix = np.vstack(embeddings).astype("float32")


    # ----------------------------
    # Build FAISS
    # ----------------------------

    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embedding_matrix)
    return index
