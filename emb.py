#emb.py

from models import embedder
import os
import json
import numpy as np
import torch
import faiss




MEMORY_PATH = "./memory"
filename = "memory.json"



def add_memory(index, memory_store, new_memory):

    memory_store.append(new_memory)

    conversation_text = "\n".join(
        f'{m["speaker"]}: {m["text"]}'  #One embedding per conversation.
        for m in new_memory["conversation"]
    )

    embedding = embedder.encode(
        conversation_text,
        convert_to_numpy=True
    ).astype("float32")


    index.add(
        embedding.reshape(1, -1)
    )


    with open(os.path.join(MEMORY_PATH, filename), "w", encoding="utf-8") as file:
        json.dump(memory_store, file, ensure_ascii=False, indent=4)



def load_memory():
    # ----------------------------
    # Load Memory
    # ----------------------------

    memory_store = []
    embeddings = []

    for filename in os.listdir(MEMORY_PATH):

        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(MEMORY_PATH, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            for chat in data:

                conversation = []

                for message in chat["conversation"]:
                    conversation.append(
                        f'{message["speaker"]}: {message["text"]}'
                    )

                memory_store.append(
                    {
                        "conversation": chat["conversation"],
                        "participant": chat.get("participants"),
                        "timestamp": chat.get("timestamp"),
                    }
                )

            


    # ----------------------------
    # Build Embeddings
    # ----------------------------

    with torch.no_grad():
        for memory in memory_store:

            conversation_text = "\n".join(
                f'{m["speaker"]}: {m["text"]}'   #One embedding per conversation.
                for m in memory["conversation"]
            )

            embedding = embedder.encode(
                conversation_text,
                convert_to_numpy=True
            )

            embeddings.append(embedding)

        return memory_store, embeddings

    
def build_index(embeddings):
    embedding_matrix = np.vstack(embeddings).astype("float32")


    # ----------------------------
    # Build FAISS
    # ----------------------------

    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embedding_matrix)
    return index
