#generate.py

from transformers import pipeline
import torch
from emb import build_index, load_memory embedder


generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

memory_store = load_memory()
index = build_index()



def search(query):

    with torch.no_grad():

        query_vector = embedder.encode(
            query,
            convert_to_numpy=True
        )

    query_vector = query_vector.reshape(1, -1).astype("float32")

    k = min(5, len(memory_store))

    distances, indices = index.search(
        query_vector,
        k
    )

    retrieved_memories = []

    for idx in indices[0]:

        if idx == -1:
            continue

        retrieved_memories.append(
            memory_store[idx]
        )

    context = "\n\n".join(
        memory["conversation"]
        for memory in retrieved_memories
    )

    

    response = prompt(context, query)

    generated = response[0]["generated_text"]

    reply = generated[len(prompt):].strip()

    # Store the interaction
    memory_store.append(
        {
            "conversation": f"User: {query}\nBot: {reply}",
            "participant": "unknown",
            "intent": "generated",
            "timestamp": None,
        }
    )

    for conv in memory_store["conversation"]:
        embedding = embedder.encode(
            conv,
            convert_to_numpy=True
        ).astype("float32")

        index.add(embedding.reshape(1, -1))

    save_memory(memory_store)

    return {
        "message": query,
        "reply": reply,
        "retrieved": len(retrieved_memories),
    }