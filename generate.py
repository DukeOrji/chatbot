from transformers import pipeline
import torch
from emb import index, memory_store, embedder


generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

# ----------------------------
# Search
# ----------------------------

def search(query: Query):

    with torch.no_grad():

        query_vector = embedder.encode(
            query.msg,
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

    prompt = f"""
    You are imitating my texting style.

    Rules:
    - Never copy messages word for word.
    - Learn my tone from retrieved conversations.
    - Prefer my slang naturally.
    - Keep replies concise.
    - Match my pacing.
    - Sound human.
    - Improve weak replies while preserving my personality.

    Retrieved Conversations:

    {context}

    Current Message:

    {query.msg}

    Your Reply:
    """

    response = generator(
        prompt,
        max_new_tokens=80,
        temperature=0.7,
        do_sample=True,
    )

    generated = response[0]["generated_text"]

    reply = generated[len(prompt):].strip()

    # Store the interaction
    memory_store.append(
        {
            "conversation": f"User: {query.msg}\nBot: {reply}",
            "participant": "unknown",
            "intent": "generated",
            "timestamp": None,
        }
    )

    return {
        "message": query.msg,
        "reply": reply,
        "retrieved": len(retrieved_memories),
    }