#generate.py

from models import embedder, reranker
from prompt import prompt
import torch
from emb import build_index, load_memory, add_memory
from datetime import datetime, timezone


memory_store, embeddings = load_memory()
index = build_index(embeddings)



def search(query):

    with torch.no_grad():

        query_vector = embedder.encode(
            query,
            convert_to_numpy=True
        )

    query_vector = query_vector.reshape(1, -1).astype("float32")

    k = min(10, len(memory_store))

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

    #reranking:  compare query against each retrieved conversation
    #assign distance scores then pick a smaller context window
    pairs = []
    for memory in retrieved_memories:
        conversation_text = "\n".join(
            f'{m["speaker"]}: {m["text"]}'
            for m in memory["conversation"]
        )

        pairs.append(
            (query, conversation_text)
        )

    if retrieved_memories:
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(scores, retrieved_memories),
            key=lambda x: x[0],
            reverse=True
        )
        top_memories = [
            memory
            for _, memory in ranked[:3] #keep best three
        ]
    
    else:
        top_memories = []


    context = []
    for memory in top_memories:
        conversation_text = "\n".join(
            f'{msg["speaker"]}: {msg["text"]}'
            for msg in memory["conversation"]
        )
        context.append(conversation_text)

    context = "\n\n".join(context)

    response = prompt(context, query)
    reply = response.message.content

    new_memory = {
        "conversation": [{
            "speaker": "user",

            "text": query
        },
        {
            "speaker": "bot",
            "text": reply
        }],

        "participants": ["unknown"],

        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    add_memory(index, memory_store, new_memory)


    return {
        "message": query,
        "reply": reply,
        "retrieved": len(retrieved_memories),
    }