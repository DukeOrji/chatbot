

def prompt(context, query):
    prompt = f"""
    Instructions:
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

    {query}

    Your Reply:
    """

    response = generator(
        prompt,
        max_new_tokens=80,
        temperature=0.7,
        do_sample=True,
    )

    return response