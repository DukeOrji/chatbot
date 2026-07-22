from sentence_transformers import SentenceTransformer, CrossEncoder


"""
Models:
qwen3:4b
qwen2.5:1.5b 

Sentence Transformers:
nomic-embed-text-v1.5
BAAI/bge-large-en-v1.5
Snowflake/snowflake-arctic-embed-l-v2.0

reranker:
bge-reranker-v2-m3

"""


model = "qwen2.5:1.5b"

embedder = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5"
)


reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")