

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

sentences = [
    "什么是多智能体系统？",
    "LangChain Agent 是如何调用工具的？"
]

embeddings = model.encode(sentences, normalize_embeddings=True)

print(embeddings.shape)
print(embeddings[0][:10])

