import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

print("BASE_URL =", os.getenv("OPENAI_BASE_URL"))
print("MODEL_NAME =", os.getenv("MODEL_NAME"))
print("KEY_PREFIX =", os.getenv("OPENAI_API_KEY")[:8])

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

resp = llm.invoke("你好，请只回复：连接成功")
print(resp.content)

# import os
# from dotenv import load_dotenv
# from openai import OpenAI
#
# load_dotenv(override=True)
#
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_BASE_URL"),
# )
#
# print("BASE_URL =", os.getenv("OPENAI_BASE_URL"))
# print("KEY_PREFIX =", os.getenv("OPENAI_API_KEY")[:8])
#
# models = client.models.list()
#
# for m in models.data:
#     print(m.id)