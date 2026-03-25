from dotenv import load_dotenv
import os 

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY"))
# # print(model)

# response = model.invoke("What is Cricket")
# # print(response)
# print(response.content) 

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Who are you?")
print(response.content)
