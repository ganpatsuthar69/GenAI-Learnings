from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=512
)

chat_model = ChatHuggingFace(llm=llm)

print("********* Welcome to My Chatbot *********")
print("---- Enter 0 if you want to Exit ----")

while True:
    prompt = input("You: ")

    if prompt == "0":
        print("Bot: Bye, See You Again")
        break

    response = chat_model.invoke(prompt)
    print("Bot:", response.content)