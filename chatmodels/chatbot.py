from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=512)

chat_model = ChatHuggingFace(llm=llm)

#massages list banai hai taaki chat history isme store ho ske but it still not that good because memory storage is too high for large chats 
messages = [
       SystemMessage(content= "You are funny AI Agent")
]

print("********* Welcome to My Chatbot *********")
print("---- Enter 0 if you want to Exit ----")

while True:
    prompt = input("You: ")
    
    if prompt == "0":
        print("Bot: Bye, See You Again")
        break
    messages.append(HumanMessage(content=prompt))
    response = chat_model.invoke(messages)
    messages.append(AIMessage(response.content))
    print("Bot:", response.content)
print(messages)