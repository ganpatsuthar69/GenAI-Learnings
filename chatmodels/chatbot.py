from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=512)

chat_model = ChatHuggingFace(llm=llm)
print("chosse your AI mode")
print("press 1 for Angry mode")
print("press 2 for funny mode ")
print("press 3 for sad mode")

choice = int(input("tell your response :- "))

if choice == 1:
    mode = "You are an angry AI agent. You respond aggressively and impatiently."
elif choice == 2:
    mode = "You are a very funny AI agent. You respond with humor and jokes."
elif choice == 3:
    mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."


#massages list banai hai taaki chat history isme store ho ske but it still not that good because memory storage is too high for large chats 
messages = [
       SystemMessage(content= mode)
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