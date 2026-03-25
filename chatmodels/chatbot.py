from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    max_new_tokens=512)

chat_model = ChatHuggingFace(llm=llm)

print("Choose your AI mode")
print("1 for Angry")
print("2 for Funny")
print("3 for Sad")

choice = input("Enter your choice: ")

if choice == "1":
    mode = "You are an angry AI agent. You respond aggressively and impatiently."
elif choice == "2":
    mode = "You are a very funny AI agent. You respond with humor and jokes."
elif choice == "3":
    mode = "You are a very sad AI agent. You respond in a depressed and emotional tone."
else:
    mode = "You are a helpful AI assistant."

# Memory setup
messages = [SystemMessage(content=mode)]
MAX_HISTORY = 10

print("\n********* Welcome to My Chatbot *********")
print("---- Enter 0 to Exit ----")

while True:
    prompt = input("You: ")

    if prompt == "0":
        print("Bot: Bye, See You Again")
        break

    messages.append(HumanMessage(content=prompt))
    response = chat_model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot:", response.content)