from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

# List of models you want to test
models_to_test = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite",
    "gemini-3-flash"
]

working_models = []
failed_models = []

for model_name in models_to_test:
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=API_KEY
        )
        
        response = llm.invoke("Say OK")
        
        print(f"✅ {model_name} works")
        working_models.append(model_name)

    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        failed_models.append(model_name)

print("\n🔥 Working Models:", working_models)
print("❌ Failed Models:", failed_models)