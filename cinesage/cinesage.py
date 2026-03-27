from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    max_new_tokens=512)

chat_model = ChatHuggingFace(llm=llm)

#This Program for Understanding Prompt Templates (for Specific Tasks)

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
    """
You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful structured information from a movie paragraph and present it in a clean readable format.

Rules:
- Do NOT add explanations
- Do NOT add extra commentary
- Follow the exact format
- If information is missing → write NULL
- Keep summary short (2–3 lines max)
- Do NOT guess unknown facts

Output Format:

Movie Title:
Release Year:
Genre:
Director:
Main Cast:
Setting/Location:
Plot:
Themes:
Ratings:
Notable Features:

Short Summary:
"""
    ),

    ("human",
    """
Extract information from this paragraph:

{paragraph}
"""
    )
])
  