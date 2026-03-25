from dotenv import load_dotenv
load_dotenv()
import os

from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HF_HOME"] = "D:/AI_Embeddings" # it only stores metadata here but .cache file of the model will stored in users/.cache for hugging face model

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2")

vector = embedding_model.embed_query("What is Artificial Intelligence?")

print(vector)