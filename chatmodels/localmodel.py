import os 
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

#create local llm by downloading model from huggingface 

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"cache_dir": "D:/AI_Models"}, #for storing at particular location for Models
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03))

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is name of actress in Dhurandhar Movie")
print(response.content)