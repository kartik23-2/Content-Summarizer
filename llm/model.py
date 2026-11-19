
import os
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "gemini-1.5-flash-latest",
    generation_config={
        "temperature": 0.2,   
        "top_p": 0.8,
        "top_k": 40
       }
    )

def ask_llm(prompt):
    response = model.generate_content(prompt)
    return response.text