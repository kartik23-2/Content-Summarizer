
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "gemini-2.5-flash-lite",
    generation_config={
        "temperature": 0.2,   
        "top_p": 0.8,
        "top_k": 40
       }
    )

def ask_llm(prompt):
    
        time.sleep(4)  # Prevent rate limit
        response = model.generate_content(prompt)
        return response.text
    