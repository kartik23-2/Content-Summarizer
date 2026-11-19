import google.generativeai as genai

#API
genai.configure(api_key="AIzaSyDzecEHxd0WPDsTBNejTB4v85kT4O5ZE54")

model = genai.GenerativeModel("gemini-2.5-flash-lite")

def ask_llm(prompt):
    response = model.generate_content(prompt)
    return response.text