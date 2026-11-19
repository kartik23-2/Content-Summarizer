def summary_prompt(text):
    return f"""
    You are an expert academic assistant.

    Summarize the following document in a clear, structured, and point-wise format.

    Follow this format STRICTLY:

    1. Overview:
       - Give a brief 2-3 line overview of what the document is about.

    2. Key Topics Covered:
       - List all major topics and concepts present in the document.

    3. Detailed Summary:
       - Explain each major topic in points
       - Keep explanations clear and concise
       - Cover definitions, properties, and important ideas

    4. Important Concepts / Theorems:
       - Highlight key formulas, properties, or theorems

    5. Exam Focus / Usage:
       - Mention how this document is useful for exams or learning

    Rules:
    - Use bullet points wherever possible
    - Do not skip important sections
    - Keep it structured and easy to read
    - Avoid unnecessary long paragraphs

    Document:
    {text}

    Summary:
    """


def question_prompt(text):
    return f"""
    Generate 10 meaningful questions from the following document.

    Only output questions.
    Make them diverse (conceptual + factual).

    {text}
    """


def qa_prompt(context, question):
    return f"""
   You are an intelligent assistant answering questions based on a document.

    Follow these rules carefully:

    1. First, search for relevant information in the provided context.
    2. If relevant information is found:
       - Answer using that context
       - Provide a clear, detailed explanation (not just 1-2 lines)
       - Expand the answer for better understanding

    3. If the context contains only partial or unclear information:
       - Use the context as a base
       - Improve and complete the answer using your own knowledge

    4. If the context does NOT contain any relevant information:
       - Answer using your own knowledge
       - Do NOT mention the document

    5. ONLY say "Not found in document" if:
       - The question is completely unrelated to the context AND
       - You are not confident about answering it correctly
    ".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """