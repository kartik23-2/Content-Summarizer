def summary_prompt(text):
    return f"""
    Summarize the following document clearly and concisely:

    {text}
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
    Answer the question strictly based on the context below.
    If the answer is not in the context, say "Not found in document".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """