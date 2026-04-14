# 📄 AI Study Assistant (RAG-based)

An intelligent document assistant that allows users to **upload files (PDF, TXT, Excel, etc.)** and interact with them using **AI-powered summarization, question generation, and Q&A**.

---

## 🚀 Features

### 📌 1. Automatic Summary

* Upload any document
* The entire content is processed by an LLM
* Generates a **clear and concise summary**

---

### 🎯 2. Question Generator

* Click a button to generate:

  * ✅ 10 meaningful questions
  * ✅ Conceptual + factual mix
* Helps in revision and self-testing

---

### 🤖 3. Ask Anything (Q&A)

* Ask any question related to the document
* Uses **RAG (Retrieval-Augmented Generation)** for:

  * Accurate answers
  * Reduced hallucination

---

### 📂 4. Multi-File Support

Supports:

* PDF (text + scanned via OCR)
* TXT
* Markdown (.md)
* Excel (.xls, .xlsx)
* CSV

---

## 🧠 Architecture

```
User Upload
    ↓
File Loader (Multi-format)
    ↓
Raw Text Extraction
    ↓
Chunking
    ↓
Embeddings (Hugging Face)
    ↓
Vector Store (FAISS)
    ↓
        ┌───────────────┐
        │               │
   Summary         Questions
 (Full Text)      (Full Text)
        │               │
        └───────┬───────┘
                ↓
          Q&A System (RAG)
                ↓
            LLM (Gemini)
                ↓
             Response
```

---

## ⚙️ Tech Stack

### 🔹 Frontend

* Streamlit

### 🔹 Backend

* Python

### 🔹 AI / ML

* Gemini (LLM for reasoning)
* Sentence Transformers (Embeddings)

### 🔹 Vector Database

* FAISS (Fast similarity search)

### 🔹 File Processing

* PyPDF2
* pytesseract (OCR)
* pdf2image
* pandas

---

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/ai-study-assistant.git
cd ai-study-assistant
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup API Key

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

---

### 5. Run the App

```bash
streamlit run app.py
```

---

## 🔐 Environment Variables

| Variable       | Description            |
| -------------- | ---------------------- |
| GEMINI_API_KEY | API key for Gemini LLM |

---

## ⚡ How It Works

### 🔹 Step 1: Upload File

User uploads any supported file.

---

### 🔹 Step 2: Text Extraction

* Extracts text using:

  * PDF parser
  * OCR (if scanned)

---

### 🔹 Step 3: Summary Generation

* Full text → Gemini
* Output: concise summary

---

### 🔹 Step 4: Question Generation

* Full text → Gemini
* Output: 10 questions

---

### 🔹 Step 5: Q&A (RAG)

```
User Question
   ↓
Convert to embedding
   ↓
Search similar chunks (FAISS)
   ↓
Send context + query to LLM
   ↓
Generate answer
```

---

## 📥 Sample Usage

### 📌 Input (User Uploads PDF)

Document: "Operating Systems Notes"

---

### 📌 Output

#### 🔹 Summary

```
This document explains process scheduling, memory management,
and deadlocks in operating systems...
```

---

#### 🔹 Generated Questions

```
1. What is process scheduling?
2. Explain deadlock conditions.
3. What is paging in memory management?
...
```

---

#### 🔹 Q&A

**User:**

```
What is deadlock?
```

**AI Response:**

```
Deadlock is a situation where multiple processes are unable to proceed
because each is waiting for resources held by others...
```

---

## 📁 Project Structure

```
project/
│── app.py
│── requirements.txt
│
├── utils/
│   ├── content_loader.py
│   ├── chunker.py
│
├── rag/
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│
├── llm/
│   ├── gemini_client.py
│   ├── prompts.py
```

---

## ⚠️ Limitations

* Large PDFs may hit token limits
* OCR accuracy depends on image quality
* No chat history (yet)

---

## 🔮 Future Improvements

* Chat history (like ChatGPT)
* Better UI/UX
* Support for DOCX, PPTX
* Streaming responses
* Multi-document querying
* input for video URL's

---

## 🙌 Contribution

Feel free to fork and improve the project!

---

## 📜 License

This project is for educational purposes.
