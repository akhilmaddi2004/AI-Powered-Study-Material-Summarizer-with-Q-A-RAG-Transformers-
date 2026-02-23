```
# 🚀 AI-Powered Study Material Summarizer + Q&A System

An intelligent AI-based application that automatically summarizes large PDF documents into structured topic-wise notes and allows users to ask questions from the document using Retrieval-Augmented Generation (RAG).

---

## 📌 Project Overview
Students, teachers, researchers, and professionals often deal with large PDF files that are time-consuming to read. This system solves that problem by automatically:

- Extracting text and images from PDFs
- Detecting topics
- Generating smart summaries
- Providing question-answering support

---

## 🎯 Problem Statement
Reading long PDFs is slow and inefficient. Important information is hard to identify quickly, and manual note-making takes extra effort. There is no simple tool that summarizes PDFs topic-wise with images and allows document-based questioning.

---

## 💡 Solution
This project provides an AI-powered system that:

- Reads PDF files automatically
- Understands document structure
- Summarizes content topic-wise
- Extracts images from PDF
- Generates a clean summarized PDF
- Answers user questions based on document content

---

## 🧠 AI Models Used

### 🔹 DistilBART (Summarization Model)
Used to convert long text into short, meaningful summaries.

### 🔹 RoBERTa + RAG (Question Answering)
Used to answer user questions using document content as knowledge source.

---

## 🏗️ System Architecture Workflow

User Upload PDF  
→ Text & Image Extraction (PyMuPDF)  
→ Topic Detection (NLTK)  
→ Text Chunking  
→ AI Summarization (DistilBART)  
→ Summary Generation  
→ PDF Output  
→ Question Answering (RAG + RoBERTa)

---

## 🛠️ Technologies Used

- Python
- Streamlit
- PyMuPDF
- Transformers (HuggingFace)
- Torch
- NLTK
- FPDF
- Pillow (Image Handling)

---

## 📂 Project Structure

```
```
AI_PDF_Summarizer/
│── app.py
│── requirements.txt
│── nltk_data/
│── README.md

```

---

## ⚙️ Installation

Clone repository:

```

git clone [https://github.com/yourusername/AI_PDF_Summarizer.git](https://github.com/yourusername/AI_PDF_Summarizer.git)
cd AI_PDF_Summarizer

```

Install dependencies:

```

pip install -r requirements.txt

```

Run project:

```

streamlit run app.py

```

---

## 📥 Required Libraries

```

streamlit
pymupdf
torch
transformers
sentencepiece
nltk
fpdf
pillow

```

---

## ✨ Features

✔ Upload any PDF document  
✔ Automatic topic detection  
✔ AI-generated summaries  
✔ Extracts images from PDF  
✔ Structured summarized output  
✔ Downloadable summary PDF  
✔ Question answering from document  

---

## 📊 Results

- Summarizes large PDFs within seconds
- Reduces reading time by ~70%
- Generates structured notes automatically
- Improves understanding efficiency

---

## 👥 End Users

- Students
- Teachers
- Researchers
- Professionals
- Competitive exam learners

---

## 🔮 Future Improvements

- Multilingual PDF support
- Voice-based question answering
- Cloud deployment
- Mobile app version
- Highlight key concepts automatically

---

## 🏆 Wow Factors

- Fully automated AI processing
- Topic-wise summaries
- Integrated Q&A system
- Image extraction
- One-click summary download

