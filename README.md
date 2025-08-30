from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from datetime import datetime

# File name
doc = SimpleDocTemplate("StudyMate_Documentation.pdf", pagesize=A4)

# Styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CenterTitle", alignment=TA_CENTER, fontSize=20, spaceAfter=20))
styles.add(ParagraphStyle(name="Heading", fontSize=14, spaceAfter=10, textColor=colors.darkblue))
styles.add(ParagraphStyle(name="Body", fontSize=11, spaceAfter=10))

story = []

# --- Cover Page ---
story.append(Spacer(1, 200))
story.append(Paragraph("📘 StudyMate – Q&A Chatbot", styles["CenterTitle"]))
story.append(Paragraph("(Streamlit + Hugging Face + PyMuPDF + IBM Watson TTS)", styles["CenterTitle"]))
story.append(Spacer(1, 50))
story.append(Paragraph("Author: <b>Siva Sushmitha Nowdu</b>", styles["CenterTitle"]))
story.append(Paragraph("Date: " + datetime.today().strftime("%d %B %Y"), styles["CenterTitle"]))
story.append(PageBreak())

# --- Table of Contents (manual for now) ---
story.append(Paragraph("Table of Contents", styles["Heading"]))
toc = [
    ["1. Introduction"],
    ["2. Features"],
    ["3. Tech Stack"],
    ["4. System Architecture"],
    ["5. Installation & Setup"],
    ["6. Usage"],
    ["7. Example"],
    ["8. Environment Variables"],
    ["9. Future Enhancements"],
    ["10. Conclusion"]
]
table = Table(toc, colWidths=[450])
table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                           ("TEXTCOLOR",(0,0),(-1,-1),colors.black),
                           ("ALIGN",(0,0),(-1,-1),"LEFT")]))
story.append(table)
story.append(PageBreak())

# --- Sections ---
sections = {
    "Introduction": """StudyMate is an interactive Q&A chatbot that transforms static PDFs into a dynamic learning experience. 
    Users can upload documents, ask natural language questions, and get accurate, context-driven answers. 
    It integrates document parsing, embeddings, generative AI, and text-to-speech capabilities.""",

    "Features": """- 📂 Upload PDF documents
- 🔍 Retrieve top-k relevant chunks using embeddings
- 🤖 Generate answers with Hugging Face models
- 🎙 Convert answers to speech with IBM Watson TTS
- ⚡ Adjustable parameters (retrieval, model choice)
- 🌐 Simple Streamlit interface""",

    "Tech Stack": """- **Frontend**: Streamlit
- **Backend**: Python
- **Document Processing**: PyMuPDF
- **Embeddings**: Sentence-Transformers
- **Answer Generation**: Hugging Face Transformers
- **Vector Store**: FAISS / Scikit-learn
- **TTS**: IBM Watson TTS""",

    "System Architecture": """1. Upload PDF → extract text using PyMuPDF
2. Chunk text into sections
3. Generate embeddings using Sentence-Transformers
4. Retrieve top-k relevant chunks
5. Generate answer using Hugging Face model
6. Convert text to speech (optional) with IBM Watson
7. Display text/audio in Streamlit""",

    "Installation & Setup": """```bash
git clone <your-repo-url>
cd studymate

python -m venv venv
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt

setx HUGGINGFACE_API_TOKEN "hf_xxx"
setx IBM_WATSON_APIKEY "your_ibm_key"
setx IBM_WATSON_URL "your_ibm_url"

streamlit run app.py
```""",

    "Usage": """1. Launch the app → `streamlit run app.py`
2. Upload PDF(s) under *Ingest Sources*
3. Ask a question in *Ask StudyMate*
4. Get AI-generated text answer (and audio if TTS enabled)
5. Adjust retrieval/model settings as needed""",

    "Example": """**Document:** Indian_Music_QA_Expanded.pdf
**Question:** "What is the earliest origin of Indian music?"
**Answer:** "The earliest origins of Indian music can be traced back to the Vedic period, where chants and hymns formed the basis of musical expression."
""",

    "Environment Variables": """- `HUGGINGFACE_API_TOKEN`: Hugging Face API token
- `IBM_WATSON_APIKEY`: IBM Watson TTS key
- `IBM_WATSON_URL`: IBM Watson service URL""",

    "Future Enhancements": """- Support for DOCX/TXT/HTML files
- Multilingual Q&A + TTS
- Fine-tuned domain-specific models
- Mobile/Chatbot integration
- Cloud deployment""",

    "Conclusion": """StudyMate shows how retrieval + generation + speech synthesis can 
turn passive learning materials into interactive experiences. 
It has applications in education, training, and research."""
}

for title, text in sections.items():
    story.append(Paragraph(title, styles["Heading"]))
    story.append(Paragraph(text, styles["Body"]))
    story.append(Spacer(1, 12))

