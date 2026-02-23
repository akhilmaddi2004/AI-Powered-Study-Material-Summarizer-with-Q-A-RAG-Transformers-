import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from io import BytesIO
from fpdf import FPDF
import re
import io
import tempfile
import os
from PIL import Image  # Added for better image handling in PDF
import time  # To show the speed difference

# ----------------- NLTK Setup -----------------
# We check if the folder exists to prevent errors on different machines
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download necessary NLTK data quietly
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.data.path.append(nltk_data_dir)

# ----------------- Streamlit UI Configuration -----------------
st.set_page_config(
    page_title="AI-Powered Study Material Summarizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapsed because we are using One Frame
)

# ----------------- 🔥 "Frosted Glass" UI CSS (Enhanced) 🔥 -----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        color: #E0E0E0;
    }

    /* Overall Background */
    .stApp {
        background: linear-gradient(135deg, #1A1A2E 0%, #0F0C29 100%);
        color: #E0E0E0;
    }

    /* Header styling */
    h1 {
        color: #00FFC2;
        text-align: center;
        font-weight: 700;
        font-size: 3.5em;
        text-shadow: 2px 2px 8px rgba(0, 255, 194, 0.4);
        margin-bottom: 0.5em;
    }
    h2 {
        color: #9DEDF8;
        font-weight: 600;
        border-bottom: 2px solid rgba(157, 237, 248, 0.3);
        padding-bottom: 10px;
        margin-top: 2em;
        margin-bottom: 1em;
    }
    h3 {
        color: #FFD700;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }
    
    /* --- Main Control Panel (The "One Frame" Container) --- */
    .css-1r6slb0 {
        background: rgba(10, 10, 31, 0.6);
        border: 2px solid #00FFC2;
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }

    /* File Uploader Styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #00FFC2;
        padding: 20px;
        border-radius: 15px;
        background-color: rgba(0, 255, 194, 0.05);
        text-align: center;
    }
    [data-testid="stFileUploader"] label {
        color: #00FFC2;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    /* Slider Styling */
    [data-testid="stSelectSlider"] label {
        color: #9DEDF8;
        font-weight: 600;
        font-size: 1.1em;
    }

    /* Buttons */
    [data-testid="stDownloadButton"] button {
        background-color: #00FFC2;
        color: #1A1A2E;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 1em 2em;
        font-size: 1.2em;
        box-shadow: 5px 5px 20px rgba(0, 255, 194, 0.5);
        transition: all 0.3s ease-in-out;
        width: 100%;
        margin-top: 2em;
    }
    [data-testid="stDownloadButton"] button:hover {
        transform: scale(1.02);
        box-shadow: 0px 0px 25px rgba(0, 255, 194, 0.8);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {
        background-color: #2D2D44;
        color: #E0E0E0;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #00FFC2;
        color: #1A1A2E !important;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #2D2D44;
        border: 1px solid #00FFC2;
        padding: 20px;
        border-radius: 10px;
    }

    /* --- 🛑 GLITCH REMOVAL (Fixing Arrows & Toolbar) 🛑 --- */
    
    /* Hide the top Streamlit header completely */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Hide the footer */
    footer { 
        display: none !important; 
    }
    
    /* Hide the "Link" arrows next to headers (The double arrow glitch) */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    a.anchor-link {
        display: none !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Load AI Summarizer -----------------
@st.cache_resource
def load_ai_brains():
    with st.spinner("Initializing AI Neural Networks... 🚀"):
        # Brain 1: The Summarizer (For the overall report)
        summ = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        # Brain 2: The QA Expert (Genuine answers from original text)
        qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
        return summ, qa

# Start both AI brains together
summarizer, qa_model = load_ai_brains()

# ----------------- Utility Functions -----------------

def sanitize_text(text):
    """
    Cleans text to ensure it works with PDF generation.
    Replaces special characters that often break FPDF.
    """
    if not text:
        return ""
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2022": "*",
        "\u2212": "-", "\u00b0": " degrees", "\u2103": " Celsius",
        "\u2109": " Fahrenheit", "\u03b1": "alpha", "\u03b2": "beta",
        "\u03b3": "gamma", "\u03c0": "pi", "\u03a9": "Ohm",
        "\u03bc": "micro", "\u2126": "Ohm", "\u2122": "TM",
        "\u221e": "infinity", "\u2260": "!=", "\u2264": "<=", "\u2265": ">="
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Encode to Latin-1 to be safe for standard PDF fonts
    text = text.encode("latin-1", "replace").decode("latin-1")
    return text

def chunk_text(text, max_words=500):
    """
    Splits long text into smaller chunks so the AI can process them.
    Respects sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks, chunk, word_count = [], "", 0
    for sentence in sentences:
        wc = len(sentence.split())
        if word_count + wc > max_words:
            chunks.append(chunk.strip())
            chunk, word_count = sentence, wc
        else:
            chunk += " " + sentence
            word_count += wc
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def get_max_length(option):
    """Maps the user slider selection to token counts."""
    return {"short": 50, "medium": 120, "detailed": 200}[option]

def extract_topics_and_images(pdf_file):
    """
    Extracts text and organizes it by topics/headers.
    Also extracts images associated with pages.
    """
    pdf_file.seek(0)
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    topic_texts = {}
    topic_images = {}
    topic_counter = 1
    current_topic = "Section 1" 

    for page_num, page in enumerate(doc):
        # Extract Text Blocks
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue
            
            # Heuristic to find Headers
            is_header = False
            if re.match(r"(Topic|Chapter|Section|Unit)\s*[\d\.]+\b.*", text, flags=re.IGNORECASE):
                is_header = True
            elif (len(text.split()) < 15 and re.match(r"^[A-Z][a-zA-Z0-9\s,&'-]+$", text) and page_num > 0):
                is_header = True
            
            if is_header:
                current_topic = text
                topic_texts.setdefault(current_topic, "")
                topic_images.setdefault(current_topic, [])
            elif len(text.split()) > 50 and current_topic not in topic_texts:
                # If we have a long block but no topic yet, make a generic one
                current_topic = f"Section {topic_counter}"
                topic_counter += 1
                topic_texts.setdefault(current_topic, "")
                topic_images.setdefault(current_topic, [])
                topic_texts[current_topic] += text + " "
            else:
                topic_texts.setdefault(current_topic, "")
                topic_texts[current_topic] += text + " "

        # Extract Images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("png")
                pix = None 
                img_stream = io.BytesIO(img_bytes)
                topic_images.setdefault(current_topic, [])
                topic_images[current_topic].append(img_stream)
            except Exception as e:
                pass 
    doc.close()
    return topic_texts, topic_images

def save_summary_pdf(topics_summary, images, file_name="summary.pdf"):
    """
    Generates a professional PDF report with the summaries and images.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for topic, summary in topics_summary.items():
        # Add Topic Header
        pdf.set_font("Arial", 'B', 14)
        pdf.multi_cell(0, 10, sanitize_text(topic))
        pdf.set_font("Arial", size=12)
        
        # Add Summary Text
        for line in summary.split("\n"):
            clean_line = sanitize_text(line)
            if clean_line.strip().startswith("*"):
                pdf.multi_cell(0, 8, "  " + clean_line.replace("*", "-").strip())
            elif clean_line.strip() != "":
                pdf.multi_cell(0, 8, clean_line)
        pdf.ln(5)

        # Add Associated Images
        for img_stream in images.get(topic, []):
            try:
                img_stream.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(img_stream.read())
                    temp_path = tmp_file.name
                
                with Image.open(temp_path) as pil_img:
                    w_img, h_img = pil_img.size
                
                max_pdf_width = 180 
                max_pdf_height = 100 
                aspect_ratio = w_img / h_img
                
                if w_img > h_img:
                    final_w = min(w_img, max_pdf_width)
                    final_h = final_w / aspect_ratio
                else:
                    final_h = min(h_img, max_pdf_height)
                    final_w = final_h * aspect_ratio
                
                if final_w > max_pdf_width:
                    final_w = max_pdf_width
                    final_h = final_w / aspect_ratio

                pdf.image(temp_path, w=final_w, h=final_h)
                os.remove(temp_path)
                pdf.ln(5)
            except Exception as e:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
                continue

    pdf.output(file_name)
    return file_name

# ----------------- Main UI Layout (Unified Frame) -----------------

# --- Hero Section ---
st.markdown("""
<div style="text-align: center; padding: 20px 0; background: rgba(10, 10, 31, 0.7); backdrop-filter: blur(10px); border-radius: 15px; margin-bottom: 2em; box-shadow: 0 8px 30px rgba(0, 255, 194, 0.2); border: 1px solid rgba(0, 255, 194, 0.3);">
    <h1>🚀 AI-Powered Study Material Summarizer 🧠</h1>
    <p style="font-size: 1.3em; color: #9DEDF8;">
        Unlock knowledge from your PDFs with intelligent, topic-wise summaries.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Control Panel (One Frame UI) ---
# We use columns to put the uploader and controls in the center, not sidebar
col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])

with col_center:
    # Use a clean URL icon instead of Base64 to prevent glitches
    st.markdown(
        f'<div style="display: flex; justify-content: center; margin-bottom: 20px;"><img src="https://cdn-icons-png.flaticon.com/512/2040/2040946.png" alt="AI Icon" style="width: 80px; height: 80px;"></div>',
        unsafe_allow_html=True
    )
    
    st.markdown("<h3 style='text-align: center;'>Upload & Configure</h3>", unsafe_allow_html=True)
    
    # 1. File Uploader (In Main Frame)
    uploaded_file = st.file_uploader(
        "📚 Drop your PDF here!",
        type=["pdf"]
    )
    
    # 2. Settings (In Main Frame)
    if uploaded_file:
        summary_length = st.select_slider(
            "📏 Choose summary detail:",
            options=["short", "medium", "detailed"],
            value="medium"
        )
    else:
        # Default value if nothing uploaded yet
        summary_length = "medium" 

# ----------------- Processing Logic -----------------
if uploaded_file is not None:
    st.divider()
    
    try:
        # --- Stage 1: Extraction ---
        st.subheader("Stage 1: Document Analysis 📝")
        
        with st.status("Extracting topics & images... ⏳", expanded=True) as status_msg:
            topics, images = extract_topics_and_images(uploaded_file)
            status_msg.update(label="Done! ✅", state="complete", expanded=False)

        if not topics or all(v.strip() == "" for v in topics.values()):
            st.warning("⚠️ Uh oh! No readable text detected. This PDF might be scanned or image-based.")
            st.stop()

        st.success(f"🥳 Successfully identified **{len(topics)}** key topics/sections!")
        
        # --- Stage 2: AI Summarization ---
        st.subheader("Stage 2: AI Summarization 🤖")
        max_len = get_max_length(summary_length)
        
        all_chunks_to_summarize = []
        chunk_to_topic_map = [] 
        
        with st.spinner("Preparing chunks for batch processing..."):
            for topic, text in topics.items():
                if not text.strip():
                    continue
                chunks = chunk_text(text)
                all_chunks_to_summarize.extend(chunks)
                chunk_to_topic_map.extend([topic] * len(chunks))
        
        if not all_chunks_to_summarize:
            st.error("❌ No text chunks found to summarize.")
            st.stop()

        st.info(f"Starting batch summarization for **{len(all_chunks_to_summarize)}** text chunks. This is *much* faster!")
        
        topics_summary_lists = {topic: [] for topic in topics}
        summarized_outputs = []
        start_time = time.time()
        
        # Using Batch Processing for Speed
        with st.status("AI is processing all chunks at once... ✨", expanded=True) as status_msg:
            try:
                summarized_outputs = summarizer(
                    all_chunks_to_summarize,
                    max_length=max_len,
                    min_length=30,
                    do_sample=False,
                    batch_size=8  # Efficient batching
                )
                end_time = time.time()
                status_msg.update(label="Batch summarization complete! 🎉", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error during batch summarization. The document might be too large. {e}")
                st.stop()

        # Re-assemble the summaries
        for i, summary_dict in enumerate(summarized_outputs):
            topic_name = chunk_to_topic_map[i]
            summary_text = summary_dict['summary_text']
            sentences = nltk.sent_tokenize(summary_text)
            points = "\n".join([f"* {sanitize_text(s.strip())}" for s in sentences])
            topics_summary_lists[topic_name].append(points)

        # Join the summaries
        topics_summary = {}
        for topic, points_list in topics_summary_lists.items():
            if points_list:
                topics_summary[topic] = "\n\n".join(points_list)
        
        st.toast("✅ Processing complete!", icon="🎓")
        st.success(f"✨ All summaries generated in **{end_time - start_time:.2f} seconds**!")
        st.markdown("---")

        # ----------------- 🔥 ACCURATE Q&A: RAG-BASED DOUBT CLEARING 🔥 -----------------
        st.header("Ask Doubts About This Material❓")
        st.write("Ask questions to get **answers** extracted from the original PDF.")

        # Full context from the original extracted PDF text
        mega_context = " ".join(topics.values())

        # Input field and Button layout
        user_query = st.text_input("What would you like to clarify?", key="genuine_qa_input")
        get_answer_btn = st.button("🔍 Get Detailed Answer")

        if get_answer_btn and user_query:
            with st.spinner("Searching original document for a genuine answer..."):
                # 1. SEARCH: Find the exact location in the original PDF
                qa_result = qa_model(question=user_query, context=mega_context)
                
                # 2. RETRIEVE: Get context window around the answer
                start = max(0, qa_result['start'] - 500)
                end = min(len(mega_context), qa_result['end'] + 800)
                relevant_evidence = mega_context[start:end]
                
                # 3. GENERATE: Summarize that evidence into 5+ lines
                detailed_expansion = summarizer(relevant_evidence, max_length=250, min_length=150, do_sample=False)
                final_answer = detailed_expansion[0]['summary_text']
                
                # UI Display Card
                st.markdown(f"""
                <div style="background: rgba(0, 255, 194, 0.05); border: 1px solid #00FFC2; padding: 25px; border-radius: 15px; margin: 20px 0;">
                    <h4 style="color:#00FFC2; margin-top:0;">Detailed Explanation:</h4>
                    <p style="font-size:1.1em; line-height:1.8; color:#FFFFFF;">{final_answer}</p>
                    <hr style="opacity:0.2;">
                    <p style="color:#FFD700; font-size:0.9em;"><b>Direct Evidence:</b> "{qa_result['answer']}"</p>
                </div>
                """, unsafe_allow_html=True)
        elif get_answer_btn and not user_query:
            st.warning("⚠️ Please type a question first!")

        st.markdown("---")

        # --- Stage 3: Interactive Results ---
        st.header("📖 Dive Into Your Smart Summary")
        
        tab_titles = [f"📚 {sanitize_text(t)[:40]}{'...' if len(t) > 40 else ''}" for t in topics_summary.keys()]
        
        with st.container(border=True):
            tabs = st.tabs(tab_titles)
            for i, (topic, summary) in enumerate(topics_summary.items()):
                with tabs[i]:
                    st.markdown(f"### {topic}")
                    st.markdown(summary)
                    
                    if images.get(topic):
                        st.markdown("---")
                        st.subheader("🖼️ Visual Aids from this Section:")
                        img_cols = st.columns(3)
                        for j, img_stream in enumerate(images[topic]):
                            img_stream.seek(0)
                            with img_cols[j % 3]:
                                st.image(img_stream, use_container_width=True, caption=f"Image {j+1}", output_format="PNG")
                        st.markdown("---")

        st.markdown("---")
        
        # --- Stage 4: Download ---
        st.header("⬇️ Grab Your Polished Summary PDF")
        
        pdf_file_path = save_summary_pdf(topics_summary, images)
        with open(pdf_file_path, "rb") as f:
            pdf_bytes = f.read()
        
        st.download_button(
            label="✨ Download Your Summarized PDF! ✨",
            data=pdf_bytes,
            file_name=f"AI_Summary_{uploaded_file.name.replace('.pdf', '')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        os.remove(pdf_file_path) # Clean up the temp file

    except Exception as e:
        st.error(f"💥 An unexpected error occurred. Please try another PDF.")
        st.exception(e)

else:
    # --- Engaging Welcome State (One Frame) ---
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### **🚀 Boost Productivity**")
        st.markdown("Spend less time reading, more time understanding.")
    with col2:
        st.markdown("#### **💡 Crystal Clear Insights**")
        st.markdown("Get the core ideas without the fluff.")
    with col3:
        st.markdown("#### **🎯 Targeted Learning**")
        st.markdown("Focus on what truly matters in each section.")

    st.markdown("<br><p style='text-align: center; color:#9DEDF8; font-size:1.1em;'>Upload your PDF above to unleash the power of AI! 🚀</p>", unsafe_allow_html=True)