# app.py
import os
import json
import tempfile
from typing import List, Tuple
from pathlib import Path

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from huggingface_hub import InferenceClient
import fitz  # pymupdf
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env (optional)
load_dotenv()

# ----------------------------
# Config / env vars
# ----------------------------
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")  # optional: for generation
IBM_APIKEY = os.getenv("IBM_WATSON_APIKEY", "")
IBM_URL = os.getenv("IBM_WATSON_URL", "")

# Default embedding model (sentence-transformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small and fast; change if you want higher quality
# If you want to use a different HF model for generation, set below
HF_GENERATION_MODEL = "google/flan-t5-small"  # example - you can change to a preferred instruct model

# ----------------------------
# Utilities: PDF ingestion
# ----------------------------
def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using PyMuPDF (pymupdf)."""
    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text:
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks (approx chunk_size tokens/characters)."""
    # chunking by characters is simple and works fine for many use cases.
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)  # create small overlap
    return chunks

# ----------------------------
# Embedding & Index
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = EMBEDDING_MODEL_NAME):
    st.sidebar.text("Loading embedding model...")
    model = SentenceTransformer(model_name)
    return model

def build_embeddings(doc_texts: List[str], embedder: SentenceTransformer):
    embeddings = embedder.encode(doc_texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_nearest_neighbors(embeddings: np.ndarray, n_neighbors: int = 5):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    return nn

# ----------------------------
# Hugging Face generation (optional)
# ----------------------------
def hf_generate_answer(question: str, contexts: List[str], hf_token: str, model: str = HF_GENERATION_MODEL, max_length: int = 256) -> str:
    """
    Use Hugging Face Inference API to generate an answer given contexts and a user question.
    Requires HF token in env (HUGGINGFACE_API_TOKEN) if using inference.huggingface.co.
    """
    if not hf_token:
        # fallback: simple template reply using contexts
        reply = "Context retrieved:\n\n" + "\n---\n".join(contexts[:3]) + f"\n\nQuestion: {question}\n\nAnswer (based on retrieved context):\n"
        return reply

    client = InferenceClient(token=hf_token)
    prompt = (
        "You are a helpful tutor. Use the provided context passages to answer the question concisely.\n\n"
        "Context:\n"
        f"{'\\n\\n'.join(contexts)}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    # call text generation model
    try:
        res = client.text_generation(model=model, inputs=prompt, parameters={"max_new_tokens": max_length, "temperature":0.2})
        # depending on HF response shape:
        if isinstance(res, dict) and "generated_text" in res:
            return res["generated_text"]
        # sometimes returns list:
        if isinstance(res, list) and res and isinstance(res[0], dict) and "generated_text" in res[0]:
            return res[0]["generated_text"]
        # fallback stringify
        return str(res)
    except Exception as e:
        return f"(Generation error) {e}"

# ----------------------------
# IBM Watson TTS helper
# ----------------------------
def ibm_tts_synthesize(text: str, apikey: str, url: str, voice: str = "en-US_AllisonV3Voice") -> bytes:
    """Synthesize text to audio (mp3 bytes) using IBM Watson TTS."""
    if not apikey or not url:
        raise RuntimeError("IBM Watson credentials not provided.")
    auth = IAMAuthenticator(apikey)
    tts = TextToSpeechV1(authenticator=auth)
    tts.set_service_url(url)
    # synthesize to mp3
    response = tts.synthesize(text, voice=voice, accept="audio/mp3").get_result()
    audio_bytes = response.content
    return audio_bytes

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="StudyMate — Streamlit RAG Chatbot", layout="wide")
st.title("StudyMate — Q&A Chatbot (Streamlit + HF + PyMuPDF + IBM TTS)")

# Sidebar: model / options
st.sidebar.header("Configuration")
embedder = load_embedder()
top_k = st.sidebar.slider("Retrieval top-k", value=5, min_value=1, max_value=10)
use_generation = st.sidebar.checkbox("Use Hugging Face generation (requires HUGGINGFACE_API_TOKEN)", value=True)
hf_model = st.sidebar.text_input("HF generation model", HF_GENERATION_MODEL)
use_tts = st.sidebar.checkbox("Enable IBM Watson TTS", value=False)
if use_tts and (not IBM_APIKEY or not IBM_URL):
    st.sidebar.warning("Set IBM_WATSON_APIKEY and IBM_WATSON_URL env vars to use TTS.")

# Storage in session state
if "docs" not in st.session_state:
    st.session_state.docs = []           # each doc: {"source": name, "chunks": [...], "embeddings": np.array}
if "index_embeddings" not in st.session_state:
    st.session_state.index_embeddings = None
if "index_docs_map" not in st.session_state:
    st.session_state.index_docs_map = []  # maps global vector idx -> (doc_name, chunk_text)
if "nn" not in st.session_state:
    st.session_state.nn = None

# Upload PDFs / add text
st.sidebar.header("Ingest sources")
uploaded = st.sidebar.file_uploader("Upload PDF(s) or TXT", accept_multiple_files=True, type=["pdf","txt"])
if uploaded:
    for f in uploaded:
        fname = f.name
        with st.spinner(f"Reading {fname}..."):
            # write to temp file if PDF to read with PyMuPDF
            if fname.lower().endswith(".pdf"):
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tf.write(f.read())
                tf.flush()
                txt = extract_text_from_pdf(tf.name)
                tf.close()
            else:
                txt = f.read().decode("utf-8")
            # chunk
            chunks = chunk_text(txt, chunk_size=800, overlap=150)
            # create embeddings
            st.sidebar.text(f"Embedding {len(chunks)} chunks from {fname} ...")
            embeddings = build_embeddings(chunks, embedder)
            # store
            st.session_state.docs.append({"source": fname, "chunks": chunks, "embeddings": embeddings})
            st.sidebar.success(f"Ingested {fname} → {len(chunks)} chunks")
    # After ingesting new docs, rebuild global index
    # flatten
    all_embeddings = []
    idx_map = []
    for doc in st.session_state.docs:
        for chunk, emb in zip(doc["chunks"], doc["embeddings"]):
            all_embeddings.append(emb)
            idx_map.append((doc["source"], chunk))
    if all_embeddings:
        all_embeddings_np = np.vstack(all_embeddings)
        st.session_state.index_embeddings = all_embeddings_np
        st.session_state.index_docs_map = idx_map
        st.session_state.nn = build_nearest_neighbors(all_embeddings_np, n_neighbors=top_k)

# Allow manual add of Q/A or text source
st.sidebar.header("Quick add text")
manual_text = st.sidebar.text_area("Paste text to add as a source (press Add)", height=120)
if st.sidebar.button("Add text source"):
    if manual_text.strip():
        chunks = chunk_text(manual_text, chunk_size=800, overlap=150)
        embeddings = build_embeddings(chunks, embedder)
        name = f"manual_{len(st.session_state.docs)+1}"
        st.session_state.docs.append({"source": name, "chunks": chunks, "embeddings": embeddings})
        st.success(f"Added manual source: {name} ({len(chunks)} chunks)")
        # rebuild global index
        all_embeddings = []
        idx_map = []
        for doc in st.session_state.docs:
            for chunk, emb in zip(doc["chunks"], doc["embeddings"]):
                all_embeddings.append(emb)
                idx_map.append((doc["source"], chunk))
        if all_embeddings:
            st.session_state.index_embeddings = np.vstack(all_embeddings)
            st.session_state.index_docs_map = idx_map
            st.session_state.nn = build_nearest_neighbors(st.session_state.index_embeddings, n_neighbors=top_k)

# Clear data
if st.sidebar.button("Clear all ingested sources"):
    st.session_state.docs = []
    st.session_state.index_embeddings = None
    st.session_state.index_docs_map = []
    st.session_state.nn = None
    st.sidebar.success("Cleared.")

st.markdown("### Ingested sources")
if st.session_state.docs:
    for d in st.session_state.docs:
        st.write(f"- **{d['source']}** — {len(d['chunks'])} chunks")
else:
    st.info("No sources ingested yet. Upload PDFs or paste text in the sidebar.")

# Main chat UI
st.markdown("---")
st.header("Ask StudyMate")
question = st.text_input("Type your question here (press Enter or Ask):")
ask_btn = st.button("Ask")

def retrieve_top_k(question: str, k: int = 5) -> List[Tuple[int, float]]:
    """Return list of (idx, distance) for top-k similar chunks."""
    if st.session_state.index_embeddings is None or st.session_state.nn is None:
        return []
    q_emb = embedder.encode([question], convert_to_numpy=True)
    dists, idxs = st.session_state.nn.kneighbors(q_emb, n_neighbors=min(k, st.session_state.index_embeddings.shape[0]))
    dists = dists[0]
    idxs = idxs[0]
    # convert cosine distance to similarity (1 - dist)
    return list(zip(idxs.tolist(), dists.tolist()))

if ask_btn and question.strip():
    with st.spinner("Retrieving relevant context..."):
        results = retrieve_top_k(question, k=top_k)
    if not results:
        st.warning("No knowledge sources indexed yet. Upload PDFs or paste text in sidebar first.")
    else:
        contexts = []
        for idx, dist in results:
            src, chunk_text = st.session_state.index_docs_map[idx]
            contexts.append(f"Source: {src}\n{chunk_text}")
        # show retrieved
        st.subheader("Retrieved context (top results)")
        for i, (idx, dist) in enumerate(results):
            src, chunk = st.session_state.index_docs_map[idx]
            similarity = 1 - dist
            st.markdown(f"**{i+1}. Source:** {src} — **similarity:** {similarity:.3f}")
            st.write(chunk[:800] + ("..." if len(chunk) > 800 else ""))
        # generate using HF or fallback
        st.subheader("Answer")
        ans = hf_generate_answer(question, contexts, HF_TOKEN if use_generation else "", model=hf_model)
        st.write(ans)

        # Optionally synthesize via IBM Watson
        if use_tts and IBM_APIKEY and IBM_URL:
            try:
                audio_bytes = ibm_tts_synthesize(ans, IBM_APIKEY, IBM_URL)
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.error(f"IBM TTS error: {e}")

# Footer: small help text
st.markdown("---")
st.markdown(
    "### Welcome to StudyMate 🎓"
    #"""
   # **How it works:**  
    #1. Upload PDFs or paste texts; the app extracts text with PyMuPDF and splits into chunks.  
   # 2. Sentence embeddings are produced (sentence-transformers).  
   # 3. At query time the app finds top-k semantically-similar chunks (NearestNeighbors).  
   # 4. Optionally, the retrieved contexts + question are fed to Hugging Face Inference API to generate a concise answer.  
   # 5. Optionally, IBM Watson Text-to-Speech can synthesize the reply to audio.  
   # """
)