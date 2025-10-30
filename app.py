import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(
    page_title="🌐 Language Translator Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🌐 Language Translator Chatbot")
st.markdown(
    """
    👋 Welcome to your personal **AI-powered Translator Chatbot**!  
    Type any sentence in your language, and I’ll translate it into your selected language.  

    💬 **Features:**
    - Supports English, Hindi, Marathi, French, Spanish, and German  
    - Built using **Hugging Face Transformers**  
    - Deployed on **Hugging Face Spaces** using **Streamlit**

    ---
    """
)

LANGS = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "French": "fr",
    "Spanish": "es",
    "German": "de"
}

MODEL_MAP = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "mr"): "Helsinki-NLP/opus-mt-en-mr",
    ("mr", "en"): "Helsinki-NLP/opus-mt-mr-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en"
}

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    return translator

text = st.text_area("✏️ Enter text to translate:", height=150)
col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("Source language", list(LANGS.keys()), index=0)
with col2:
    tgt_lang = st.selectbox("Target language", list(LANGS.keys()), index=1)

if st.button("🚀 Translate"):
    if not text.strip():
        st.warning("Please enter some text.")
    elif src_lang == tgt_lang:
        st.info("Source and target languages are the same.")
    else:
        model_name = MODEL_MAP.get((LANGS[src_lang], LANGS[tgt_lang]))
        if not model_name:
            st.error("❌ No model found for this language pair.")
        else:
            with st.spinner("⏳ Loading model... please wait (first time only)"):
                translator = load_model(model_name)
                output = translator(text, max_length=512)
                translated_text = output[0]['translation_text']

            st.success("✅ Translation complete!")
            st.subheader("Translated text:")
            st.write(translated_text)
