import streamlit as st
from io import StringIO
from docx import Document
import re
import math
from collections import defaultdict, Counter

# --- Stopwords (Spanish, can be extended) ---
STOP_WORDS = {
    'que', 'con', 'por', 'para', 'una', 'los', 'las', 'del', 'como',
    'sin', 'sobre', 'entre', 'hasta', 'desde', 'esta', 'este', 'esto',
    'son', 'hay', 'muy', 'más', 'pero', 'sus', 'ser', 'ese', 'esa',
    'uno', 'dos', 'tres', 'todo', 'toda', 'todos', 'todas', 'otro',
    'otra', 'otros', 'otras', 'mismo', 'misma', 'mismos', 'mismas'
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\sáéíóúñü]', ' ', text)
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    return words

def read_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def compute_tf_idf(docs):
    # docs: dict of {name: content}
    processed = {name: preprocess_text(content) for name, content in docs.items()}
    vocabulary = set(w for words in processed.values() for w in words)
    doc_freq = defaultdict(int)
    for words in processed.values():
        for w in set(words):
            doc_freq[w] += 1
    tf_idf = {}
    N = len(docs)
    for name, words in processed.items():
        tf_idf[name] = {}
        total = len(words)
        word_count = Counter(words)
        for term in vocabulary:
            tf = word_count[term] / total if total > 0 else 0
            idf = math.log(N / doc_freq[term]) if doc_freq[term] > 0 else 0
            tf_idf[name][term] = tf * idf
    return tf_idf, vocabulary

def create_query_vector(query, vocabulary):
    words = preprocess_text(query)
    qvec = {}
    for w in words:
        if w in vocabulary:
            qvec[w] = qvec.get(w, 0) + 1
    return qvec

def cosine_similarity(qvec, dvec):
    dot = sum(qvec.get(k, 0) * dvec.get(k, 0) for k in set(qvec) | set(dvec))
    qmag = math.sqrt(sum(v ** 2 for v in qvec.values()))
    dmag = math.sqrt(sum(v ** 2 for v in dvec.values()))
    if qmag == 0 or dmag == 0:
        return 0
    return dot / (qmag * dmag)

def highlight_terms(text, query):
    # Highlight full query first
    if query.strip():
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    # Then highlight each word (avoid double-highlighting)
    for term in set(preprocess_text(query)):
        if term and len(term) > 2:
            pattern = re.compile(rf'(?<!<mark>)\b{re.escape(term)}\b(?!</mark>)', re.IGNORECASE)
            text = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    return text

st.title("Buscador de documentos")
st.write("---")
st.markdown("Grupo 7")
st.markdown("Soria, C., Zurita, M.")
st.markdown("Modelo algebraico")

st.header('Selecciona tus documentos', divider='rainbow')

uploaded_files = st.file_uploader(
    "Arrastra y suelta hasta 20 archivos (.txt, .docx)",
    type=["txt", "docx"],
    accept_multiple_files=True,
    help="Puedes subir hasta 20 archivos"
)

if uploaded_files:
    if len(uploaded_files) > 20:
        st.warning("Por favor, sube un máximo de 20 archivos.")
    else:
        docs = {}
        for file in uploaded_files:
            docs[file.name] = read_file(file)

        query = st.text_input("Escribe tu consulta")
        if query:
            tf_idf, vocabulary = compute_tf_idf(docs)
            qvec = create_query_vector(query, vocabulary)
            results = []
            for name, content in docs.items():
                sim = cosine_similarity(qvec, tf_idf[name])
                results.append((name, content, sim))
            results.sort(key=lambda x: x[2], reverse=True)
            st.subheader("Resultados ordenados por relevancia:")
            for i, (name, content, sim) in enumerate(results):
                total_words = len(content.split())
                match_words = sum(1 for _ in re.finditer(re.escape(query), content, re.IGNORECASE))
                percentage = (match_words / total_words * 100) if total_words > 0 else 0
                expander_title = (
                    f"**{i+1}.** {name} "
                    f"(Similitud: {sim:.4f} | "
                    f"Coincidencias: {match_words} | "
                    f"Porcentaje: {percentage:.2f}%)"
                )
                with st.expander(expander_title):
                    highlighted_content = highlight_terms(content, query)
                    st.markdown(
                        f"""
                        <div style="max-height:400px;overflow:auto;white-space:pre-wrap;">
                            {highlighted_content}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
else:
    st.info("Sube tus archivos para comenzar la búsqueda.")