import streamlit as st
from io import StringIO
from docx import Document

def read_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def rank_documents(query, docs):
    # Simple ranking: count occurrences of query in each doc
    ranked = []
    for name, content in docs.items():
        score = content.lower().count(query.lower())
        ranked.append((name, content, score))
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked

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
            ranked_docs = rank_documents(query, docs)
            st.subheader("Resultados ordenados por relevancia:")
            def highlight(text, term):
                import re
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                # Wrap matches in <mark>
                return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
            for i, (name, content, score) in enumerate(ranked_docs):
                total_words = len(content.split())
                match_words = score
                percentage = (match_words / total_words * 100) if total_words > 0 else 0
                expander_title = (
                    f"**{i+1}.** {name} "
                    f"(Relevancia: {score} | "
                    f"Coincidencias: {match_words} | "
                    f"Porcentaje: {percentage:.2f}%)"
                )
                with st.expander(expander_title):
                    # Highlight and show the whole document in a scrollable div
                    highlighted_content = highlight(content, query)
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