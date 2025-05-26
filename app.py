import streamlit as st
import os
import re
import numpy as np
import unicodedata
from nltk.corpus import stopwords
from nltk import download
from docx import Document
import pandas as pd
from functools import reduce

# descargar las stopwords de nltk si no están presentes
download('stopwords')
spanish_stopwords = set(stopwords.words('spanish'))
english_stopwords = set(stopwords.words('english'))
stopwords_set = spanish_stopwords | english_stopwords

# funciones crudas
def load_text(uploaded_file):
    """Carga el texto de un archivo .txt o .docx"""
    if uploaded_file.name.endswith('.txt'):
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return content
    elif uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        return '\n'.join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Tipo de archivo no soportado")

def normalize_text(text):
    """Convierte el texto a minúsculas y elimina acentos"""
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

def tokenize(text):
    """Divide el texto en palabras usando expresiones regulares"""
    return re.findall(r'\b\w+\b', text)

def clean_words(text):
    """Tokeniza, normaliza y elimina stopwords del texto"""
    tokens = tokenize(normalize_text(text))
    return [word for word in tokens if word not in stopwords_set and word.isalpha()]

def cosine_similarity(vec1, vec2):
    """Calcula la similitud coseno entre dos vectores"""
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def vectorize(words, vocab):
    """Convierte una lista de palabras en un vector binario según el vocabulario"""
    return np.array([1 if word in words else 0 for word in vocab], dtype=int)

def highlight_terms(text, query):
    terms = set(clean_words(query))
    for term in sorted(terms, key=len, reverse=True):
        if term:
            pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
            text = pattern.sub(r'<mark>\1</mark>', text)
    return text

st.sidebar.title("Buscador de documentos")
st.sidebar.header("Grupo 7", divider='rainbow')
st.sidebar.markdown("Soria, C., Zurita, M.")
st.sidebar.markdown("Modelo algebraico")

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
        # paso 1: cargar los documentos y guardarlos en un array raw_docs
        raw_docs = [load_text(uploaded_file) for uploaded_file in uploaded_files]
        
        st.subheader("Paso 1")
       
        # ---impresión del paso 1 en el navegador
        st.write("En el paso 1 del modelo algebraico (similitud coseno) de recuperación de información, se cargan los documentos y se guarda el contenido del texto de cada uno en un array `raw_docs`.")
        with st.expander("Ver salida de consola"):
            step_one_output=["1. Guardar los documentos en un array raw_docs:"]
            for i, doc in enumerate(raw_docs, 1):
                # asegurarse de que doc es una cadena de texto
                if isinstance(doc, (bytes, memoryview)):
                    doc = doc.tobytes().decode('utf-8') if isinstance(doc, memoryview) else doc.decode('utf-8')
                words = doc.split()
                step_one_output.append(f"Documento {i} (raw_docs[{i-1}]): {len(words)} palabras. Primeras 5 palabras: {words[:5]}")
            step_one_output.append("NOTA: raw_docs contiene el texto completo de cada documento para cada posición, se ha tokenizado el print para mostrar las primeras 5 palabras.\n")            
            step_one_output = "\n".join(step_one_output)
            st.code(step_one_output, language='plaintext')       
            # --fin del print

        # paso 2: limpiar y tokenizar los documentos y el search_query
        st.subheader("Paso 2")
        search_query = st.text_input("Escribe el texto que deseas buscar")
        
        if search_query:
            st.write("En el paso 2, se limpia y tokeniza el texto de los documentos y la consulta de búsqueda. Se utilizó `ntlk` para eliminar stopwords en inglés y español.")
            clean_docs = list(map(clean_words, raw_docs))
            clean_query = clean_words(search_query)

            with st.expander("Ver salida de consola"):
                # ---impresión del paso 2 en el navegador
                step_two_output=["2. Limpieza y tokenizado de los documentos crudos en un array de 2 dimensiones:"]
                for i, doc in enumerate(clean_docs, 1):
                    step_two_output.append(f"Documento {i} (clean_docs[{i-1}][0 ... 4]): {len(doc)} palabras. Primeras 5 palabras: {doc[:5]}")
                step_two_output.append(f"\nsearch_query: '{search_query}'; clean_query: {clean_query} (L= {len(clean_query)})")
                step_two_output = "\n".join(step_two_output)
                # ---fin del print
                st.code(step_two_output, language='plaintext')

            with st.expander("Ver dataset de stopwords en español"):
                st.write(f"Stopwords en español: {spanish_stopwords}\n\n")

            with st.expander("Ver dataset de stopwords en inglés"):
                st.write(f"Stopwords en inglés: {english_stopwords}\n\n")
            # paso 3: crear el vocabulario único
            st.subheader("Paso 3")
            st.write("En el paso 3, se crea un vocabulario único a partir de las palabras de los documentos y la consulta de búsqueda. El vocabulario se guarda en una lista ordenada `vocab`.")
            vocab = sorted(set(reduce(lambda x, y: x + y, clean_docs, clean_query)))

            with st.expander("Ver salida de consola"):
                # ---impresión del paso 3 en el navegador
                step_three_output = [f"3. Creación del vocabulario único:"]
                step_three_output.append(f"10 primeras palabras de vocab: {vocab[:10]}, L = {len(vocab)} palabras\n")
                step_three_output = "\n".join(step_three_output)

                st.code(step_three_output, language='plaintext')
                # ---fin del print

            # paso 4: crear la matriz de frecuencia binaria
            st.subheader("Paso 4")
            st.write("En el paso 4, se crean los vectores binarios para los documentos y la consulta de búsqueda. Cada vector tiene una longitud igual al tamaño del vocabulario, y contiene 1 si la palabra está presente en el documento o consulta, o 0 si no lo está.")
            doc_vectors = list(map(lambda doc: vectorize(doc, vocab), clean_docs))
            query_vector = vectorize(clean_query, vocab)
            with st.expander("Ver salida de consola"):
                # ---impresión del paso 4 en el navegador
                step_four_output = [f"4. Creación del vectores binarios para documentos y query:"]
                step_four_output.append(f"Vector de la consulta (query_vector): {query_vector} (L = {len(query_vector)} palabras)")
                indices = np.where(query_vector == 1)[0]
                step_four_output.append("Índices con 1 en query vector: " + str(indices.tolist()))
                step_four_output.append("Aciertos: " + str(len(indices)))
                step_four_output.append("Corresponde a las palabras de vocab: " + str([vocab[i] for i in indices]) + "\n")
                for i, (doc_vector, uploaded_file) in enumerate(zip(doc_vectors, uploaded_files), 1):
                    step_four_output.append(f"Vector del Documento {i} - '{uploaded_file.name}' - (doc_vectors[{i-1}]): {doc_vector} (L = {len(doc_vector)} palabras)")
                    indices = np.where(doc_vector == 1)[0]
                    step_four_output.append(f"Índices con 1 en doc_vectors[{i-1}]: " + str(indices.tolist()))
                    step_four_output.append("Aciertos: " + str(len(indices)))
                    step_four_output.append("Corresponde a las 5 primeras palabras de vocab: " + str([vocab[i] for i in indices[:5]]) + "\n")
                step_four_output = "\n".join(step_four_output)
                st.code(step_four_output, language='plaintext')

            # paso 5: calcular la similitud coseno
            st.subheader("Paso 5")
            st.write("En el paso 5, se calcula la similitud coseno entre el vector de la consulta y los vectores de los documentos. Los resultados se ordenan por relevancia. La fórmula de similitud coseno aplicada para el modelo algebraico es:")
            st.latex(r'''
                \text{similitud coseno} =
                \cos(\theta) =
                \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}
            ''')
            scores = [cosine_similarity(query_vector, doc_vector) for doc_vector in doc_vectors]
            ranked = sorted(zip(uploaded_files, raw_docs, scores), key=lambda x: x[2], reverse=True)
            with st.expander("Ver salida de consola"):
                # ---impresión del paso 5 en el navegador
                step_five_output = ["5. Calcular similitud coseno y ordenar resultados:"]
                for rank, (file, doc, score) in enumerate(ranked, 1):
                    step_five_output.append(f"\n{rank}. {os.path.basename(file.name)}")
                    step_five_output.append(f"Similitud coseno: {score:.4f}")
                step_five_output = "\n".join(step_five_output)
                st.code(step_five_output, language='plaintext')

            st.subheader("Resultados ordenados por relevancia")
            for rank, (file, doc, score) in enumerate(ranked, 1):
                st.write(f"{rank}. {os.path.basename(file.name)} - Similitud coseno: {score:.4f}")
                highlighted_doc = highlight_terms(doc, search_query)
                with st.expander(f"Ver contenido del documento"):
                    st.markdown(
                        f'<div style="max-height: 200px; overflow-y: auto; background: #f9f9f9; padding: 8px; border-radius: 4px; white-space: pre-wrap;">{highlighted_doc}</div>',
                        unsafe_allow_html=True
                    )

            # Crear DataFrame con nombres de archivo y similitud coseno
            doc_names = [os.path.basename(file.name) for file, _, _ in ranked]
            df_sim = pd.DataFrame({
                "Documento": doc_names,
                "Similitud coseno": [score for _, _, score in ranked]
            })

            df_sim["Documento"] = pd.Categorical(df_sim["Documento"], categories=doc_names, ordered=True)
            df_sim = df_sim.set_index("Documento")

            st.subheader("Gráfica de similitud coseno por documento")
            st.line_chart(df_sim)

else:       
    st.info("Sube tus archivos para comenzar la búsqueda.")