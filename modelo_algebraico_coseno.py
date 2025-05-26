# modelo algebraico (similitud coseno)

# objetivo del programa: 
# 1. leer tres documentos (docx o txt) y guardar las palabras en listas.
# 2. eliminar las stopwords y conservar solo palabras relevantes.
# 3. crear un vocabulario único con todas las palabras de los documentos y la consulta.
# 4. calcular la frecuencia de cada palabra en cada documento (vector binario).
# 5. calcular la similitud coseno entre la consulta y cada documento.
# 6. ordenar los documentos según la similitud y mostrar los resultados.
# 7. resaltar los términos de búsqueda en los documentos.

# importación de librerías

import os
import re
import numpy as np
import unicodedata
from nltk.corpus import stopwords
from nltk import download
from docx import Document

# descargar las stopwords de nltk si no están presentes
download('stopwords')
spanish_stopwords = set(stopwords.words('spanish'))

# rutas de archivos y consulta
file_paths = ['.\\docs\\doc1.docx', '.\\docs\\doc2.docx', '.\\docs\\doc3.docx']  # Rutas de los documentos
search_query = "machine learning"  # Consulta de búsqueda

# funciones de procesamiento de texto
def load_text(file_path):
    """Carga el texto de un archivo .txt o .docx"""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
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
    return [word for word in tokens if word not in spanish_stopwords and word.isalpha()]

def cosine_similarity(vec1, vec2):
    """Calcula la similitud coseno entre dos vectores"""
    if not np.any(vec1) or not np.any(vec2):
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def vectorize(words, vocab):
    """Convierte una lista de palabras en un vector binario según el vocabulario"""
    return np.array([1 if word in words else 0 for word in vocab], dtype=int)

# --- Procesamiento de documentos ---
def main():
    print()
    # Cargar el texto original de cada documento
    raw_docs = [load_text(path) for path in file_paths]   

    # ---impresión del paso 1 en consola
    print("1. Guardar los documentos en un array raw_docs de 3 celdas:")
    for i, doc in enumerate(raw_docs, 1):
        words = doc.split()
        print(f"Documento {i} (raw_docs[{i-1}]): {len(words)} palabras. Primeras 5 palabras: {words[:5]}")
    print("NOTA: raw_docs contiene el texto completo de cada documento para cada posición, se ha tokenizado el print para mostrar las primeras 5 palabras.\n")
    # ---fin del print

    # Limpiar y tokenizar cada documento (eliminando stopwords)
    clean_docs = [clean_words(text) for text in raw_docs]
    
    # ---impresión del paso 2 en consola
    print("2. Limpieza y tokenizado de los documentos crudos en un array de 2 dimensiones:")
    for i, doc in enumerate(clean_docs, 1):
        print(f"Documento {i} (clean_docs[{i-1}][0 ... 4]): {len(doc)} palabras. Primeras 5 palabras: {doc[:5]}")
    
    # ---fin del print

    # Limpiar y tokenizar la consulta de búsqueda
    clean_query = clean_words(search_query)
    print(f"search_query: '{search_query}'; clean_query: {clean_query} (L= {len(clean_query)})")
    print("")

    # --- Crear vocabulario único ---
    # Unir todas las palabras de los documentos y la consulta, y ordenarlas
    vocab = sorted(set(clean_query + sum(clean_docs, [])))

    # ---impresión del paso 3 en consola
    print(f"3. Creación del vocabulario único:")
    print(f"5 primeras palabras de vocab: {vocab[:5]}, L = {len(vocab)} palabras\n")
    # ---fin del print

    # --- Crear matriz de frecuencia binaria ---
    # Vector de la consulta
    query_vector = vectorize(clean_query, vocab)
    # Vectores de cada documento
    doc_vectors = [vectorize(doc, vocab) for doc in clean_docs]

    # ---impresión del paso 4 en consola
    print(f"4. Creación del vectores binarios para documentos y query:")
    print(f"Vector de la consulta (query_vector): {query_vector} (L = {len(query_vector)} palabras)")

    indices = np.where(query_vector == 1)[0]
    print("Índices con 1 en query vector:", indices)
    print("Aciertos:", len(indices))
    print("Corresponde a las palabras de vocab:", [vocab[i] for i in indices], "\n")

    for i, doc_vector in enumerate(doc_vectors, 1):
        print(f"Vector del Documento {i} (doc_vectors[{i-1}]): {doc_vector} (L = {len(doc_vector)} palabras)")       
        indices = np.where(doc_vector == 1)[0]
        print(f"Índices con 1 en doc_vectors[{i-1}]:", indices)
        print("Aciertos:", len(indices))
        print("Corresponde a las 5 primeras palabras de vocab:", [vocab[i] for i in indices[:5]], "\n")
    # ---fin del print

    # --- Calcular similitud coseno ---
    scores = [cosine_similarity(query_vector, doc_vector) for doc_vector in doc_vectors]

    # --- Ordenar y mostrar resultados ---
    # Ordenar documentos por similitud de mayor a menor
    ranked = sorted(zip(file_paths, raw_docs, scores), key=lambda x: x[2], reverse=True)
    
    # ---impresión del paso 5 en consola
    print("5. Resultados ordenados por similitud:\n")
    for rank, (path, doc, score) in enumerate(ranked, 1):
        print(f"{rank}. {os.path.basename(path)}")
        print(f"Similitud coseno: {score:.4f}")
    # --fin del print
    print()
        
if __name__ == '__main__':
    main()