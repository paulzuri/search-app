import streamlit as st
import pandas as pd
import re
import spacy
from nltk import download
from nltk.corpus import stopwords
from rdflib import Graph, Namespace, Literal
from rank_bm25 import BM25Okapi
import graphviz
import urllib.parse
from pyvis.network import Network
import streamlit.components.v1 as components

# Download and set up resources
download('stopwords')
english_stopwords = set(stopwords.words('english'))
stopwords_set = english_stopwords

# Load spaCy English model
def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        from spacy.cli.download import download as spacy_download
        spacy_download('en_core_web_sm')
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# Load dataset from local CSV
@st.cache_data
def load_dataset():
    df = pd.read_csv('amazon_reviews.csv')
    df.columns = [c.strip() for c in df.columns]
    st.write("Columns in CSV:", list(df.columns))  # Show columns for debugging
    return df

# Helper to get the review text column name
def get_review_text_column(df):
    for col in ['reviews.text', 'reviewText', 'review_text', 'text', 'review', 'body']:
        if col in df.columns:
            return col
    st.error("No review text column found in CSV!")
    st.stop()

# Extract price and model from columns (not from reviewText)
def extract_price_model(df):
    df = df.copy()
    df['price'] = df.get('prices', None)
    df['brand'] = df.get('brand', None)
    return df

# NER for key entities in reviewText
def compute_ner(df, review_col):
    df = df.copy()
    df['entities'] = df[review_col].fillna('').apply(lambda t: [(ent.text, ent.label_) for ent in nlp(str(t)).ents])
    return df

# Extract events based on English verbs using NLTK WordNet
def compute_events(df, review_col):
    df = df.copy()
    def extract_verbs(text):
        doc = nlp(str(text))
        return [token.lemma_ for token in doc if token.pos_ == "VERB"]
    df['events'] = df[review_col].fillna('').apply(extract_verbs)
    return df

# Example relation extraction (adjusted for English)
def compute_relations(df, review_col):
    df = df.copy()
    def extract_rel(t):
        rels = []
        t = str(t)
        # Example: "X for Y"
        for m in re.finditer(r'(\w+)\s+for\s+(\w+)', t):
            rels.append((m.group(1), 'for', m.group(2)))
        # Example: "X is Y"
        for m in re.finditer(r'(\w+)\s+is\s+(\w+)', t):
            rels.append((m.group(1), 'is', m.group(2)))
        return rels
    df['relations'] = df[review_col].fillna('').apply(extract_rel)
    return df

# Build RDF graph
def build_rdf(df):
    g = Graph()
    EX = Namespace('http://example.org/')
    for _, row in df.iterrows():
        # Safely encode username and id for URI
        user_val = str(row.get('reviews.username', 'unknown'))
        user_uri = EX[urllib.parse.quote(user_val, safe='')]
        pid_val = str(row.get('id', 'unknown'))
        pid_uri = EX[urllib.parse.quote(pid_val, safe='')]
        g.add((user_uri, EX.rated, pid_uri))
        g.add((pid_uri, EX.rating, Literal(row.get('reviews.rating', ''))))
        g.add((pid_uri, EX.summary, Literal(row.get('reviews.title', ''))))
    return g

# Initialize BM25
def init_bm25(corpus):
    return BM25Okapi(corpus)

# Sidebar
title = "Análisis semántico con reviews de Amazon"
st.sidebar.title(title)
st.sidebar.markdown("Gavilanes, E., Zurita, M.")

# Step 1: Load dataset
st.header("1. Carga del dataset de reviews de Amazon")
df = load_dataset()
st.success(f"Loaded {len(df)} reviews with {df.shape[1]} columns.")

# Find the review text column
review_col = get_review_text_column(df)

# Step 2: Extract price and model
st.header("2. Extracción de datos mediante expresiones regulares")
with st.spinner("Extracting price and model..."):
    df = extract_price_model(df)
show_cols = [c for c in ['name', 'price', 'brand'] if c in df.columns or c in ['price', 'brand']]
st.write(df[show_cols].head())

# Step 3: NER
st.header("3. Named Entity Recognition (NER) mediante spaCy")
with st.spinner("Detecting entities..."):
    df = compute_ner(df, review_col)
st.write(df[[review_col, 'entities']].head())

# Step 4: Event extraction
st.header("4. Extracción de eventos mediante lista de verbos creada con spaCy")
with st.spinner("Extracting key events..."):
    df = compute_events(df, review_col)
st.write(df[[review_col, 'events']].head())

# Step 5: Relation extraction
st.header("5. Extracción de relaciones mediante expresiones regulares")
with st.spinner("Extracting relations..."):
    df = compute_relations(df, review_col)
st.write(df[[review_col, 'relations']].head())

# Step 6: RDF Construction
st.header("6. Construcción de grafo RDF mediante rdflib")
with st.spinner("Building RDF graph..."):

    # your real columns
    user_col   = 'reviews.username'
    id_col     = 'id'
    rating_col = 'reviews.rating'
    title_col  = 'reviews.title'

    # build the rdflib graph
    from rdflib import Graph, Namespace, Literal, URIRef
    import urllib.parse

    g = Graph()
    EX = Namespace('http://example.org/')

    for _, row in df.iterrows():
        u   = str(row[user_col]) if not pd.isna(row[user_col]) else 'unknown'
        pid = str(row[id_col])       if not pd.isna(row[id_col])   else 'unknown'

        uri_user = EX[urllib.parse.quote(u,   safe='')]
        uri_prod = EX[urllib.parse.quote(pid, safe='')]

        g.add((uri_user, EX.rated,   uri_prod))
        g.add((uri_prod, EX.rating,  Literal(row[rating_col]  if not pd.isna(row[rating_col]) else '')))
        g.add((uri_prod, EX.summary, Literal(row[title_col]   if not pd.isna(row[title_col])  else '')))

    st.success(f"Graph with {len(g)} triples.")

    # visualize with graphviz
    import graphviz
    def visualize_rdf(graph: Graph) -> graphviz.Digraph:
        dot = graphviz.Digraph()
        for s, p, o in graph:
            # node names: just the local part after '/'
            s_label = str(s).split('/')[-1]
            if isinstance(o, URIRef):
                o_label = str(o).split('/')[-1]
            else:
                o_label = str(o)
            # add nodes & edge
            dot.node(s_label, s_label)
            dot.node(o_label, o_label)
            dot.edge(s_label, o_label, label=str(p).split('/')[-1])
        return dot

    dot = visualize_rdf(g)
    # pass the Digraph object directly
    st.graphviz_chart(dot)

    # Pyvis visualization
    def visualize_rdf_pyvis(graph: Graph):
        net = Network(height="500px", width="100%", notebook=False, directed=True)
        for s, p, o in graph:
            s_label = str(s).split('/')[-1]
            o_label = str(o).split('/')[-1] if str(o).startswith('http') else str(o)
            net.add_node(s_label, label=s_label)
            net.add_node(o_label, label=o_label)
            net.add_edge(s_label, o_label, label=str(p).split('/')[-1])
        net.repulsion(node_distance=120, central_gravity=0.33)
        net.save_graph("rdf_graph.html")
        with open("rdf_graph.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=550, scrolling=True)

    visualize_rdf_pyvis(g)

# Step 7: BM25 Search
st.header("7. Búsqueda BM25 con ranking")
corpus = [str(t).split() for t in df[review_col].fillna('')]
bm25 = init_bm25(corpus)
q = st.text_input("Query:")
if q:
    tokens = [w for w in q.split() if w.lower() not in stopwords_set]
    scores = bm25.get_scores(tokens)
    import numpy as np
    top = np.argsort(scores)[::-1]
    shown = 0
    i = 0
    while shown < 5 and i < len(top):
        idx = top[i]
        if scores[idx] == 0:
            i += 1
            continue  # Skip reviews with score 0
        review_text = df.iloc[idx].get(review_col, '')
        rating = df.iloc[idx].get('reviews.rating', '')
        if pd.isna(review_text) and pd.isna(rating):
            i += 1
            continue  # skip if both are NaN
        if pd.isna(review_text):
            review_text = "[No review text]"
        if pd.isna(rating):
            rating = "N/A"
        username = df.iloc[idx].get('reviews.username', '')
        # Highlight query tokens in review_text
        highlighted_text = review_text
        for token in tokens:
            if token.strip() == "":
                continue
            # Use regex for case-insensitive replacement, word boundaries
            highlighted_text = re.sub(
                rf'({re.escape(token)})',
                r'<mark>\1</mark>',
                highlighted_text,
                flags=re.IGNORECASE
            )
        st.markdown(f"**ID:** {df.iloc[idx].get('id', '')} (Rating: {rating}) Score: {scores[idx]:.2f}")
        st.markdown(f"**User:** {username}")
        st.markdown(f"**Title:** {df.iloc[idx].get('reviews.title', '')}")
        with st.expander("Show review text"):
            st.markdown(highlighted_text, unsafe_allow_html=True)
        shown += 1
        i += 1
