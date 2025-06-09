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
    # Define keywords for each event type
    purchase_keywords = ["buy", "bought", "purchase", "purchased", "order", "ordered"]
    return_keywords = ["return", "returned", "refund", "refunded", "exchange", "exchanged"]
    complain_keywords = ["complain", "complained", "issue", "problem", "bad", "broken", "defective", "doesn't work", "didn't work", "not working", "malfunction", "malfunctioned", "faulty"]

    def extract_events(text):
        text_lower = str(text).lower()
        events = []
        if any(word in text_lower for word in purchase_keywords):
            events.append("purchase")
        if any(word in text_lower for word in return_keywords):
            events.append("return")
        if any(word in text_lower for word in complain_keywords):
            events.append("complain")
        return events

    df['events'] = df[review_col].fillna('').apply(extract_events)
    return df

# Example relation extraction (adjusted for English)
def compute_relations(df, review_col):
    df = df.copy()
    # Define simple sentiment/action mappings
    positive_verbs = ["love", "like", "recommend", "enjoy"]
    negative_verbs = ["hate", "dislike", "return", "refund", "bad", "broken", "defective", "problem", "issue"]
    # You can expand these lists as needed

    sentiment_map = {
        "negative": negative_verbs,
        "positive": positive_verbs
    }

    def extract_relations(row):
        user = row.get('reviews.username', 'unknown')
        product = row.get('name', 'unknown')
        text = str(row.get(review_col, '')).lower()
        rels = []
        for verb in positive_verbs:
            if verb in text:
                rels.append((user, verb, product))
        for verb in negative_verbs:
            if verb in text:
                rels.append((user, verb, product))
        return rels

    df['relations'] = df.apply(extract_relations, axis=1)
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

    from rdflib import Graph, Namespace, Literal, URIRef
    import urllib.parse

    g = Graph()
    EX = Namespace('http://example.org/')

    for _, row in df.iterrows():
        user = str(row['reviews.username']) if not pd.isna(row['reviews.username']) else 'unknown'
        # Use relations extracted for this review
        relations = row.get('relations', [])
        # If relations is a string (from CSV), try to eval to list
        if isinstance(relations, str):
            try:
                import ast
                relations = ast.literal_eval(relations)
            except Exception:
                relations = []
        for subj, pred, obj in relations:
            # Encode all parts for URI safety
            subj_uri = EX[urllib.parse.quote(str(subj), safe='')]
            pred_uri = EX[urllib.parse.quote(str(pred), safe='')]
            obj_uri = EX[urllib.parse.quote(str(obj), safe='')]
            g.add((subj_uri, pred_uri, obj_uri))
        # Optionally, also link the user to the review/product as before
        # user_uri = EX[urllib.parse.quote(user, safe='')]
        # pid = str(row['id']) if not pd.isna(row['id']) else 'unknown'
        # pid_uri = EX[urllib.parse.quote(pid, safe='')]
        # g.add((user_uri, EX.rated, pid_uri))
        # g.add((pid_uri, EX.rating,  Literal(row['reviews.rating'] if not pd.isna(row['reviews.rating']) else '')))
        # g.add((pid_uri, EX.summary, Literal(row['reviews.title'] if not pd.isna(row['reviews.title']) else '')))

    st.success(f"Graph with {len(g)} triples.")

    # Pyvis visualization
    def visualize_rdf_pyvis(graph: Graph):
        net = Network(height="500px", width="100%", notebook=False, directed=True)
        for s, p, o in graph:
            s_label = str(s).split('/')[-1]
            o_label = str(o).split('/')[-1] if str(o).startswith('http') else str(o)
            p_label = str(p).split('/')[-1]
            net.add_node(s_label, label=s_label)
            net.add_node(o_label, label=o_label)
            net.add_edge(s_label, o_label, label=p_label)
        net.repulsion(node_distance=120, central_gravity=0.33)
        net.save_graph("rdf_graph.html")
        with open("rdf_graph.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=550, scrolling=True)

    visualize_rdf_pyvis(g)

# Step 7: BM25 Search (Semantic)
st.header("7. Búsqueda BM25 semántica con ranking")

# Define sentiment synonyms for query expansion
SENTIMENT_SYNONYMS = {
    "negative": ["hate", "dislike", "return", "refund", "bad", "broken", "defective", "problem", "issue", "complain"],
    "positive": ["love", "like", "recommend", "enjoy"]
}

# Build a semantic corpus: review text + events + relation verbs
def build_semantic_corpus(df, review_col):
    corpus = []
    for _, row in df.iterrows():
        text = str(row.get(review_col, ''))
        events = " ".join(row.get('events', []))
        # Only use the predicate/verb from relations
        relations = row.get('relations', [])
        if isinstance(relations, str):
            try:
                import ast
                relations = ast.literal_eval(relations)
            except Exception:
                relations = []
        rel_verbs = " ".join([str(r[1]) for r in relations])
        combined = f"{text} {events} {rel_verbs}"
        corpus.append(combined.split())
    return corpus

corpus = build_semantic_corpus(df, review_col)
bm25 = init_bm25(corpus)
q = st.text_input("Query:")
if q:
    # Expand query with sentiment synonyms
    tokens = []
    for w in q.split():
        tokens.extend(SENTIMENT_SYNONYMS.get(w.lower(), [w.lower()]))
    tokens = [w for w in tokens if w not in stopwords_set]
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
        # Highlight original query tokens in review_text
        highlighted_text = review_text
        for token in q.split():
            if token.strip() == "":
                continue
            highlighted_text = re.sub(
                rf'({re.escape(token)})',
                r'<mark>\1</mark>',
                highlighted_text,
                flags=re.IGNORECASE
            )
        st.markdown(f"**Product:** {df.iloc[idx].get('name', '')} (Rating: {rating}) Score: {scores[idx]:.2f}")
        st.markdown(f"**User:** {username}")
        st.markdown(f"**Title:** {df.iloc[idx].get('reviews.title', '')}")
        with st.expander("Show review text"):
            st.markdown(highlighted_text, unsafe_allow_html=True)
        shown += 1
        i += 1