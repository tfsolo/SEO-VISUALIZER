import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from hdbscan import HDBSCAN
import google.generativeai as genai
import plotly.express as px
from umap import UMAP

# --- PAGE CONFIG ---
st.set_page_config(page_title="Advanced SEO Clusterer", layout="wide")

st.title("SEO Keyword Visualizer & Clustering Tool")
st.subheader("For Business-Driven Keyword Clustering")
st.markdown("""
This tool stacks 4 layers of data to create clusters:
1. **Semantics** (Text Meaning)
2. **Metrics** (Volume, KD, Competition)
3. **Intent** (Informational vs Transactional)
4. **SERP Features** (Video, Ads, Snippets)
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Configuration")
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API Key loaded securely! 🔒")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
    
    st.header("2. Hyperparameters")
    min_topic_size = st.slider("Min Topic Size", 5, 100, 10)
    min_samples = st.slider("Min Samples (Density)", 1, 50, 5)
    
    st.header("3. Weights")
    if st.button("Default"):
        st.session_state['w_text'] = 80
        st.session_state['w_num'] = 30
        st.session_state['w_int'] = 20
        st.session_state['w_serp'] = 20
        st.rerun() 
    
    w_text_pct = st.slider("Text Weight (%)", 0, 300, value=80, key="w_text")
    w_num_pct = st.slider("Metrics Weight (%)", 0, 300, value=30, key="w_num")
    w_int_pct = st.slider("Intent Weight (%)", 0, 300, value=20, key="w_int")
    w_serp_pct = st.slider("SERP Weight (%)", 0, 300, value=20, key="w_serp")

    w_text = w_text_pct / 100.0
    w_num = w_num_pct / 100.0
    w_int = w_int_pct / 100.0
    w_serp = w_serp_pct / 100.0
    
    run_btn = st.button("RUN ANALYSIS", type="primary")

# --- FUNCTIONS ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def process_multi_modal_embeddings(df, _model, w_text, w_num, w_int, w_serp):
    docs = df['cleaned_doc'].tolist()
    text_embeddings = _model.encode(docs, show_progress_bar=True)
    
    num_cols = ['Volume', 'Keyword Difficulty', 'Competitive Density']
    temp_df = df.copy()
    for col in num_cols:
        if col not in temp_df.columns: temp_df[col] = 0
            
    temp_df['Volume'] = temp_df['Volume'].fillna(0)
    temp_df['Keyword Difficulty'] = temp_df['Keyword Difficulty'].fillna(temp_df['Keyword Difficulty'].median())
    temp_df['Competitive Density'] = temp_df['Competitive Density'].fillna(temp_df['Competitive Density'].median())
    
    scaler = MinMaxScaler()
    scaled_num = scaler.fit_transform(temp_df[num_cols].values)
    
    if 'Intent' in temp_df.columns:
        intent_data = temp_df[['Intent']].fillna('Unknown').astype(str).values
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_intent = encoder.fit_transform(intent_data)
        null_mask = (temp_df['Intent'].isna()) | (temp_df['Intent'] == 'Unknown')
        encoded_intent[null_mask] = 0
    else:
        encoded_intent = np.zeros((len(df), 1))

    if 'SERP Features' in temp_df.columns:
        serp_data = temp_df['SERP Features'].fillna("").astype(str).tolist()
        def comma_tokenizer(text): return [x.strip() for x in text.split(',')]
        serp_vectorizer = CountVectorizer(tokenizer=comma_tokenizer, min_df=0.01)
        try:
            serp_vectors = serp_vectorizer.fit_transform(serp_data).toarray()
        except ValueError:
            serp_vectors = np.zeros((len(df), 1))
    else:
        serp_vectors = np.zeros((len(df), 1))

    combined_embeddings = np.hstack((
        text_embeddings * w_text,
        scaled_num * w_num,
        encoded_intent * w_int,
        serp_vectors * w_serp
    ))
    
    # Force float64 for compatibility
    return combined_embeddings.astype(np.float64), docs

def clean_text(text):
    text = str(text)
    text = re.sub(r'^(what is|what are|how to|why is)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(2020|2021|2022|2023|2024|2025)\b', '', text)
    return text.strip()

@st.dialog("Topic Deep Dive")
def show_topic_details(topic_name, df):
    st.write(f"### 🔍 {topic_name}")
    topic_data = df[df['Topic Label'] == topic_name]
    top_10 = topic_data.sort_values(by='Volume', ascending=False).head(10)
    
    cols_to_show = ['Keyword', 'Volume']
    cpc_col = next((c for c in df.columns if 'CPC' in c.upper()), None)
    if cpc_col: cols_to_show.append(cpc_col)
    comp_col = next((c for c in df.columns if 'Density' in c or 'Competition' in c), None)
    if comp_col: cols_to_show.append(comp_col)
    
    st.dataframe(top_10[cols_to_show], use_container_width=True, hide_index=True)

# --- MAIN LOGIC ---

uploaded_file = st.file_uploader("Upload SEMrush/Ahrefs CSV", type=['csv'])

if uploaded_file:
    # 1. RUN ANALYSIS (Only happens when button clicked)
    if run_btn:
        if not api_key:
            st.error("🔑 API Key Required to proceed.")
            st.stop()
            
        st.info("Reading Data...")
        df = pd.read_csv(uploaded_file)
        
        # Cleaning & Filtering
        if 'Keyword' not in df.columns:
            st.error("CSV must contain 'Keyword' column.")
            st.stop()
        df = df.dropna(subset=['Keyword'])
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
            df = df[df['Volume'] > 0]
        df['cleaned_doc'] = df['Keyword'].apply(clean_text)
        
        # Embeddings & Clustering
        with st.spinner("Processing..."):
            model = load_embedding_model()
            embeddings, docs = process_multi_modal_embeddings(df, model, w_text, w_num, w_int, w_serp)
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_topic_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
            topic_model = BERTopic(
                hdbscan_model=hdbscan_model,
                vectorizer_model=CountVectorizer(stop_words="english"),
                verbose=True
            )
            topics, probs = topic_model.fit_transform(docs, embeddings)
            df['Topic'] = topics

        # Labeling
        with st.spinner("AI analyzing business value..."):
            genai.configure(api_key=api_key)
            gemini = genai.GenerativeModel('gemini-1.5-flash')
            topic_info = topic_model.get_topic_info()
            labels = {}
            for index, row in topic_info.iterrows():
                tid = row['Topic']
                if tid == -1:
                    labels[tid] = "Outliers"
                    continue
                keywords = [t[0] for t in topic_model.get_topic(tid)][:10]
                rep_docs = df[df['Topic'] == tid]['Keyword'].head(5).tolist()
                prompt = f"Act as an SEO Strategist. Keywords: {', '.join(keywords)}. Queries: {', '.join(rep_docs)}. Task: Create a business topic name (max 4 words). Example Output: 'CRM Integration Pricing'"
                try:
                    response = gemini.generate_content(prompt)
                    labels[tid] = response.text.strip().replace('"', '')
                except:
                    labels[tid] = keywords[0]
                time.sleep(0.5)
            df['Topic Label'] = df['Topic'].map(labels)
            
            # --- SAVE TO SESSION STATE (The Fix) ---
            st.session_state['processed_df'] = df
            st.session_state['embeddings'] = embeddings # Save for 2D map
            st.success("Analysis Complete!")

    # 2. DISPLAY RESULTS (Checks if data exists in memory)
    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        embeddings = st.session_state['embeddings']
        
        tab1, tab2, tab3 = st.tabs(["Summary Data", "2D Map", "Download"])
        
        with tab1:
            st.subheader("Cluster Business Value")
            st.info("👆 Click on any row below to see top keywords.")

            summary = df.groupby('Topic Label').agg({
                'Keyword': 'count',
                'Volume': 'sum',
                'Keyword Difficulty': 'mean'
            }).reset_index().sort_values('Volume', ascending=False)
            
            summary.columns = ['Topic Label', 'Count', 'Total Volume', 'Avg KD']
            summary_clean = summary[~summary['Topic Label'].str.contains("Outlier|Noise", case=False, na=False)]
            
            # Interactive Table
            selection = st.dataframe(
                summary_clean, 
                use_container_width=True, 
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row"
            )

            if len(selection.selection.rows) > 0:
                selected_index = selection.selection.rows[0]
                selected_topic = summary_clean.iloc[selected_index]['Topic Label']
                show_topic_details(selected_topic, df)
            
        with tab2:
            st.subheader("Topic Landscape")
            umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
            plot_df = pd.DataFrame(umap_2d, columns=['x', 'y'])
            plot_df['Topic'] = df['Topic Label']
            plot_df['Keyword'] = df['Keyword']
            plot_df['Volume'] = df['Volume']
            fig = px.scatter(plot_df, x='x', y='y', color='Topic', hover_data=['Keyword', 'Volume'], title="Semantic & Business Metric Projection")
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "final_clusters.csv", "text/csv")