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

st.title("SEO Keywrod Visializer & Clustering Tool")
st.subheader("For Business-Driven Keyword Clustering")
st.markdown("""
This tool stacks 4 layers of data to create clusters:
1. **Semantics** (Text Meaning)
2. **Metrics** (Volume, KD, Competition)
3. **Intent** (Informational vs Transactional)
4. **SERP Features** (Video, Ads, Snippets)
""")

# --- SIDEBAR: ALL YOUR TUNING PARAMETERS ---
with st.sidebar:
    st.header("1. Configuration")
    
    # Check if the key is stored in Streamlit Secrets
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API Key loaded securely from Secrets! 🔒")
    else:
        # If not found, ask the user to paste it
        api_key = st.text_input("Gemini API Key", type="password")
    
    st.header("2. Hyperparameters (The Tuning)")
    # Replaces your "Auto-Loop" with real-time manual control
    min_topic_size = st.slider("Min Topic Size", 5, 100, 10)
    min_samples = st.slider("Min Samples (Density)", 1, 50, 5)
    
    st.header("3. Weights")
    
    # --- RESET BUTTON LOGIC ---
    if st.button("Default"):
        # Reset to percentage values (e.g., 90 = 0.9)
        st.session_state['w_text'] = 80
        st.session_state['w_num'] = 30
        st.session_state['w_int'] = 20
        st.session_state['w_serp'] = 20
        st.rerun() 
    
    # Sliders now use integers (0-300)
    w_text_pct = st.slider("Text Weight (%)", 0, 300, value=80, key="w_text")
    w_num_pct = st.slider("Metrics Weight (%)", 0, 300, value=30, key="w_num")
    w_int_pct = st.slider("Intent Weight (%)", 0, 300, value=20, key="w_int")
    w_serp_pct = st.slider("SERP Weight (%)", 0, 300, value=20, key="w_serp")

    # Convert back to decimals for the math
    w_text = w_text_pct / 100.0
    w_num = w_num_pct / 100.0
    w_int = w_int_pct / 100.0
    w_serp = w_serp_pct / 100.0
    
    run_btn = st.button("RUN ANALYSIS", type="primary")

# --- CACHED FUNCTIONS ---

@st.cache_resource
def load_embedding_model():
    # We stick to MiniLM for speed. If you want EXACT original results, change to 'thenlper/gte-base'
    # but be warned: it will crash 500MB+ RAM limits on free cloud hosting.
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def process_multi_modal_embeddings(df, _model, w_text, w_num, w_int, w_serp):
    """
    This function EXACTLY replicates your 'Part 5' Stacking Logic.
    """
    docs = df['cleaned_doc'].tolist()
    
    # --- LAYER 1: TEXT ---
    text_embeddings = _model.encode(docs, show_progress_bar=True)
    
    # --- LAYER 2: NUMERICAL ---
    # Ensure cols exist
    num_cols = ['Volume', 'Keyword Difficulty', 'Competitive Density']
    temp_df = df.copy()
    for col in num_cols:
        if col not in temp_df.columns:
            temp_df[col] = 0
            
    # Impute (Median Strategy)
    temp_df['Volume'] = temp_df['Volume'].fillna(0)
    temp_df['Keyword Difficulty'] = temp_df['Keyword Difficulty'].fillna(temp_df['Keyword Difficulty'].median())
    temp_df['Competitive Density'] = temp_df['Competitive Density'].fillna(temp_df['Competitive Density'].median())
    
    scaler = MinMaxScaler()
    scaled_num = scaler.fit_transform(temp_df[num_cols].values)
    
    # --- LAYER 3: INTENT (Revised for Pure Zero Impact) ---
    if 'Intent' in temp_df.columns:
        # 1. Fill NaNs with a placeholder temporarily
        intent_data = temp_df[['Intent']].fillna('Unknown').astype(str).values
        
        # 2. Encode everything
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_intent = encoder.fit_transform(intent_data)
        
        # 3. FORCE ZERO VECTORS for the original Nulls
        # Find rows where the original Intent was NaN or "Unknown"
        null_mask = (temp_df['Intent'].isna()) | (temp_df['Intent'] == 'Unknown')
        
        # Multiply those rows by 0 to wipe out their influence
        encoded_intent[null_mask] = 0
    else:
        encoded_intent = np.zeros((len(df), 1))

    # --- LAYER 4: SERP FEATURES (Added Back In!) ---
    if 'SERP Features' in temp_df.columns:
        serp_data = temp_df['SERP Features'].fillna("").astype(str).tolist()
        def comma_tokenizer(text):
            return [x.strip() for x in text.split(',')]
        
        # min_df=0.01 matches your script logic
        serp_vectorizer = CountVectorizer(tokenizer=comma_tokenizer, min_df=0.01)
        # Check if vocabulary is empty (e.g. if column is all empty)
        try:
            serp_vectors = serp_vectorizer.fit_transform(serp_data).toarray()
        except ValueError:
            # Fallback if vocab is empty
            serp_vectors = np.zeros((len(df), 1))
    else:
        serp_vectors = np.zeros((len(df), 1))

    # --- WEIGHTED STACKING ---
    # Combine all layers
    combined_embeddings = np.hstack((
        text_embeddings * w_text,
        scaled_num * w_num,
        encoded_intent * w_int,
        serp_vectors * w_serp
    ))
    
    return combined_embeddings, docs

def clean_text(text):
    text = str(text)
    text = re.sub(r'^(what is|what are|how to|why is)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(2020|2021|2022|2023|2024|2025)\b', '', text)
    return text.strip()

# --- POP-UP FUNCTION ---
@st.dialog("Topic Deep Dive")
def show_topic_details(topic_name, df):
    st.write(f"### 🔍 {topic_name}")
    
    # Filter data for this topic
    topic_data = df[df['Topic Label'] == topic_name]
    
    # Sort by Volume to get Top 10
    top_10 = topic_data.sort_values(by='Volume', ascending=False).head(10)
    
    # Dynamic Column Finder (Finds CPC/Density even if names vary)
    cols_to_show = ['Keyword', 'Volume']
    
    # Check for CPC
    cpc_col = next((c for c in df.columns if 'CPC' in c.upper()), None)
    if cpc_col: cols_to_show.append(cpc_col)
        
    # Check for Density
    comp_col = next((c for c in df.columns if 'Density' in c or 'Competition' in c), None)
    if comp_col: cols_to_show.append(comp_col)
    
    # Display the table
    st.dataframe(
        top_10[cols_to_show], 
        use_container_width=True, 
        hide_index=True
    )


# --- MAIN APP ---

uploaded_file = st.file_uploader("Upload SEMrush/Ahrefs CSV", type=['csv'])

if uploaded_file and run_btn:
    if not api_key:
        st.error("🔑 API Key Required to proceed.")
        st.stop()
        
    st.info("Reading Data...")
    df = pd.read_csv(uploaded_file)
    
    # 1. Filter & Clean
    if 'Keyword' not in df.columns:
        st.error("CSV must contain 'Keyword' column.")
        st.stop()
        
    df = df.dropna(subset=['Keyword'])
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
        df = df[df['Volume'] > 0]
        
    df['cleaned_doc'] = df['Keyword'].apply(clean_text)
    
    # 2. Generate SUPER VECTORS (Cached)
    with st.spinner("Generating Multi-Modal Embeddings..."):
        model = load_embedding_model()
        # Passing all your weights into the processor
        embeddings, docs = process_multi_modal_embeddings(df, model, w_text, w_num, w_int, w_serp)
        
    # 3. Clustering
    with st.spinner("Running HDBSCAN & BERTopic..."):
        # We use your exact HDBSCAN setup
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=min_samples, # Added this parameter back
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        topic_model = BERTopic(
            hdbscan_model=hdbscan_model,
            vectorizer_model=CountVectorizer(stop_words="english"),
            verbose=True
        )
        
        # Fit with the custom embeddings
        topics, probs = topic_model.fit_transform(docs, embeddings)
        df['Topic'] = topics

    # 4. Labeling (Optimized Single Pass)
    with st.spinner("AI analyzing business value..."):
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        topic_info = topic_model.get_topic_info()
        labels = {}
        
        # Limit to top 50 topics to save time/cost if dataset is huge
        # You can remove [:50] if you want to label everything
        for index, row in topic_info.iterrows():
            tid = row['Topic']
            if tid == -1:
                labels[tid] = "Outliers"
                continue
                
            # Get keywords and representative docs
            keywords = [t[0] for t in topic_model.get_topic(tid)][:10]
            rep_docs = df[df['Topic'] == tid]['Keyword'].head(5).tolist()
            
            prompt = f"""
            Act as an SEO Strategist. 
            Keywords: {", ".join(keywords)}
            Queries: {", ".join(rep_docs)}
            
            Task: Create a business topic name (max 4 words). 
            Example Output: "CRM Integration Pricing"
            """
            
            try:
                # Simple retry logic
                response = gemini.generate_content(prompt)
                labels[tid] = response.text.strip().replace('"', '')
            except:
                labels[tid] = keywords[0] # Fallback
            
            # Tiny sleep to prevent rate limits
            time.sleep(0.5)
            
        df['Topic Label'] = df['Topic'].map(labels)

    # 5. Output
    st.success("Analysis Complete!")
    
    # TABS FOR VISUALIZATION
    tab1, tab2, tab3 = st.tabs(["Summary Data", "2D Map", "Download"])
    
    with tab1:
        st.subheader("Cluster Business Value")
        st.info("👆 Click on any row below to see top keywords.")

        # 1. Create the summary table
        summary = df.groupby('Topic Label').agg({
            'Keyword': 'count',
            'Volume': 'sum',
            'Keyword Difficulty': 'mean'
        }).reset_index().sort_values('Volume', ascending=False)
        
        # 2. Rename columns
        summary.columns = ['Topic Label', 'Count', 'Total Volume', 'Avg KD']
        
        # 3. HIDE OUTLIERS (The Fix)
        # We filter out rows where Topic Label contains "Outlier" or "Noise"
        summary_clean = summary[~summary['Topic Label'].str.contains("Outlier|Noise", case=False, na=False)]
        
        # 4. Display Interactive Table
        # on_select="rerun" makes the app reload instantly when clicked
        selection = st.dataframe(
            summary_clean, 
            use_container_width=True, 
            hide_index=True,
            on_select="rerun",  # <--- This enables the click action
            selection_mode="single-row"
        )

        # 5. Handle the Click
        if len(selection.selection.rows) > 0:
            # Get the index of the clicked row
            selected_index = selection.selection.rows[0]
            # Get the Topic Name from that row
            selected_topic = summary_clean.iloc[selected_index]['Topic Label']
            
            # Launch the Pop-up
            show_topic_details(selected_topic, df)
        
    with tab2:
        st.subheader("Topic Landscape")
        # UMAP for 2D projection
        umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
        
        plot_df = pd.DataFrame(umap_2d, columns=['x', 'y'])
        plot_df['Topic'] = df['Topic Label']
        plot_df['Keyword'] = df['Keyword']
        plot_df['Volume'] = df['Volume']
        
        fig = px.scatter(
            plot_df, x='x', y='y', color='Topic',
            hover_data=['Keyword', 'Volume'],
            title="Semantic & Business Metric Projection"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "final_clusters.csv", "text/csv")