import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Config
st.set_page_config(page_title="K-Means Student Clustering", page_icon="🎓", layout="wide")
st.title("🎓 K-Means Clustering per Analisi Studenti")

# Data generation
@st.cache_data
def generate_data(n=1000):
    np.random.seed(42)
    data = {
        'student_id': range(1, n + 1),
        'anno_iscrizione': np.random.choice([2020, 2021, 2022, 2023, 2024], n),
        'eta': np.clip(np.random.normal(22, 2.5, n), 18, 35).astype(int),
        'ore_studio_settimanali': np.clip(np.random.gamma(2, 8, n), 5, 60),
        'numero_esami_superati': np.clip(np.random.poisson(12, n), 0, 30),
        'media_voti': np.clip(np.random.normal(24, 3, n), 18, 30),
        'partecipazione_eventi': np.random.choice([0, 1], n),
        'lavoro_part_time': np.random.choice([0, 1], n),
        'carriera_in_corso': np.random.choice([0, 1], n)
    }
    df = pd.DataFrame(data)
    # Add some missing values
    missing_idx = np.random.choice(n, int(0.05 * n), replace=False)
    df.loc[missing_idx[:len(missing_idx)//2], 'anno_iscrizione'] = None
    return df

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configurazioni")
    
    # Data source
    use_sample = st.checkbox("Usa dati esempio", True)
    if use_sample:
        n_samples = st.slider("N. studenti", 100, 2000, 1000, 100)
        df = generate_data(n_samples)
    else:
        file = st.file_uploader("Carica CSV", type=['csv'])
        if file:
            df = pd.read_csv(file)
        else:
            st.stop()
    
    # Feature selection
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'student_id']
    features = st.multiselect("Features", numeric_cols, default=numeric_cols[:5])
    if not features:
        st.error("Seleziona almeno una feature!")
        st.stop()
    
    # Algorithm params
    n_clusters = st.slider("N. cluster", 2, 10, 4)
    init_method = st.selectbox("Inizializzazione", ["k-means++", "random"])
    scaling = st.selectbox("Normalizzazione", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
    imputation = st.selectbox("Valori mancanti", ["mean", "median", "most_frequent", "knn", "drop"])

# Preprocessing
@st.cache_data
def preprocess(df, features, imputation, scaling):
    df_proc = df[features].copy()
    
    # Handle missing values
    if imputation == "drop":
        df_proc = df_proc.dropna()
    else:
        imputer = KNNImputer(n_neighbors=5) if imputation == "knn" else SimpleImputer(strategy=imputation)
        df_proc = pd.DataFrame(imputer.fit_transform(df_proc), columns=features, index=df_proc.index)
    
    # Scaling
    if scaling != "None":
        scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}[scaling]
        df_proc = pd.DataFrame(scaler.fit_transform(df_proc), columns=features, index=df_proc.index)
    
    return df.loc[df_proc.index], df_proc

df_orig, df_proc = preprocess(df, features, imputation, scaling)
st.success(f"✅ {len(df_proc)} studenti, {len(features)} features processate")

# EDA
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Statistiche")
    st.dataframe(df_proc.describe())
with col2:
    st.subheader("🔗 Correlazioni")
    fig_corr = px.imshow(df_proc.corr(), text_auto=True, title="Matrice Correlazione")
    st.plotly_chart(fig_corr, use_container_width=True)

# Elbow analysis
@st.cache_data
def elbow_analysis(data, max_k=8):
    sse, sil_scores = [], []
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init=init_method, random_state=42)
        labels = kmeans.fit_predict(data)
        sse.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(data, labels))
    return k_range, sse, sil_scores

if st.button("🔍 Analisi Elbow"):
    k_range, sse, sil_scores = elbow_analysis(df_proc)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = px.line(x=list(k_range), y=sse, markers=True, title="Elbow Method", 
                           labels={'x': 'K', 'y': 'SSE'})
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col2:
        fig_sil = px.line(x=list(k_range), y=sil_scores, markers=True, title="Silhouette Score",
                         labels={'x': 'K', 'y': 'Silhouette'})
        st.plotly_chart(fig_sil, use_container_width=True)
    
    best_k = k_range[np.argmax(sil_scores)]
    st.info(f"💡 K ottimale secondo Silhouette: {best_k}")

# Clustering
if st.button("▶️ Esegui Clustering", type="primary"):
    with st.spinner("Clustering in corso..."):
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, random_state=42)
        labels = kmeans.fit_predict(df_proc)
        
        # Results
        df_results = df_orig.copy()
        df_results['cluster'] = labels
        df_results['distance'] = np.min(kmeans.transform(df_proc), axis=1)
        df_results['quality'] = pd.cut(df_results['distance'], 
                                     bins=[0, np.percentile(df_results['distance'], 33),
                                           np.percentile(df_results['distance'], 66), np.inf],
                                     labels=['Tipico', 'Moderato', 'Atipico'])
        
        sil_score = silhouette_score(df_proc, labels)
        
        # Store in session
        st.session_state.update({
            'results': df_results, 'processed': df_proc, 'kmeans': kmeans,
            'features': features, 'silhouette': sil_score
        })
        
        st.success("✅ Clustering completato!")

# Results visualization
if 'results' in st.session_state:
    df_results = st.session_state['results']
    kmeans = st.session_state['kmeans']
    sil_score = st.session_state['silhouette']
    
    st.header("📈 Risultati")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{sil_score:.3f}")
    col2.metric("N. Cluster", n_clusters)
    col3.metric("N. Studenti", len(df_results))
    
    # Distribution
    cluster_counts = df_results['cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(values=cluster_counts.values, 
                        names=[f"Cluster {i}" for i in cluster_counts.index],
                        title="Distribuzione Cluster")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(x=[f"Cluster {i}" for i in cluster_counts.index], 
                        y=cluster_counts.values, title="Studenti per Cluster")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # PCA visualization
    if len(features) > 2:
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(df_proc)
        
        fig_pca = px.scatter(x=pca_data[:, 0], y=pca_data[:, 1],
                           color=[f"Cluster {i}" for i in df_results['cluster']],
                           title=f"PCA Clustering (Var: {pca.explained_variance_ratio_.sum():.1%})")
        
        # Add centroids
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        fig_pca.add_scatter(x=centroids_pca[:, 0], y=centroids_pca[:, 1],
                          mode='markers', marker=dict(symbol='x', size=15, color='black'),
                          name='Centroidi')
        st.plotly_chart(fig_pca, use_container_width=True)
    
    # Centroids table
    st.subheader("📋 Centroidi")
    centroids_df = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=features,
                               index=[f"Cluster {i}" for i in range(n_clusters)]).round(3)
    st.dataframe(centroids_df, use_container_width=True)
    
    # Quality distribution
    quality_dist = df_results.groupby(['cluster', 'quality']).size().unstack(fill_value=0)
    fig_quality = px.bar(quality_dist, title="Qualità per Cluster", barmode='stack')
    st.plotly_chart(fig_quality, use_container_width=True)
    
    # Results table
    st.subheader("📄 Risultati Dettagliati")
    show_clusters = st.multiselect("Mostra cluster", sorted(df_results['cluster'].unique()), 
                                  default=sorted(df_results['cluster'].unique()))
    
    filtered_results = df_results[df_results['cluster'].isin(show_clusters)].head(100)
    st.dataframe(filtered_results, use_container_width=True)
    
    # Download
    csv = df_results.to_csv(index=False)
    st.download_button("📁 Scarica CSV", csv, "clustering_results.csv", "text/csv")

# Educational section
with st.expander("📚 Come Funziona K-Means"):
    st.markdown("""
    ## 🎯 Algoritmo K-Means
    
    **K-Means** raggruppa studenti simili in k cluster basandosi sulle features selezionate.
    
    ### 🔄 Processo:
    1. **Inizializzazione**: Posiziona k centroidi casuali
    2. **Assegnazione**: Ogni studente va al centroide più vicino  
    3. **Aggiornamento**: I centroidi si spostano al centro del loro cluster
    4. **Iterazione**: Ripete 2-3 fino a convergenza
    
    ### 📊 Preprocessing:
    - **Normalizzazione**: Scala le features per evitare bias
    - **Valori Mancanti**: Sostituiti con media/mediana/KNN
    
    ### 📈 Metriche:
    - **Silhouette Score**: Qualità clustering (-1 a 1, più alto = meglio)
    - **Elbow Method**: Trova k ottimale guardando dove il miglioramento rallenta
    
    ### 💡 Interpretazione:
    Ogni cluster rappresenta un "tipo" di studente con caratteristiche simili.
    La distanza dal centroide indica quanto lo studente è "tipico" del suo gruppo.
    """)
