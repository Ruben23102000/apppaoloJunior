import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Configurazione
st.set_page_config(page_title="K-Means Student Clustering", page_icon="🎓", layout="wide")

# CSS ottimizzato
st.markdown("""
<style>
.metric-card {background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4CAF50;}
.step-box {background: #e8f5e8; padding: 15px; border-radius: 8px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 K-Means Clustering per Analisi Studenti")

# Generazione dati
@st.cache_data
def generate_data(n=1000):
    np.random.seed(42)
    return pd.DataFrame({
        'student_id': range(1, n + 1),
        'anno_iscrizione': np.random.choice([2020, 2021, 2022, 2023, 2024], n),
        'eta': np.clip(np.random.normal(22, 2.5, n), 18, 35).astype(int),
        'ore_studio_settimanali': np.clip(np.random.gamma(2, 8, n), 5, 60),
        'numero_esami_superati': np.clip(np.random.poisson(12, n), 0, 30),
        'media_voti': np.clip(np.random.normal(24, 3, n), 18, 30),
        'partecipazione_eventi': np.random.choice([0, 1], n),
        'lavoro_part_time': np.random.choice([0, 1], n),
        'carriera_in_corso': np.random.choice([0, 1], n)
    })

# Preprocessing
@st.cache_data
def preprocess_data(df, features, imputation, scaling):
    df_proc = df[features].copy()
    
    # Gestione valori mancanti
    if imputation == "drop":
        df_proc = df_proc.dropna()
    else:
        imputer = KNNImputer(n_neighbors=5) if imputation == "knn" else SimpleImputer(strategy=imputation)
        df_proc = pd.DataFrame(imputer.fit_transform(df_proc), columns=features, index=df_proc.index)
    
    # Normalizzazione
    if scaling != "None":
        scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}[scaling]
        df_proc = pd.DataFrame(scaler.fit_transform(df_proc), columns=features, index=df_proc.index)
    
    return df.loc[df_proc.index], df_proc

# Analisi Elbow
@st.cache_data
def elbow_analysis(data, max_k=8):
    sse, sil_scores = [], []
    k_range = range(2, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(data)
        sse.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(data, labels))
    return k_range, sse, sil_scores

# Tabs principali
tab1, tab2, tab3 = st.tabs(["📊 Analisi", "🎯 Guida Visiva", "📚 Teoria"])

# TAB 1: ANALISI PRINCIPALE
with tab1:
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurazioni")
        n_samples = st.slider("N. studenti", 100, 2000, 1000, 100)
        df = generate_data(n_samples)
        
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'student_id']
        features = st.multiselect("Features", numeric_cols, default=numeric_cols[:5])
        
        if not features:
            st.error("Seleziona almeno una feature!")
            st.stop()
        
        n_clusters = st.slider("N. cluster", 2, 10, 4)
        init_method = st.selectbox("Inizializzazione", ["k-means++", "random"])
        scaling = st.selectbox("Normalizzazione", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
        imputation = st.selectbox("Valori mancanti", ["mean", "median", "most_frequent", "knn", "drop"])

    # Preprocessing
    df_orig, df_proc = preprocess_data(df, features, imputation, scaling)
    st.success(f"✅ {len(df_proc)} studenti, {len(features)} features processate")

    # EDA rapida
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Statistiche")
        st.dataframe(df_proc.describe())

    with col2:
        st.subheader("🔗 Correlazioni")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_proc.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

    # Analisi Elbow
    if st.button("🔍 Analisi Elbow"):
        k_range, sse, sil_scores = elbow_analysis(df_proc)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(k_range, sse, 'bo-')
            ax.set_xlabel('K'); ax.set_ylabel('SSE'); ax.set_title('Elbow Method'); ax.grid(True)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(k_range, sil_scores, 'ro-')
            ax.set_xlabel('K'); ax.set_ylabel('Silhouette'); ax.set_title('Silhouette Score'); ax.grid(True)
            st.pyplot(fig)
        
        st.info(f"💡 K ottimale: {k_range[np.argmax(sil_scores)]}")

    # Clustering
    if st.button("▶️ Esegui Clustering", type="primary"):
        with st.spinner("Clustering..."):
            kmeans = KMeans(n_clusters=n_clusters, init=init_method, random_state=42)
            labels = kmeans.fit_predict(df_proc)
            
            df_results = df_orig.copy()
            df_results['cluster'] = labels
            df_results['distance'] = np.min(kmeans.transform(df_proc), axis=1)
            
            sil_score = silhouette_score(df_proc, labels)
            
            st.session_state.update({
                'results': df_results, 'processed': df_proc, 'kmeans': kmeans,
                'features': features, 'silhouette': sil_score
            })
            
            st.success("✅ Clustering completato!")

    # Visualizzazione risultati
    if 'results' in st.session_state:
        df_results = st.session_state['results']
        kmeans = st.session_state['kmeans']
        sil_score = st.session_state['silhouette']
        
        st.header("📈 Risultati")
        
        # Metriche
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{sil_score:.3f}")
        col2.metric("N. Cluster", n_clusters)
        col3.metric("N. Studenti", len(df_results))
        
        # Visualizzazioni
        cluster_counts = df_results['cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(cluster_counts.values, labels=[f"Cluster {i}" for i in cluster_counts.index], autopct='%1.1f%%')
            ax.set_title("Distribuzione Cluster")
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar([f"Cluster {i}" for i in cluster_counts.index], cluster_counts.values)
            ax.set_title("Studenti per Cluster")
            st.pyplot(fig)
        
        # PCA
        if len(features) > 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(df_proc)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
            
            for i in range(n_clusters):
                mask = df_results['cluster'] == i
                ax.scatter(pca_data[mask, 0], pca_data[mask, 1], c=[colors[i]], label=f'Cluster {i}', alpha=0.7)
            
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', marker='x', s=200, linewidths=3, label='Centroidi')
            
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
            ax.set_title(f"PCA Clustering (Var: {pca.explained_variance_ratio_.sum():.1%})")
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Tabelle
        st.subheader("📋 Centroidi")
        centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=features, 
                                   index=[f"Cluster {i}" for i in range(n_clusters)]).round(3)
        st.dataframe(centroids_df)
        
        st.subheader("📄 Risultati")
        st.dataframe(df_results.head(100))
        
        # Download
        csv = df_results.to_csv(index=False)
        st.download_button("📁 Scarica CSV", csv, "clustering_results.csv", "text/csv")

# TAB 2: GUIDA VISIVA
with tab2:
    st.header("🎯 K-Means: Algoritmo Passo dopo Passo")
    
    # Dati demo semplificati
    @st.cache_data
    def create_demo_data():
        np.random.seed(42)
        group1 = np.random.normal([15, 25], [3, 1.5], (6, 2))
        group2 = np.random.normal([35, 22], [2, 1], (6, 2))
        group3 = np.random.normal([20, 28], [2, 1], (6, 2))
        data = np.vstack([group1, group2, group3])
        return pd.DataFrame(data, columns=['ore_studio', 'media_voti'])

    demo_df = create_demo_data()
    
    # Controlli
    if 'demo_step' not in st.session_state:
        st.session_state.demo_step = 0
        st.session_state.demo_centroids = np.array([[12, 23], [25, 26], [38, 21]])
        st.session_state.demo_assignments = None
    
    col1, col2, col3 = st.columns(3)
    if col1.button("⏮️ Precedente"): st.session_state.demo_step = max(0, st.session_state.demo_step - 1)
    if col2.button("⏭️ Successivo"): st.session_state.demo_step = min(3, st.session_state.demo_step + 1)
    if col3.button("🔄 Reset"): st.session_state.demo_step = 0
    
    # Passi della demo
    steps = ["Dati Iniziali", "Inizializzazione Centroidi", "Assegnazione Cluster", "Risultato Finale"]
    
    st.markdown(f"""
    <div class="step-box">
        <h3>Passo {st.session_state.demo_step + 1}: {steps[st.session_state.demo_step]}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizzazione
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'blue', 'green']
    
    if st.session_state.demo_step == 0:
        ax.scatter(demo_df['ore_studio'], demo_df['media_voti'], c='gray', s=100, alpha=0.7)
    
    elif st.session_state.demo_step == 1:
        ax.scatter(demo_df['ore_studio'], demo_df['media_voti'], c='gray', s=100, alpha=0.7)
        for i, centroid in enumerate(st.session_state.demo_centroids):
            ax.scatter(centroid[0], centroid[1], c=colors[i], s=200, marker='X', edgecolors='black', linewidth=2)
    
    elif st.session_state.demo_step >= 2:
        # Calcola assegnazioni
        assignments = []
        for _, point in demo_df.iterrows():
            distances = [np.sqrt((point['ore_studio'] - c[0])**2 + (point['media_voti'] - c[1])**2) 
                        for c in st.session_state.demo_centroids]
            assignments.append(np.argmin(distances))
        
        # Visualizza punti colorati
        for i, (_, point) in enumerate(demo_df.iterrows()):
            cluster = assignments[i]
            ax.scatter(point['ore_studio'], point['media_voti'], c=colors[cluster], s=100, alpha=0.7)
        
        # Centroidi
        for i, centroid in enumerate(st.session_state.demo_centroids):
            ax.scatter(centroid[0], centroid[1], c=colors[i], s=200, marker='X', edgecolors='black', linewidth=2)
        
        # Aggiorna centroidi per passo finale
        if st.session_state.demo_step == 3:
            new_centroids = []
            for cluster_id in range(3):
                cluster_points = demo_df[np.array(assignments) == cluster_id]
                if len(cluster_points) > 0:
                    new_centroids.append([cluster_points['ore_studio'].mean(), cluster_points['media_voti'].mean()])
                else:
                    new_centroids.append(st.session_state.demo_centroids[cluster_id])
            st.session_state.demo_centroids = np.array(new_centroids)
    
    ax.set_xlabel('Ore Studio'); ax.set_ylabel('Media Voti')
    ax.set_title(f'Passo {st.session_state.demo_step + 1}: {steps[st.session_state.demo_step]}')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# TAB 3: TEORIA
with tab3:
    st.header("📚 Come Funziona K-Means")
    
    st.markdown("""
    ## 🎯 Algoritmo K-Means
    
    K-Means è un algoritmo di clustering che raggruppa i dati in k cluster basandosi sulla similarità.
    
    ### 🔄 Passi dell'Algoritmo:
    
    1. **Inizializzazione**: Posiziona k centroidi casuali
    2. **Assegnazione**: Ogni punto va al centroide più vicino  
    3. **Aggiornamento**: Sposta centroidi al centro dei cluster
    4. **Iterazione**: Ripeti 2-3 fino a convergenza
    
    ### 📏 Formula Distanza:
    ```
    d(x,c) = √[(x₁-c₁)² + (x₂-c₂)² + ... + (xₙ-cₙ)²]
    ```
    
    ### ⚙️ Preprocessing Importante:
    
    - **Normalizzazione**: Scale diverse dominano le distanze
    - **Valori Mancanti**: Possono distorcere i cluster
    - **Selezione Features**: Rimuovi features irrilevanti
    
    ### 📊 Metriche di Valutazione:
    
    - **Silhouette Score**: Qualità separazione cluster (-1 a 1)
    - **Elbow Method**: Trova k ottimale con SSE
    - **Inertia**: Somma distanze quadrate dai centroidi
    
    ### 💡 Consigli Pratici:
    
    - Usa k-means++ per inizializzazione migliore
    - Prova diversi valori di k (2-10)
    - Normalizza sempre i dati
    - Interpreta i cluster nel contesto del dominio
    """)
    
    # Esempio pratico con formule
    st.subheader("🧮 Esempio Calcolo")
    
    with st.expander("Calcolo Distanza Euclidea"):
        st.markdown("""
        **Studente A**: 20 ore studio, 25 media voti  
        **Centroide C1**: 18 ore studio, 24 media voti
        
        **Distanza**: √[(20-18)² + (25-24)²] = √[4 + 1] = √5 = 2.24
        """)
    
    with st.expander("Aggiornamento Centroide"):
        st.markdown("""
        **Cluster con 3 studenti**:
        - Studente 1: (15, 23)
        - Studente 2: (18, 25)  
        - Studente 3: (21, 27)
        
        **Nuovo Centroide**: ((15+18+21)/3, (23+25+27)/3) = (18, 25)
        """)
