import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
import scipy.stats
from scipy.stats import hmean
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import seaborn as sns

# --- Ignoriši upozorenja ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Konfiguracija Stranice ---
st.set_page_config(
    page_title="Semantička analiza",
    page_icon=":material/diversity_2:",
    layout="wide"
)

# --- Inicijalizacija session state ---
if 'words' not in st.session_state:
    st.session_state.words = []
if 'word_embeddings' not in st.session_state:
    st.session_state.word_embeddings = None
if 'eigenvalues' not in st.session_state:
    st.session_state.eigenvalues = None
if 'explained_variances' not in st.session_state:
    st.session_state.explained_variances = None
if 'pca_error_msg' not in st.session_state:
    st.session_state.pca_error_msg = None
if 'avg_cos_dist' not in st.session_state:
    st.session_state.avg_cos_dist = None
if 'total_variance' not in st.session_state:
    st.session_state.total_variance = None
if 'pca_components' not in st.session_state:
    st.session_state.pca_components = []
if 'dist_matrix' not in st.session_state:
    st.session_state.dist_matrix = []
if 'pairwise_distances' not in st.session_state:
    st.session_state.pairwise_distances = []

# --- Učitavanje modela---
@st.cache_resource
def load_model():
    try:
        model_name = "sentence-transformers/LaBSE"
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Greška pri učitavanju modela '{model_name}': {e}")
        return None

model = load_model()
if model is not None:
    st.sidebar.success("Model uspešno učitan.")

# --- Dobijanje embedinga ---
@st.cache_data(show_spinner=False)
def get_embeddings(words, _model):
    if not _model:
        return None, "Model nije dostupan."
    if not words:
        return np.array([]), None

    try:
        embeddings = _model.encode(words, convert_to_numpy=True,normalize_embeddings=True)
        return embeddings, None
    except Exception as e:
        return None, f"Greška pri generisanju embedinga: {e}"

# --- Funkcija za PCA Analizu ---
def calculate_pca(embeddings):
    if embeddings is None or embeddings.shape[0] < 2:
        return None, None, None, "Potrebno je bar 2 validna embedinga za PCA analizu."
    if np.allclose(embeddings, embeddings[0,:], atol=1e-6):
        return None, None, None, "Svi embedinzi su skoro identični. PCA nije smislena."

    n_samples, n_features = embeddings.shape
    n_components = min(n_samples, n_features)
    if n_components < 1:
        return None, None, None, "Nije moguće izračunati komponente."

    try:
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(embeddings)
        eigenvalues = pca.explained_variance_
        explained_variance_ratios = pca.explained_variance_ratio_
        total_variance = np.sum(eigenvalues)

        valid_indices = explained_variance_ratios > 1e-6
        if not np.any(valid_indices) and len(eigenvalues) > 0:
            valid_indices = [0]

        if not np.any(valid_indices):
            return None, None, None, "Ni jedna PCA komponenta nema značajnu varijansu."

        eigenvalues = eigenvalues[valid_indices]
        explained_variance_ratios = explained_variance_ratios[valid_indices]

        return pca_components, eigenvalues, explained_variance_ratios, total_variance, None
    except Exception as e:
        return None, None, None, None, f"Greška tokom PCA analize: {e}"

# --- Funkcija za računanje prosečne kosinusne distance ---
def calculate_avg_cosine_distance(embeddings):
    if embeddings is None or embeddings.shape[0] < 2: 
        return None, "Potrebno je bar 2 embedinga za računanje kosinusne udaljenosti"
    try:
        sim_matrix = cosine_similarity(embeddings) # (10x10)
        dist_matrix = 1.0 - sim_matrix             # (10x10)
        n = dist_matrix.shape[0]
        upper_triangle_indices = np.triu_indices(n, k=1)
        pairwise_distances = pd.DataFrame(dist_matrix[upper_triangle_indices].ravel())
        avg_cos_dist = np.mean(pairwise_distances) if pairwise_distances.size > 0 else 0.0
        norms = norm(embeddings, axis=1)
        print("---------------Norme po vektoru---------------", norms)
        print("---------------Matrica distanci embedinga---------------", dist_matrix)
        print("---------------Distance---------------", pairwise_distances)

        return avg_cos_dist,dist_matrix,pairwise_distances, None
    except Exception as e:
        return None, f"Greška kod kosinusne udaljenosti: {e}"

# --- Interfejs ---
st.title("Semantička analiza reči")
st.markdown("""
Unesite tačno 10 imenica u nominativu jednine na srpskom jeziku, razdvojenih razmakom ili svaku u novom redu.
""")

# Provera da li je model uspešno učitan
if model is None:
    st.warning("Model nije uspešno učitan. Aplikacija ne može da nastavi sa radom.", icon=":material/warning:")
else:
    # Unos teksta od korisnika
    input_text = st.text_area("Reči:", height=150, placeholder="Budite kreativni.")

    # Dugme za pokretanje analize
    if st.button("Analiza", type="primary", icon= ":material/psychology_alt:"):
        if not input_text.strip():
            st.warning("Molimo unesite reči.", icon=":material/warning:")
        else:
            # Obrada unosa
            st.session_state.words = [word.strip().lower() for word in input_text.split() if word.strip()]

            if len(st.session_state.words) != 10:
                st.error(f"Uneto je {len(st.session_state.words)} reči. Molimo unesite tačno 10 reči.", icon=":material/error:")
            else:
                st.info(f"Reči: {', '.join(st.session_state.words)}", icon=":material/psychology:")

                # Generisanje Embedinga
                embeddings_error = False
                with st.spinner("Računanje embedinga..."):
                    st.session_state.word_embeddings, embed_error_msg = get_embeddings(st.session_state.words, model)
                    if embed_error_msg:
                        st.error(f"Greška pri dobijanju embedinga: {embed_error_msg}", icon=":material/error:")
                        embeddings_error = True
                    elif st.session_state.word_embeddings is None or st.session_state.word_embeddings.shape[0] != 10:
                        st.error("Nije dobijen očekivani broj embedinga (10).", icon=":material/error:")
                        embeddings_error = True

                # Nastavi samo ako nema greške u embedinzima
                if not embeddings_error:
                    st.success(f"Embedinzi uspešno izračunati! Dimenzije: `{st.session_state.word_embeddings.shape}`", icon=":material/done_outline:")

                    # --- Izračunavanje i Prikaz Metrika Različitosti ---
                    st.divider()
                    st.subheader("Mere semantičke distance")

                    # PCA Analiza
                    with st.spinner("PCA analiza..."):
                        st.session_state.pca_components, st.session_state.eigenvalues, st.session_state.explained_variances, st.session_state.total_variance, st.session_state.pca_error_msg = calculate_pca(st.session_state.word_embeddings)
                    
                    # Kosinusna udaljenost
                    with st.spinner("Računanje kosinusne udaljenosti..."):
                        st.session_state.avg_cos_dist,st.session_state.dist_matrix,st.session_state.pairwise_distances, cos_err = calculate_avg_cosine_distance(st.session_state.word_embeddings)
                    
                    # Prikaz metrika
                    col1, col2 = st.columns(2)

                    with col1:
                        if cos_err:
                            st.error(f"Kosinus: {cos_err}", icon=":material/error:")
                        elif st.session_state.pairwise_distances is not None:
                            st.metric(label="Medijalna udaljenost", value=f"{np.median(st.session_state.pairwise_distances):.4f}")

                    with col2:
                        if cos_err:
                            st.error(f"Kosinus: {cos_err}", icon=":material/error:")
                        elif st.session_state.avg_cos_dist is not None:
                            st.metric(label="Srednja udaljenost", value=f"{st.session_state.avg_cos_dist:.4f}")

                    # --- Knn metrike ---
                    with st.expander("Ocene distance k najsličnijih reči"):
                        if 'pairwise_distances' in st.session_state and st.session_state.pairwise_distances is not None:
                            distances = st.session_state.pairwise_distances.iloc[:, 0].sort_values().values
                            n = len(distances)
                            k_values = list(range(2, n + 1))
                            harmonic_means = []
                            arithmetic_means = []
                            medians = []

                            for k in k_values:
                                top_k_distances = distances[:k]
                                harmonic_means.append(hmean(top_k_distances))
                                arithmetic_means.append(np.mean(top_k_distances))
                                medians.append(np.median(top_k_distances))

                            # Kreiranje DataFrame-a za Plotly
                            plot_data = pd.DataFrame({
                                'k': k_values,
                                'Harmonijska sredina': harmonic_means,
                                'Prosek': arithmetic_means,
                                'Medijana': medians
                            })

                            # Kreiranje interaktivnog grafa pomoću Plotly Express
                            fig = px.line(plot_data, x='k', y=['Harmonijska sredina','Prosek','Medijana'],
                                          title='Ocene',
                                          labels={'value': 'Ocena', 'k': 'Broj najsličnijih reči'})
                            st.plotly_chart(fig)

                        else:
                            st.info("Podaci o parnim udaljenostima još nisu dostupni.")


                    # --- Histogram ---       
                    st.divider()
                    with st.subheader("Histogram"):
                        if cos_err:
                            st.error(f"Kosinus: {cos_err}", icon=":material/error:")
                        elif st.session_state.pairwise_distances is not None:
                            fig_histogram = px.histogram(
                                st.session_state.pairwise_distances,
                                nbins=10,
                                range_x=[0, 1],
                                color_discrete_sequence=["#5d6e25"],
                                labels={"value": "Kosinusna sličnost"},
                                title="Raspodela kosinusnih sličnosti"
                            )
                            fig_histogram.update_layout(
                                bargap=0.02,
                                xaxis=dict(
                                    tickvals=[i/20 for i in range(21)],
                                    tickformat=".2f"
                                )
                            )
                            fig_histogram.update_traces(
                                showlegend=False
                            )
                            st.plotly_chart(fig_histogram)
                    

    # --- Prikaz PCA ---
    if st.session_state.word_embeddings is not None and len(st.session_state.words) == 10 and st.session_state.pca_error_msg is None and st.session_state.pca_components is not None:
        st.divider()
        

        if st.session_state.explained_variances is not None:
            st.divider()
            with st.expander("PCA analiza - objašnjena varijansa"):
                st.subheader("Varijabilitet po PCA komponentama")
                
                # Priprema podataka
                components = [f"PC{i+1}" for i in range(len(st.session_state.explained_variances))]
                explained_var = st.session_state.explained_variances
                cumulative_var = np.cumsum(explained_var)
                
                # Kreiranje kombinovanog grafa (bar + line)
                fig = go.Figure()
                
                # Bar plot za individualnu varijansu
                fig.add_trace(go.Bar(
                    x=components,
                    y=explained_var,
                    name="Individualna varijansa",
                    marker_color='#5d6e25',
                    text=[f"{v:.1%}" for v in explained_var],
                    textposition='auto'
                ))
                
                # Line plot za kumulativnu varijansu
                fig.add_trace(go.Scatter(
                    x=components,
                    y=cumulative_var,
                    name="Kumulativna varijansa",
                    line=dict(color='#80431d', width=3),
                    marker=dict(size=8),
                    text=[f"{v:.1%}" for v in cumulative_var],
                    textposition="top center"
                ))
                
                # Podešavanje layout-a
                fig.update_layout(
                    barmode='group',
                    yaxis=dict(
                        title="Objašnjena varijansa",
                        tickformat=".0%",
                        range=[0, 1.1]  # Osigurava prostor za tekst
                    ),
                    xaxis_title="PCA komponente",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # --- HEATMAP KOSINUSNE DISTANCE ---
            st.divider()
            st.subheader("Heatmap semantičke razlike")
            
            if 'avg_cos_dist' in st.session_state and st.session_state.avg_cos_dist is not None:
                # Priprema podataka za heatmap
                cos_sim_matrix = st.session_state.dist_matrix
                df_cos_sim = pd.DataFrame(
                    cos_sim_matrix,
                    index=st.session_state.words,
                    columns=st.session_state.words
                )
                
                # Kreiranje heatmap sa Plotly
                fig_heatmap = px.imshow(
                    df_cos_sim,
                    text_auto=".2f",
                    color_continuous_scale='Jet',
                    labels=dict(x="Reč", y="Reč", color="Razlika"),
                    zmin=0,
                    zmax=1
                )
                fig_heatmap.update_layout(
                    width=700,
                    height=700,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False
                )
                
                fig_heatmap.update_traces(
                    hovertemplate="Reč 1: %{y}<br>Reč 2: %{x}<br><extra></extra>"
                )

                # Prikaz heatmap
                st.plotly_chart(fig_heatmap, use_container_width=True)
                # Opciono: Download matrice kao CSV
                csv = df_cos_sim.to_csv().encode('utf-8')
                st.download_button(
                    label="Preuzmi matricu kao CSV",
                    data=csv,
                    file_name='kosinusna_razlika.csv',
                    mime='text/csv'
                )

# --- Dodatne Informacije ---
st.sidebar.header("Info")
st.sidebar.header("Model")
st.sidebar.markdown("[LaBSE](https://huggingface.co/sentence-transformers/LaBSE)")
st.sidebar.header("Korišćene biblioteke")
st.sidebar.markdown("""
* Streamlit
* Sentence Transformers
* Scikit-learn
* NumPy
* SciPy
* Matplotlib
* Pandas
* Plotly
""")
# Primeri

# Velika slova ignorise ali moram da sredim i osisanu latinicu koja vidno utice na rezultate.

# bicikl ananas oblak skalamerija sočivo melanholija inicijali interpunkcija godišnjica kreda - 0.6495

# tegla poklopac staklo turšija krastavčići krastavčić krastavac kiselo kupus čep - 0.4810

# [theme]
# base="dark"
# primaryColor="#da5555"
# secondaryBackgroundColor="#748ebd"
