import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
import scipy.stats
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import normalize
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

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
        pairwise_distances = dist_matrix[upper_triangle_indices]
        avg = np.mean(pairwise_distances) if pairwise_distances.size > 0 else 0.0
        norms = norm(embeddings, axis=1)
        normalized_norms = norm(normalize(embeddings),axis=1)
        print("---------------Norme po vektoru--------------- /n", norms)
        print("---------------Matrica distanci embedinga--------------- /n", dist_matrix)

        return avg,dist_matrix, None
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
                        st.session_state.avg_cos_dist,st.session_state.dist_matrix, cos_err = calculate_avg_cosine_distance(st.session_state.word_embeddings)
                    
                    # Prikaz metrika
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.session_state.pca_error_msg:
                            st.error(f"PCA: {st.session_state.pca_error_msg}", icon=":material/error:")
                        elif st.session_state.explained_variances is not None:
                            st.metric(label="Ukupna varijansa (PCA)", value=f"{st.session_state.total_variance:.4f}")
                            st.caption("Veća vrednost ≈ veći ukupan 'spread' podataka.")

                    with col2:
                        if cos_err:
                            st.error(f"Kosinus: {cos_err}", icon=":material/error:")
                        elif st.session_state.avg_cos_dist is not None:
                            st.metric(label="Prosečna cos udaljenost", value=f"{st.session_state.avg_cos_dist:.4f}")
                            st.caption("Veća vrednost ≈ veća prosečna semantička različitost.")

                    with col3:
                        if st.session_state.pca_error_msg:
                            st.metric(label="Objašnjena varijansa (PC1)", value="N/A")
                        elif st.session_state.explained_variances is not None and len(st.session_state.explained_variances) > 0:
                            st.metric(label="Objašnjena varijansa (PC1)", value=f"{st.session_state.explained_variances[0]:.2%}")
                            st.caption("Opisan varijabilitet treba biti što raspršeniji po komponentama")

    # --- Prikaz PCA ---
    if st.session_state.word_embeddings is not None and len(st.session_state.words) == 10 and st.session_state.pca_error_msg is None and st.session_state.pca_components is not None:
        st.divider()
        

        # --- Prikaz PCA baze ---
        if st.session_state.explained_variances is not None:
            st.divider()
            with st.expander("PCA analiza - objašnjena varijansa"):
                st.subheader("Varijabilitet po PCA komponentama")
                
                # Priprema podataka za tabelu
                components_data = {
                    "Komponenta": [f"PC{i+1}" for i in range(len(st.session_state.eigenvalues))],
                    "Objašnjena varijansa": st.session_state.explained_variances,
                    "Kumulativna varijansa": [np.sum(st.session_state.explained_variances[:i+1]) 
                                            for i in range(len(st.session_state.explained_variances))]
                }
                
                # Kreiranje tabele sa stilom
                df_pca = pd.DataFrame(components_data)
                st.dataframe(
                    df_pca.style.format({
                        'Objašnjena varijansa': '{:.2%}',
                        'Kumulativna varijansa': '{:.2%}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
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
                    color_continuous_scale='Inferno',
                    labels=dict(x="Reč", y="Reč", color="Razlika [0,1]")
                )
                fig_heatmap.update_layout(
                    width=800,
                    height=800,
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



            # --- t-SNE Vizuelizacija ---
            st.divider()
            st.subheader("3D t-SNE projekcija semantičkih odnosa")

            # Parametri
            col1, col2 = st.columns(2)
            with col1:
                perplexity = st.slider(
                    "Perplexity",
                    min_value=2,
                    max_value=max(5, len(st.session_state.words)-1),
                    value=min(5, len(st.session_state.words)-1),
                    key="tsne_perplexity_3d"
                )
            with col2:
                learning_rate = st.slider(
                    "Learning Rate",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    key="tsne_lr_3d"
                )

            # Izračunavanje
            if st.session_state.word_embeddings is not None and len(st.session_state.words) > 2:
                try:
                    tsne = TSNE(
                        n_components=3,  # 3 komponente za 3D prikaz
                        perplexity=perplexity,
                        learning_rate=learning_rate,
                        random_state=42,
                        init='pca'
                    )
                    
                    with st.spinner('Izračunavam 3D t-SNE projekciju...'):
                        tsne_results = tsne.fit_transform(st.session_state.word_embeddings)
                    
                    # Priprema podataka
                    tsne_df = pd.DataFrame({
                        'Reč': st.session_state.words,
                        't-SNE X': tsne_results[:, 0],
                        't-SNE Y': tsne_results[:, 1],
                        't-SNE Z': tsne_results[:, 2]
                    })
                    
                    # Kreiranje "clean" 3D scatter plota bez vizuelnih elemenata
                    fig_3d = px.scatter_3d(
                        tsne_df,
                        x='t-SNE X',
                        y='t-SNE Y',
                        z='t-SNE Z',
                        text='Reč',
                        hover_name='Reč'
                    )

                    # Potpuno čisti izgled
                    fig_3d.update_traces(
                        marker=dict(
                            size=1,          # (ne može biti 0)
                            opacity=0,
                            color='white'        
                        ),
                        textfont=dict(
                            size=18,         # Povećajte font po potrebi
                            color='firebrick'    # Boja teksta
                        ),
                        hovertemplate='%{hovertext}<extra></extra>'  # Sakrij dodatne hover informacije
                    )

                    fig_3d.update_layout(
                        scene=dict(
                            xaxis=dict(
                                visible=False,  # Sakrij X osu
                                showbackground=False,
                                showticklabels=False
                            ),
                            yaxis=dict(
                                visible=False,  # Sakrij Y osu
                                showbackground=False,
                                showticklabels=False
                            ),
                            zaxis=dict(
                                visible=False,  # Sakrij Z osu
                                showbackground=False,
                                showticklabels=False
                            ),
                            bgcolor='rgba(0,0,0,0)'  # Providna pozadina
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',  # Providna pozadina izvan grafikona
                        margin=dict(l=0, r=0, b=0, t=0),
                        showlegend=False
                    )

                    # Opciono: Centriranje teksta
                    fig_3d.update_traces(textposition='middle center')

                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Greška pri 3D t-SNE analizi: {str(e)}")
            else:
                st.warning("Potrebno je najmanje 3 reči za 3D prikaz")



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
