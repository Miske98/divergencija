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
if 'input_type' not in st.session_state:
    st.session_state.input_type = "Reči"
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
    st.session_state.pca_components = None
if 'dist_matrix' not in st.session_state:
    st.session_state.dist_matrix = None
if 'pairwise_distances' not in st.session_state:
    st.session_state.pairwise_distances = None

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
def get_embeddings(items, _model):
    if not _model:
        return None, "Model nije dostupan."
    if not items:
        return np.array([]), None

    try:
        embeddings = _model.encode(items, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings, None
    except Exception as e:
        return None, f"Greška pri generisanju embedinga: {e}"

# --- Funkcija za PCA Analizu ---
def calculate_pca(embeddings):
    if embeddings is None or embeddings.shape[0] < 2:
        return None, None, None, None, "Potrebno je bar 2 validna embedinga za PCA analizu."


    if embeddings.shape[0] > 1 and np.allclose(embeddings, embeddings[0,:], atol=1e-6):
         return None, None, None, None, "Svi embedinzi su skoro identični. PCA analiza nije smislena."

    n_samples, n_features = embeddings.shape
    n_components = min(n_samples, n_features)
    if n_components < 1:
        return None, None, None, None, "Nije moguće izračunati komponente (broj uzoraka ili feature-a manji od 1)."

    try:
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(embeddings)
        eigenvalues = pca.explained_variance_
        explained_variance_ratios = pca.explained_variance_ratio_
        total_variance = np.sum(eigenvalues)

        # Filtriranje komponenti sa skoro nultom varijansom
        # Prag može zavisiti od skale podataka. Dinamički prag je bolji od fiksnog.
        # Zadržavamo komponente gde je objašnjena varijansa > mali procenat ukupne varijanse prve komponente ili fiksni prag
        valid_indices = explained_variance_ratios > max(1e-9, explained_variance_ratios[0] * 1e-5) if explained_variance_ratios.size > 0 else []
        if not np.any(valid_indices) and eigenvalues.size > 0 and eigenvalues[0] > 0:
             valid_indices = [0]

        if not np.any(valid_indices):
            return None, None, None, None, "Ni jedna PCA komponenta nema značajnu varijansu."

        # Filtriraj rezultate samo za validne komponente
        eigenvalues = eigenvalues[valid_indices]
        explained_variance_ratios = explained_variance_ratios[valid_indices]
        pca_components = pca_components[:, valid_indices]

        return pca_components, eigenvalues, explained_variance_ratios, total_variance, None
    except Exception as e:
        return None, None, None, None, f"Greška tokom PCA analize: {e}"

# --- Funkcija za računanje prosečne kosinusne distance ---
def calculate_avg_cosine_distance(embeddings):
    """Računa matricu kosinusnih udaljenosti i prosečnu udaljenost."""
    if embeddings is None or embeddings.shape[0] < 2:
        return None, None, None, "Potrebno je bar 2 embedinga za računanje kosinusne udaljenosti."
    try:
        sim_matrix = cosine_similarity(embeddings) # (N x N) matrica sličnosti
        dist_matrix = 1.0 - sim_matrix             # (N x N) matrica udaljenosti (1 - sličnost)

        # Izvuci parne udaljenosti iz gornjeg trougla matrice udaljenosti (bez dijagonale)
        n = dist_matrix.shape[0]
        upper_triangle_indices = np.triu_indices(n, k=1)
        pairwise_distances_values = dist_matrix[upper_triangle_indices].ravel()

        # Kreiraj DataFrame od parnih udaljenosti za lakšu analizu i vizualizaciju
        pairwise_distances_df = pd.DataFrame(pairwise_distances_values, columns=['Distance'])

        # Računaj prosečnu kosinusnu udaljenost (ignorišući NaN ili Inf ako postoje)
        avg_cos_dist = np.mean(pairwise_distances_values[np.isfinite(pairwise_distances_values)]) if pairwise_distances_values.size > 0 else 0.0

        return avg_cos_dist, dist_matrix, pairwise_distances_df, None
    except Exception as e:
        return None, None, None, f"Greška kod kosinusne udaljenosti: {e}"

# --- Interfejs ---
st.title("Semantička analiza")

# --- Izbor tipa unosa ---
st.session_state.input_type = st.radio(
    "Izaberite tip unosa:",
    ("Reči", "Rečenice"),
    index=0 if st.session_state.input_type == "Reči" else 1,
    horizontal=True
)

# --- Dinamička uputstva --- 
if st.session_state.input_type == "Reči":
    st.markdown("""
    Unesite tačno **10 reči** (poželjno imenica u nominativu jednine na srpskom jeziku), razdvojenih razmakom ili svaku u novom redu.
    """)
    placeholder_text = "Unesite 10 reči..."
    expected_count = 10
    count_unit = "reči"
else: # Rečenice
    st.markdown("""
    Unesite tačno **5 rečenica** na srpskom jeziku. Svaka rečenica mora biti završena tačkom (`.`).
    """)
    placeholder_text = "Unesite 5 rečenica, svaku završite tačkom."
    expected_count = 5
    count_unit = "rečenica"


# Provera da li je model uspešno učitan
if model is None:
    st.warning("Model nije uspešno učitan. Aplikacija ne može da nastavi sa radom.", icon=":material/warning:")
else:
    # Unos teksta od korisnika u text area
    input_text = st.text_area(f"Unos ({st.session_state.input_type}):", height=150, placeholder=placeholder_text)

    # Dugme za pokretanje analize
    if st.button("Analiza", type="primary", icon= ":material/psychology_alt:"):
        st.session_state.word_embeddings = None
        st.session_state.pca_components = None
        st.session_state.eigenvalues = None
        st.session_state.explained_variances = None
        st.session_state.total_variance = None
        st.session_state.pca_error_msg = None
        st.session_state.avg_cos_dist = None
        st.session_state.dist_matrix = None
        st.session_state.pairwise_distances = None
        st.session_state.words = [] # Resetuj listu unosa

        if not input_text.strip():
            st.warning(f"Molimo unesite {count_unit}.", icon=":material/warning:")
        else:
            if not isinstance(st.session_state.words, list):
                 print(f"DEBUG: st.session_state.words nije lista ({type(st.session_state.words)}). Resetovanje na [].")
                 st.session_state.words = [] # RESETUJEMO NA PRAZNU LISTU AKO NIJE LISTA
                 print("Pretvaranje   ",[item.strip() for item in input_text.split('.') if item.strip()])
                 print("Items su : ",st.session_state.words)

            # Obrada unosa na osnovu tipa
            if st.session_state.input_type == "Reči":
                st.session_state.words = [item.strip().lower() for item in input_text.split() if item.strip()]
            else:
                st.session_state.words = [item.strip() for item in input_text.split('.') if item.strip()]
            print(st.session_state.words)

            if len(st.session_state.words) != expected_count:
                st.error(f"Uneto je {len(st.session_state.words)} {count_unit}. Molimo unesite tačno {expected_count} {count_unit}.", icon=":material/error:")
                # Resetuj listu unosa i rezultate na grešku u broju
                st.session_state.words = []
                st.session_state.word_embeddings = None
                st.session_state.pca_components = None
                st.session_state.eigenvalues = None
                st.session_state.explained_variances = None
                st.session_state.total_variance = None
                st.session_state.pca_error_msg = None
                st.session_state.avg_cos_dist = None
                st.session_state.dist_matrix = None
                st.session_state.pairwise_distances = None
            elif len(st.session_state.words) < 2:
                 # Dodatna provera: potrebne su bar 2 stavke za PCA i kosinusnu udaljenost
                 st.error(f"Potrebno je bar 2 {count_unit} za analizu.", icon=":material/error:")
                 st.session_state.words = []
                 st.session_state.word_embeddings = None
                 st.session_state.pca_components = None
                 st.session_state.eigenvalues = None
                 st.session_state.explained_variances = None
                 st.session_state.total_variance = None
                 st.session_state.pca_error_msg = None
                 st.session_state.avg_cos_dist = None
                 st.session_state.dist_matrix = None
                 st.session_state.pairwise_distances = None

            else:
                # Ako validacija prođe, prikaži unete stavke
                st.info(f"Unete {st.session_state.input_type}: {', '.join(st.session_state.words)}", icon=":material/psychology:")

                # Generisanje Embedinga
                embeddings_error = False
                with st.spinner(f"Računanje embedinga za {expected_count} {count_unit}..."):
                    # KORISTI se st.session_state.words za dobijanje embedinga
                    st.session_state.word_embeddings, embed_error_msg = get_embeddings(st.session_state.words, model)

                    if embed_error_msg:
                        st.error(f"Greška pri dobijanju embedinga: {embed_error_msg}", icon=":material/error:")
                        embeddings_error = True
                        # Resetuj rezultate na grešku
                        st.session_state.words = []
                        st.session_state.word_embeddings = None
                        st.session_state.pca_components = None
                        st.session_state.eigenvalues = None
                        st.session_state.explained_variances = None
                        st.session_state.total_variance = None
                        st.session_state.pca_error_msg = None
                        st.session_state.avg_cos_dist = None
                        st.session_state.dist_matrix = None
                        st.session_state.pairwise_distances = None

                    # Provera da li je dobijen očekivani broj embedinga
                    elif st.session_state.word_embeddings is None or st.session_state.word_embeddings.shape[0] != expected_count:
                         st.error(f"Nije dobijen očekivani broj embedinga ({expected_count}). Dobijeno {st.session_state.word_embeddings.shape[0] if st.session_state.word_embeddings is not None else 'None'}).", icon=":material/error:")
                         embeddings_error = True
                         # Resetuj rezultate na grešku
                         st.session_state.words = []
                         st.session_state.word_embeddings = None
                         st.session_state.pca_components = None
                         st.session_state.eigenvalues = None
                         st.session_state.explained_variances = None
                         st.session_state.total_variance = None
                         st.session_state.pca_error_msg = None
                         st.session_state.avg_cos_dist = None
                         st.session_state.dist_matrix = None
                         st.session_state.pairwise_distances = None


                if not embeddings_error:
                    st.divider()
                    st.subheader("Mere semantičke distance")

                    # PCA
                    st.session_state.pca_components, st.session_state.eigenvalues, st.session_state.explained_variances, st.session_state.total_variance, st.session_state.pca_error_msg = calculate_pca(st.session_state.word_embeddings)

                    if st.session_state.pca_error_msg:
                        st.error(f"Greška tokom PCA analize: {st.session_state.pca_error_msg}", icon=":material/error:")
                        # Reset PCA rezultate na grešku
                        st.session_state.pca_components = None
                        st.session_state.eigenvalues = None
                        st.session_state.explained_variances = None
                        st.session_state.total_variance = None
                    else:
                         st.success("PCA analiza uspešno izvršena.", icon=":material/check_circle:")


                    # Računanje kosinusne udaljenosti
                    # KORISTI se st.session_state.word_embeddings
                    st.session_state.avg_cos_dist, st.session_state.dist_matrix, st.session_state.pairwise_distances, cos_dist_error_msg = calculate_avg_cosine_distance(st.session_state.word_embeddings)

                    if cos_dist_error_msg:
                         st.warning(f"Upozorenje pri računanju kosinusne udaljenosti: {cos_dist_error_msg}", icon=":material/warning:")
                         st.session_state.avg_cos_dist = None
                         st.session_state.dist_matrix = None
                         st.session_state.pairwise_distances = None
                    else:
                         if st.session_state.avg_cos_dist is not None:
                             st.success("Kosinusna udaljenost uspešno izračunata.", icon=":material/check_circle:")

                    # Prikaz metrika
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if cos_dist_error_msg:
                            st.error(f"Medijana: {cos_dist_error_msg}", icon=":material/error:")
                        elif isinstance(st.session_state.pairwise_distances, pd.DataFrame) and not st.session_state.pairwise_distances.empty:
                            st.metric(label="Medijalna udaljenost", value=f"{np.median(st.session_state.pairwise_distances.iloc[:, 0]):.4f}")

                    with col2:
                        if cos_dist_error_msg:
                            st.error(f"Srednja: {cos_dist_error_msg}", icon=":material/error:")
                        elif st.session_state.avg_cos_dist is not None:
                            st.metric(label="Srednja udaljenost", value=f"{st.session_state.avg_cos_dist:.4f}")
                    with col3:
                        if cos_dist_error_msg:
                            st.error(f"Harmonijska: {cos_dist_error_msg}", icon=":material/error:")
                        elif isinstance(st.session_state.pairwise_distances, pd.DataFrame) and not st.session_state.pairwise_distances.empty:
                            st.metric(label="Harmonijska udaljenost", value=f"{hmean(st.session_state.pairwise_distances.iloc[:, 0]):.4f}")

                    # --- Knn metrike ---
                    with st.subheader("Ocene distance k najsličnijih parova"):
                        if isinstance(st.session_state.pairwise_distances, pd.DataFrame) and not st.session_state.pairwise_distances.empty:
                            # Pristup vrednostima distanci, sortirano
                            distances = st.session_state.pairwise_distances.iloc[:, 0].sort_values().reset_index(drop=True)
                            n_distances = distances.size

                            if n_distances > 0:
                                k_values = list(range(1, n_distances + 1))

                                harmonic_means = []
                                arithmetic_means = []
                                medians = []

                                for k in k_values:
                                    top_k_distances = distances.head(k)
                                    hmean_val = hmean(top_k_distances) if k > 0 and not (top_k_distances == 0).any() else np.nan
                                    harmonic_means.append(hmean_val)
                                    arithmetic_means.append(np.mean(top_k_distances))
                                    medians.append(np.median(top_k_distances))

                                plot_data = pd.DataFrame({
                                    'k': k_values,
                                    'Harmonijska sredina': harmonic_means,
                                    'Prosek': arithmetic_means,
                                    'Medijana': medians
                                })
                                fig = px.line(plot_data, x='k', y=['Harmonijska sredina','Prosek','Medijana'],
                                              title='Ocene distance k najsličnijih parova',
                                              labels={'value': 'Vrednost distance', 'k': 'Broj najsličnijih parova'})
                                fig.update_traces(mode='markers+lines')
                                st.plotly_chart(fig)
                            else:
                                st.info("Nema dovoljno parova distanci za računanje k-najsličnijih.")

                        else:
                            st.info("Podaci o parnim udaljenostima još nisu dostupni ili su nevažeći.")


                    # --- Histogram ---
                    st.divider()
                    with st.subheader("Histogram kosinusnih udaljenosti"):
                        if cos_dist_error_msg:
                            st.error(f"Histogram: {cos_dist_error_msg}", icon=":material/error:")
                        elif isinstance(st.session_state.pairwise_distances, pd.DataFrame) and not st.session_state.pairwise_distances.empty:
                            fig_histogram = px.histogram(
                                st.session_state.pairwise_distances,
                                x="Distance",
                                nbins=20,
                                range_x=[0, 1],
                                color_discrete_sequence=["#5d6e25"],
                                labels={"Distance": "Kosinusna udaljenost"},
                                title="Raspodela kosinusnih udaljenosti između parova"
                            )
                            fig_histogram.update_layout(
                                bargap=0.02,
                                xaxis=dict(
                                    tickvals=[i/10 for i in range(11)],
                                    tickformat=".1f"
                                )
                            )
                            fig_histogram.update_traces(
                                showlegend=False
                            )
                            st.plotly_chart(fig_histogram)
                        else:
                            st.info("Podaci o parnim udaljenostima za histogram nisu dostupni ili su nevažeći.")


    # --- Prikaz PCA ---
    if (st.session_state.word_embeddings is not None
        and st.session_state.pca_error_msg is None
        and isinstance(st.session_state.pca_components, np.ndarray)
        and st.session_state.pca_components.shape[0] > 0
        and len(st.session_state.words) == st.session_state.pca_components.shape[0] # Broj items mora da odgovara broju redova u pca_components
       ):
        st.divider()

        if st.session_state.explained_variances is not None and len(st.session_state.explained_variances) > 0:
            st.divider()
            with st.subheader("Varijabilitet po PCA komponentama"):

                components = [f"PC{i+1}" for i in range(len(st.session_state.explained_variances))]
                explained_var = st.session_state.explained_variances
                cumulative_var = np.cumsum(explained_var)

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=components,
                    y=explained_var,
                    name="Individualna varijansa",
                    marker_color='#5d6e25',
                    text=[f"{v:.1%}" for v in explained_var],
                    textposition='auto'
                ))
                fig.add_trace(go.Scatter(
                    x=components,
                    y=cumulative_var,
                    name="Kumulativna varijansa",
                    line=dict(color='#80431d', width=3),
                    marker=dict(size=8),
                    text=[f"{v:.1%}" for v in cumulative_var],
                    textposition="top center"
                ))
                fig.update_layout(
                    barmode='group',
                    yaxis=dict(
                        title="Objašnjena varijansa",
                        tickformat=".0%",
                        range=[0, 1.1]
                    ),
                    xaxis_title="PCA komponente",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                     title=f"Objašnjena varijansa PCA ({st.session_state.input_type})"
                )

                st.plotly_chart(fig, use_container_width=True)

        if isinstance(st.session_state.dist_matrix, np.ndarray) and st.session_state.dist_matrix.shape[0] > 1 and len(st.session_state.words) == st.session_state.dist_matrix.shape[0]:
             st.divider()
             st.subheader("Heatmap semantičke razlike")
             if st.session_state.avg_cos_dist is not None:
                cos_dist_matrix = st.session_state.dist_matrix
                df_cos_sim = pd.DataFrame(
                    cos_dist_matrix,
                    index=st.session_state.words,
                    columns=st.session_state.words
                )

                fig_heatmap = px.imshow(
                    df_cos_sim,
                    text_auto=".2f",
                    color_continuous_scale='Jet',
                    labels=dict(x=st.session_state.input_type, y=st.session_state.input_type, color="Razlika"),
                    zmin=0,
                    zmax=1
                )
                fig_heatmap.update_layout(
                    width=800,
                    height=800,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    title=f"Matrica kosinusnih udaljenosti ({st.session_state.input_type})"
                )

                fig_heatmap.update_traces(
                    hovertemplate=f"{st.session_state.input_type.capitalize()} 1: %{{y}}<br>{st.session_state.input_type.capitalize()} 2: %{{x}}<br>Udaljenost: %{{z:.2f}}<extra></extra>"
                )

                # Prikaz heatmap
                st.plotly_chart(fig_heatmap)
                # Opciono: Download matrice kao CSV
                csv = df_cos_sim.to_csv().encode('utf-8')
                st.download_button(
                    label="Preuzmi matricu udaljenosti kao CSV",
                    data=csv,
                    file_name='kosinusna_udaljenost_matrica.csv',
                    mime='text/csv'
                )
             elif st.session_state.avg_cos_dist is None and st.session_state.word_embeddings is not None:
                 st.info("Nije moguće prikazati Heatmap jer kosinusna udaljenost nije uspešno izračunata.", icon=":material/info:")
             elif st.session_state.word_embeddings is not None and st.session_state.word_embeddings.shape[0] < 2:
                 st.info("Nema dovoljno stavki (potrebno >1) za prikaz Heatmap matrice.", icon=":material/info:")


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
* Seaborn
""") # Dodat Seaborn

st.sidebar.header("Napomene")
st.sidebar.markdown("""
* Model LaBSE je razvijen da podržava više jezika i rečenice. Rezultati za pojedinačne reči mogu biti manje precizni u poređenju sa modelima treniranim specifično za reči ili srpski jezik.
* Unos rečenica završavajte tačkom (`.`) za ispravno razdvajanje.
* Ošišana latinica može uticati na rezultate modela.
""")

# Primeri (ostavljeni kao komentari)
# Velika slova ignorise ali moram da sredim i osisanu latinicu koja vidno utice na rezultate.

# Primeri za reči:
# bicikl ananas oblak skalamerija sočivo melanholija inicijali interpunkcija godišnjica kreda

# Primeri za reči:
# tegla poklopac staklo turšija krastavčići krastavčić krastavac kiselo kupus čep

# Primeri za rečenice:
# Volim da jedem jabuke. Moja omiljena boja je plava. Danas je sunčan dan. Sutra idem u šetnju. Ovo je primer rečenice.