import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import hmean
import pandas as pd
from sentence_transformers import SentenceTransformer
import plotly.express as px
from openai import OpenAI

# --- OpenAI klijent (API ključ preko Streamlit secrets ili placeholder) ---
client = OpenAI(api_key=st.secrets.get("openai", {}).get("api_key", "YOUR_OPENAI_API_KEY"))

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
if 'avg_cos_dist' not in st.session_state:
    st.session_state.avg_cos_dist = None
if 'dist_matrix' not in st.session_state:
    st.session_state.dist_matrix = None
if 'pairwise_distances' not in st.session_state:
    st.session_state.pairwise_distances = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = "LaBSE"

# --- Učitavanje LaBSE modela ---
@st.cache_resource
def load_labse_model():
    try:
        model_name = "sentence-transformers/LaBSE"
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Greška pri učitavanju modela '{model_name}': {e}")
        return None

labse_model = load_labse_model()
if labse_model is not None:
    st.sidebar.success("LaBSE model uspešno učitan.")

# --- Embeding funkcije ---
@st.cache_data(show_spinner=False)
def get_embeddings_labse(items, _model):
    if not _model:
        return None, "Model nije dostupan."
    if not items:
        return np.array([]), None
    try:
        embeddings = _model.encode(items, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings, None
    except Exception as e:
        return None, f"Greška pri generisanju embeddinga: {e}"

@st.cache_data(show_spinner=False)
def get_embeddings_openai(items, model_name="text-embedding-3-small"):
    try:
        embeddings = []
        for item in items:
            response = client.embeddings.create(
                model=model_name,
                input=item
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings), None
    except Exception as e:
        return None, f"Greška pri generisanju embeddinga (OpenAI): {e}"

# --- Funkcija za računanje prosečne kosinusne distance ---
def calculate_avg_cosine_distance(embeddings):
    if embeddings is None or embeddings.shape[0] < 2:
        return None, None, None, "Potrebno je bar 2 embeddinga za računanje kosinusne udaljenosti."
    try:
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = 1.0 - sim_matrix

        n = dist_matrix.shape[0]
        upper_triangle_indices = np.triu_indices(n, k=1)
        pairwise_distances_values = dist_matrix[upper_triangle_indices].ravel()

        pairwise_distances_df = pd.DataFrame(pairwise_distances_values, columns=['Distance'])
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

# --- Izbor embeddera ---
st.session_state.embedder = st.radio(
    "Izaberite embedder:",
    ("LaBSE", "OpenAI 3-small", "OpenAI 3-large"),
    index=["LaBSE", "OpenAI 3-small", "OpenAI 3-large"].index(st.session_state.embedder),
    horizontal=True
)

# --- Dinamička uputstva ---
if st.session_state.input_type == "Reči":
    st.markdown("Unesite tačno **10 reči** (poželjno imenica u nominativu jednine na srpskom jeziku).")
    expected_count = 10
    count_unit = "reči"
else:
    st.markdown("Unesite tačno **5 rečenica** na srpskom jeziku.")
    expected_count = 5
    count_unit = "rečenica"

# --- Unos ---
st.subheader(f"Unesite {expected_count} {count_unit}:")
input_items = []
cols = st.columns(2)
for i in range(expected_count):
    with cols[i % 2]:
        label = f"Reč {i+1}:" if st.session_state.input_type == "Reči" else f"Rečenica {i+1}:"
        placeholder = f"Unesite reč {i+1}" if st.session_state.input_type == "Reči" else f"Unesite rečenicu {i+1}"
        item = st.text_input(label, key=f"item_{i}", placeholder=placeholder)
        input_items.append(item)

# --- Analiza ---
if st.button("Analiza", type="primary", icon=":material/psychology_alt:"):
    st.session_state.word_embeddings = None
    st.session_state.avg_cos_dist = None
    st.session_state.dist_matrix = None
    st.session_state.pairwise_distances = None
    st.session_state.words = []

    if not all(input_items):
        st.error(f"Molimo popunite sva {expected_count} polja.", icon=":material/error:")
    else:
        if st.session_state.input_type == "Reči":
            st.session_state.words = [item.strip().lower() for item in input_items if item.strip()]
        else:
            st.session_state.words = [item.strip() for item in input_items if item.strip()]

        if len(st.session_state.words) < 2:
            st.error(f"Potrebno je bar 2 {count_unit} za analizu.", icon=":material/error:")
        else:
            st.info(f"Unete {st.session_state.input_type}: {', '.join(st.session_state.words)}", icon=":material/psychology:")

            embeddings_error = False
            with st.spinner(f"Računanje embeddinga za {len(st.session_state.words)} {count_unit}..."):
                if st.session_state.embedder == "LaBSE":
                    st.session_state.word_embeddings, embed_error_msg = get_embeddings_labse(st.session_state.words, labse_model)
                elif st.session_state.embedder == "OpenAI 3-small":
                    st.session_state.word_embeddings, embed_error_msg = get_embeddings_openai(st.session_state.words, "text-embedding-3-small")
                elif st.session_state.embedder == "OpenAI 3-large":
                    st.session_state.word_embeddings, embed_error_msg = get_embeddings_openai(st.session_state.words, "text-embedding-3-large")

                if embed_error_msg:
                    st.error(f"Greška pri dobijanju embeddinga: {embed_error_msg}", icon=":material/error:")
                    embeddings_error = True

            if not embeddings_error:
                st.divider()
                st.subheader("Mere semantičke distance")

                st.session_state.avg_cos_dist, st.session_state.dist_matrix, st.session_state.pairwise_distances, cos_dist_error_msg = calculate_avg_cosine_distance(st.session_state.word_embeddings)

                if cos_dist_error_msg:
                    st.error(f"Greška pri računanju kosinusne udaljenosti: {cos_dist_error_msg}", icon=":material/error:")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Medijalna udaljenost", f"{np.median(st.session_state.pairwise_distances.iloc[:,0]):.4f}")
                    with col2:
                        st.metric("Srednja udaljenost", f"{st.session_state.avg_cos_dist:.4f}")
                    with col3:
                        st.metric("Harmonijska udaljenost", f"{hmean(st.session_state.pairwise_distances.iloc[:,0]):.4f}")

# --- Prikaz heatmap matrice ---
if (st.session_state.word_embeddings is not None and
    isinstance(st.session_state.dist_matrix, np.ndarray) and
    st.session_state.dist_matrix.shape[0] > 1 and
    len(st.session_state.words) == st.session_state.dist_matrix.shape[0]):

    st.divider()
    st.subheader("Matrica semantičke distance")
    df_cos_sim = pd.DataFrame(
        st.session_state.dist_matrix,
        index=st.session_state.words,
        columns=st.session_state.words
    )

    fig_heatmap = px.imshow(
        df_cos_sim,
        text_auto=".2f",
        color_continuous_scale='ice',
        aspect='equal',
        labels=dict(x=st.session_state.input_type, y=st.session_state.input_type, color="Udaljenost"),
        zmin=0,
        zmax=1
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    csv = df_cos_sim.to_csv().encode('utf-8')
    st.download_button(
        label="Preuzmi matricu udaljenosti kao CSV",
        data=csv,
        file_name='semanticka_udaljenost_matrica.csv',
        mime='text/csv'
    )