import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
import scipy.stats # Za entropiju
import networkx as nx

# Ignoriši upozorenja koja mogu doći iz transformers biblioteke
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Primeri
# kompjuter mis tastatura ekran kabal dugme cd program internet slusalice
# kompjuter miš tastatura ekran kabal dugme cd program internet slušalice
# šrafciger išijas grananje skalamerija sočivo melanholija inicijali interpunkcija ravnodnevnica promaja

# --- Konfiguracija Stranice ---
st.set_page_config(
    page_title="Test kreativnosti",
    page_icon="🇷🇸",
    layout="wide"
)

# --- Učitavanje Modela i Tokenizera (Keširano radi brzine) ---
@st.cache_resource
def load_model_and_tokenizer():
    """Učitava Bertić model i tokenizer sa Hugging Face."""
    try:
        model_name = "classla/bcms-bertic"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        st.success(f"Model '{model_name}' i tokenizer uspešno učitani.", icon="✅")
        return tokenizer, model
    except Exception as e:
        st.error(f"Greška pri učitavanju modela '{model_name}': {e}")
        st.error("Proverite internet konekciju i da li je naziv modela tačan.")
        return None, None

tokenizer, model = load_model_and_tokenizer()

# --- Funkcija za Dobijanje Embedinga ---
@st.cache_data(show_spinner=False)
def get_word_embeddings(words, _model, _tokenizer):
    """Generiše embedinge za listu reči koristeći dati model i tokenizer."""
    embeddings = []
    if not _model or not _tokenizer:
        return None, "Model ili tokenizer nisu dostupni."
    try:
        with torch.no_grad():
            for word in words:
                inputs = _tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = _model(**inputs)
                word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                if word_embedding.ndim == 0: # Handle potential scalar output edge case
                     return None, f"Greška: Dobijen skalar umesto vektora za reč '{word}'."
                embeddings.append(word_embedding.cpu().numpy()) # Prebaci na CPU pre konverzije u numpy
        # Provera da li su svi embedinzi iste dimenzije
        if len(embeddings) > 0:
            first_shape = embeddings[0].shape
            if not all(emb.shape == first_shape for emb in embeddings):
                 return None, "Greška: Nisu svi generisani embedinzi iste dimenzije."
        return np.array(embeddings), None
    except Exception as e:
        st.error(f"Neočekivana greška pri generisanju embedinga: {e}") # Prikazi grešku u UI
        return None, f"Greška pri generisanju embedinga za neku od reči."


# --- Funkcija za PCA Analizu ---
def calculate_pca_components(embeddings):
    """Vrši PCA analizu i vraća eigenvalue i objašnjene varijanse."""
    if embeddings is None or embeddings.shape[0] < 2:
        return None, None, "Potrebno je bar 2 validna embedinga za PCA analizu."

    # Proveri da li postoji varijansa u podacima
    if np.allclose(embeddings, embeddings[0,:], atol=1e-6):
         return None, None, "Svi embedinzi su skoro identični. PCA nije smislena."

    # Odredi broj komponenti
    n_samples, n_features = embeddings.shape
    n_components = min(n_samples, n_features) # Maksimalan broj komponenti

    # Smanji broj komponenti ako je varijansa niska (da izbegneš skoro nula eigenvalue)
    # Ovo je naprednija optimizacija, za sada računamo sve
    # pca_full = PCA(n_components=n_components_max)
    # pca_full.fit(embeddings)
    # n_components = np.sum(pca_full.explained_variance_ > 1e-9) # Broj komponenti sa varijansom > epsilon
    if n_components < 1:
          return None, None, "Nije moguće izračunati komponente (nema dovoljno varijanse ili podataka)."

    try:
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)

        eigenvalues = pca.explained_variance_
        explained_variance_ratios = pca.explained_variance_ratio_

        # Filtriraj komponente sa zanemarljivom varijansom (manje od npr. 0.01%)
        valid_indices = explained_variance_ratios > 1e-5
        if not np.any(valid_indices): # Ako nema validnih, vrati prvu? Ili grešku?
             if len(eigenvalues)>0: # Vrati bar prvu ako postoji
                  valid_indices = [0] # Default na prvu
             else:
                  return None, None, "Ni jedna komponenta nema značajnu varijansu."

        eigenvalues = eigenvalues[valid_indices]
        explained_variance_ratios = explained_variance_ratios[valid_indices]


        return eigenvalues, explained_variance_ratios, None
    except Exception as e:
        return None, None, f"Greška tokom PCA analize: {e}"

# --- Funkcija za Računanje Entropije Eigenvalue ---
def calculate_eigenvalue_entropy(explained_variance_ratios):
    """Računa normalizovanu Shannon entropiju raspodele objašnjene varijanse."""
    if explained_variance_ratios is None or len(explained_variance_ratios) < 1:
        return 0.0 # Nema entropije ako nema komponenti

    # Filtriraj nule i negativne vrednosti (iako ne bi trebalo da budu negativne)
    ratios = np.array(explained_variance_ratios)
    ratios = ratios[ratios > 1e-10] # Koristi mali epsilon

    if len(ratios) == 0:
         return 0.0 # Nema entropije ako nema pozitivnih odnosa
    if len(ratios) == 1:
         return 0.0 # Entropija jedne tačke je 0

    # Normalizuj odnose da suma bude 1 (za slučaj da smo filtrirali neke)
    ratios = ratios / ratios.sum()

    # Izračunaj entropiju
    entropy = scipy.stats.entropy(ratios, base=2)

    # Normalizuj entropiju na opseg [0, 1]
    max_entropy = np.log2(len(ratios))
    if max_entropy <= 1e-10: # Izbegavanje deljenja nulom ako je len(ratios) == 1
        return 0.0
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


# --- Funkcija za Računanje Prosečne Kosinusne Udaljenosti ---
def calculate_avg_cosine_distance(embeddings):
    """Računa prosečnu kosinusnu udaljenost između svih parova embedinga."""
    if embeddings is None or embeddings.shape[0] < 2:
        return 0.0 # Nema udaljenosti ako nema bar 2 tačke

    try:
        # Izračunaj matricu kosinusne sličnosti
        sim_matrix = cosine_similarity(embeddings)

        # Pretovri u matricu udaljenosti (1 - similarity)
        dist_matrix = 1.0 - sim_matrix

        # Uzmi samo gornji trougao matrice bez dijagonale
        # (jer je matrica simetrična i udaljenost tačke od same sebe je 0)
        n = dist_matrix.shape[0]
        upper_triangle_indices = np.triu_indices(n, k=1)
        pairwise_distances = dist_matrix[upper_triangle_indices]

        # Izračunaj prosek
        if len(pairwise_distances) == 0:
            return 0.0 # Slučaj kada imamo samo jednu reč (iako provera na početku to sprečava)
        avg_distance = np.mean(pairwise_distances)
        return avg_distance

    except Exception as e:
         st.error(f"Greška pri računanju kosinusne udaljenosti: {e}")
         return None # Vrati None u slučaju greške

# --- Funkcija za Računanje Prosečne Kosinusne Udaljenosti ---
def calculate_distance_and_plot_similarity(embeddings, plot_graph=True, similarity_threshold=0.0):
    """
    Računa prosečnu kosinusnu udaljenost između svih parova embedinga
    i opciono prikazuje graf sličnosti.

    Argumenti:
        embeddings (np.array): NumPy niz gde svaki red predstavlja jedan vektor (embeding).
        plot_graph (bool): Ako je True, prikazuje graf sličnosti pomoću NetworkX.
        similarity_threshold (float): Minimalna sličnost da bi se ivica prikazala u grafu.
                                       Podrazumevano je 0.0 (prikazuje sve veze).

    Vraća:
        float: Prosečnu kosinusnu udaljenost, ili None u slučaju greške.
               Ako plot_graph=True, takođe prikazuje graf.
    """
    if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] < 2:
        print("Greška: Ulaz mora biti NumPy niz sa bar dva reda (vektora).")
        # Vraćamo 0.0 za prosečnu distancu kao u originalnoj funkciji, mada bi None ili error bio bolji
        # Ali da ostanemo konzistentni sa originalnim kodom za return vrednost u ovom slučaju.
        # Graf se neće crtati.
        if embeddings is not None and embeddings.shape[0] == 1: return 0.0
        return None # Za None ili nevalidan tip

    try:
        num_vectors = embeddings.shape[0]

        # 1. Izračunaj matricu kosinusne sličnosti
        sim_matrix = cosine_similarity(embeddings)

        # 2. Pretovri u matricu udaljenosti (1 - similarity)
        dist_matrix = 1.0 - sim_matrix

        # 3. Izračunaj prosečnu udaljenost (kao u originalnoj funkciji)
        upper_triangle_indices = np.triu_indices(num_vectors, k=1)
        pairwise_distances = dist_matrix[upper_triangle_indices]

        avg_distance = 0.0
        if pairwise_distances.size > 0:
            avg_distance = np.mean(pairwise_distances)
        # else: avg_distance ostaje 0.0 (slučaj sa samo jednim vektorom tehnički ne stiže ovde zbog provere na početku)


        # --- DODATO: Crtanje grafa pomoću NetworkX ---
        if plot_graph:
            try:
                G = nx.Graph()
                # Dodaj čvorove (predstavljaju indekse vektora)
                G.add_nodes_from(range(num_vectors))

                # Dodaj ivice sa težinom jednakom sličnosti, ako je iznad praga
                max_similarity_for_scaling = 0.0
                edges_to_add = []
                for i in range(num_vectors):
                    for j in range(i + 1, num_vectors):
                        similarity = sim_matrix[i, j]
                        if similarity >= similarity_threshold:
                             edges_to_add.append((i, j, {'weight': similarity}))
                             if similarity > max_similarity_for_scaling:
                                 max_similarity_for_scaling = similarity

                G.add_edges_from(edges_to_add)

                # Ako nema ivica (npr. visok threshold), ipak prikaži čvorove
                if not G.edges() and not G.nodes():
                     print("Nema čvorova ili ivica za prikazivanje grafa (proverite threshold).")

                else:
                    plt.figure(figsize=(8, 8))
                    # Pozicioniraj čvorove (može se eksperimentisati sa layoutima: spring_layout, circular_layout,...)
                    pos = nx.spring_layout(G, seed=42) # seed za reproduktivnost

                    # Izdvoji težine za debljinu ivica
                    # Skaliramo debljinu radi bolje vizualizacije (npr. * 5)
                    # Pazimo na deljenje nulom ako je max_similarity_for_scaling 0
                    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                    if max_similarity_for_scaling > 0:
                         edge_widths = [(w / max_similarity_for_scaling) * 5 + 0.5 for w in edge_weights] # Skaliranje + minimalna debljina
                    else:
                         edge_widths = [1 for _ in edge_weights] # Default debljina ako nema pozitivnih sličnosti


                    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue')
                    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.7)
                    nx.draw_networkx_labels(G, pos, font_size=10)

                    plt.title(f"Graf sličnosti vektora (Prag = {similarity_threshold:.2f})")
                    plt.axis('off') # Isključi ose
                    plt.show()

            except Exception as graph_e:
                 # Koristimo print umesto st.error jer ne znamo da li se koristi Streamlit
                 print(f"Greška pri generisanju ili prikazivanju grafa: {graph_e}")
        # --- KRAJ DODATOG DELA ---

        return avg_distance

    except Exception as e:
        print(f"Greška pri računanju kosinusne udaljenosti/sličnosti: {e}")
        return None # Vrati None u slučaju greške



# --- Streamlit Interfejs ---
st.title("Analiza različitosti reči")
st.markdown("""
Unesite tačno 10 reči na srpskom jeziku, razdvojenih razmakom ili svaku u novom redu.
""")

# Provera da li je model uspešno učitan
if tokenizer is None or model is None:
    st.warning("Model nije uspešno učitan. Aplikacija ne može da nastavi sa radom.", icon="⚠️")
else:
    # Unos teksta od korisnika
    input_text = st.text_area("Unesite 10 reči:", height=150, placeholder="Primer: kuća drvo reka sunce nebo čovek knjiga misao ljubav sreća")

    # Dugme za pokretanje analize
    if st.button("Izračunaj Različitost", type="primary"):
        if not input_text.strip():
            st.warning("Molimo unesite reči.", icon="⚠️")
        else:
            # Obrada unosa
            words = [word.strip() for word in input_text.split() if word.strip()]

            if len(words) != 10:
                st.error(f"Uneto je {len(words)} reči. Molimo unesite tačno 10 reči.", icon="❌")
            else:
                st.info(f"Unete reči: {', '.join(words)}", icon="📄")

                # Generisanje Embedinga
                with st.spinner("Računam embedinge reči pomoću Bertić modela..."):
                    word_embeddings, embed_error_msg = get_word_embeddings(words, model, tokenizer)

                if embed_error_msg:
                    st.error(f"Greška u embedinzima: {embed_error_msg}", icon="❌")
                elif word_embeddings is not None:
                    st.success("Embedinzi uspešno izračunati!", icon="✅")
                    st.markdown(f"Dimenzije matrice embedinga: `{word_embeddings.shape}`")

                    # --- Izračunavanje i Prikaz Metrika Različitosti ---
                    st.divider()
                    st.subheader("Mere Semantičke Različitosti")

                    col1, col2, col3 = st.columns(3) # Podeli prostor za metrike

                    # 1. Metrika: Entropija Eigenvalue (PCA)
                    with col1:
                        with st.spinner("Vrsim PCA analizu..."):
                            eigenvalues, explained_variances, pca_error_msg = calculate_pca_components(word_embeddings)

                        if pca_error_msg:
                            st.error(f"PCA greška: {pca_error_msg}", icon="❌")
                            st.metric(label="Normalizovana Entropija Eigenvalue (PCA)", value="N/A")
                        elif explained_variances is not None:
                            entropy_score = calculate_eigenvalue_entropy(explained_variances)
                            st.metric(label="Normalizovana Entropija Eigenvalue (PCA)", value=f"{entropy_score:.4f}")
                            st.caption("Bliže 1 = Veća multidim. različitost")
                        else:
                             st.metric(label="Normalizovana Entropija Eigenvalue (PCA)", value="N/A")


                    # 2. Metrika: Prosečna Kosinusna Udaljenost
                    with col2:
                         with st.spinner("Računam prosečnu kosinusnu udaljenost..."):
                              avg_cos_dist = calculate_avg_cosine_distance(word_embeddings)

                         if avg_cos_dist is None: # Provera za None u slučaju greške u funkciji
                              st.metric(label="Prosečna Kosinusna Udaljenost", value="Greška")
                              st.caption("Problem pri računanju.")
                         else:
                              st.metric(label="Prosečna Kosinusna Udaljenost", value=f"{avg_cos_dist:.4f}")
                              st.caption("Bliže 1 (max 2) = Veća prosečna različitost")

                    # 3. Metrika: Prosecna kosinusna udaljenost i graf
                    with col3:
                        with st.spinner("Računam prosečnu kosinusnu udaljenost..."):
                              cos_dist = calculate_distance_and_plot_similarity(word_embeddings)

                        if cos_dist is None: # Provera za None u slučaju greške u funkciji
                              st.metric(label="Prosečna Kosinusna Udaljenost 2", value="Greška")
                              st.caption("Problem pri računanju.")
                        else:
                              st.metric(label="Prosečna Kosinusna Udaljenost", value=f"{cos_dist:.4f}")
                              st.caption("Bliže 1 (max 2) = Veća prosečna različitost")


                    # --- Prikaz Detalja PCA (ako je uspešno) ---
                    if not pca_error_msg and eigenvalues is not None and explained_variances is not None:
                        st.divider()
                        st.subheader("Detalji PCA Analize")

                        # Prikaz tabele sa vrednostima
                        components_data = {
                            "Komponenta": [f"PC{i+1}" for i in range(len(eigenvalues))],
                            "Sopstvena Vrednost (Eigenvalue)": eigenvalues,
                            "Objašnjena Varijansa (%)": [f"{var:.2%}" for var in explained_variances]
                        }
                        st.dataframe(components_data, use_container_width=True)

                        # Prikaz Scree Plota
                        try: # Dodatni try-except za crtanje
                            fig, ax = plt.subplots()
                            ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
                            ax.set_xlabel("Glavna Komponenta")
                            ax.set_ylabel("Sopstvena Vrednost (Eigenvalue)")
                            ax.set_title("Scree Plot")
                            # Podesi X osu da prikazuje cele brojeve
                            ax.set_xticks(np.arange(1, len(eigenvalues) + 1))
                            # Opciono: log skala ako su vrednosti veoma različite
                            # ax.set_yscale('log')
                            st.pyplot(fig)
                            st.caption("*Grafikon sopstvenih vrednosti. Brži pad ukazuje na manju suštinsku dimenzionalnost (manju različitost).*")
                        except Exception as plot_err:
                            st.warning(f"Nije moguće nacrtati Scree plot: {plot_err}")

                    # --- Prikaz Grafa (ako je uspešno) ---
                    if not pca_error_msg and eigenvalues is not None and explained_variances is not None:
                        st.divider()
                        st.subheader("Graf sličnosti reči")
                        # Prikaz Grafa
                        try: # Dodatni try-except za crtanje
                            fig, ax = plt.subplots()
                            ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
                            ax.set_title("Scree Plot")
                            # Podesi X osu da prikazuje cele brojeve
                            ax.set_xticks(np.arange(1, len(eigenvalues) + 1))
                            # Opciono: log skala ako su vrednosti veoma različite
                            # ax.set_yscale('log')
                            st.pyplot(fig)
                            st.caption("*Grafikon sopstvenih vrednosti. Brži pad ukazuje na manju suštinsku dimenzionalnost (manju različitost).*")
                        except Exception as plot_err:
                            st.warning(f"Nije moguće nacrtati graf: {plot_err}")

                else:
                     st.error("Došlo je do nepoznate greške pri generisanju embedinga.", icon="❌")


# --- Dodatne Informacije ---
st.sidebar.header("Info")
st.sidebar.markdown("""
Ova aplikacija koristi model za embedinge i izračunava pet metrika semantičke različitosti za 10 unetih reči:
- **Entropija Eigenvalue (PCA):** Meri ravnomernost raspodele varijanse.
- **Prosečna kosinusna distanca:** Meri prosečnu udaljenost parova reči.
- **Varijacija vektora**
- **Euklidska distanca**
""")
st.sidebar.header("Embeder")
st.sidebar.markdown("[classla/bcms-bertic](https://huggingface.co/classla/bcms-bertic)")
st.sidebar.header("Biblioteke")
st.sidebar.markdown("""
* Streamlit
* Transformers (Hugging Face)
* PyTorch
* Scikit-learn (PCA, cosine_similarity)
* NumPy
* SciPy (stats.entropy)
""")