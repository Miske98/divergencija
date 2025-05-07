import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
import scipy.stats
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import normalize

# --- Ignoriši upozorenja ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Konfiguracija Stranice ---
st.set_page_config(
    page_title="Test divergentnog razmišljanja",
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


# --- Učitavanje Modela i Tokenizera ---
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model_name = "classla/bcms-bertic"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Greška pri učitavanju modela '{model_name}': {e}")
        st.error("Proverite internet konekciju i da li je naziv modela tačan.")
        return None, None

tokenizer, model = load_model_and_tokenizer()
if model is not None:
     st.sidebar.success("Model 'Bertić' uspešno učitan.")

# --- Funkcija za Dobijanje Embedinga ---
@st.cache_data(show_spinner=False)
def get_word_embeddings(words, _model, _tokenizer):
    embeddings = []
    if not _model or not _tokenizer:
        return None, "Model ili tokenizer nisu dostupni."
    if not words:
         return np.array([]), None

    try:
        with torch.no_grad():
            for word in words:
                if not word: continue
                inputs = _tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = _model(**inputs)
                word_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                if word_embedding.ndim == 0:
                    return None, f"Greška: Dobijen skalar umesto vektora za reč '{word}'."
                embeddings.append(word_embedding.cpu().numpy())
        if not embeddings:
             return np.array([]), "Nije generisan nijedan validan embedding."

        embeddings_array = np.array(embeddings)
        if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len([w for w in words if w]):
             return None, "Greška u dimenzijama generisanih embedinga."

        return embeddings_array, None
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

# --- Funkcija za Računanje Prosečne Kosinusne Udaljenosti ---
def calculate_avg_cosine_distance(embeddings):
    if embeddings is None or embeddings.shape[0] < 2: 
        return None, "Potrebno je bar 2 embedinza za računanje kosinusne udaljenosti"
    try:
        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = 1.0 - sim_matrix
        n = dist_matrix.shape[0]
        upper_triangle_indices = np.triu_indices(n, k=1)
        pairwise_distances = dist_matrix[upper_triangle_indices]
        avg = np.mean(pairwise_distances) if pairwise_distances.size > 0 else 0.0
        return avg, None
    except Exception as e:
        return None, f"Greška kod kosinusne udaljenosti: {e}"

# --- Streamlit Interfejs ---
st.title("Test divergentnog razmišljanja")
st.markdown("""
Unesite tačno 10 imenica na srpskom jeziku, razdvojenih razmakom ili svaku u novom redu.
""")

# Provera da li je model uspešno učitan
if tokenizer is None or model is None:
    st.warning("Model nije uspešno učitan. Aplikacija ne može da nastavi sa radom.", icon="⚠️")
else:
    # Unos teksta od korisnika
    input_text = st.text_area("Reči:", height=150, placeholder="Budite kreativni.")

    # Dugme za pokretanje analize
    if st.button("Analiza", type="primary", icon= ":material/psychology_alt:"):
        if not input_text.strip():
            st.warning("Molimo unesite reči.", icon="⚠️")
        else:
            # Obrada unosa
            st.session_state.words = [word.strip().lower() for word in input_text.split() if word.strip()]

            if len(st.session_state.words) != 10:
                st.error(f"Uneto je {len(st.session_state.words)} reči. Molimo unesite tačno 10 reči.", icon="❌")
            else:
                st.info(f"Reči: {', '.join(st.session_state.words)}", icon=":material/psychology:")

                # Generisanje Embedinga
                embeddings_error = False
                with st.spinner("Računanje embedinga..."):
                    st.session_state.word_embeddings, embed_error_msg = get_word_embeddings(st.session_state.words, model, tokenizer)
                    if embed_error_msg:
                        st.error(f"Greška pri dobijanju embedinga: {embed_error_msg}", icon="❌")
                        embeddings_error = True
                    elif st.session_state.word_embeddings is None or st.session_state.word_embeddings.shape[0] != 10:
                        st.error("Nije dobijen očekivani broj embedinga (10).", icon="❌")
                        embeddings_error = True

                # Nastavi samo ako nema greške u embedinzima
                if not embeddings_error:
                    st.success(f"Embedinzi uspešno izračunati! Dimenzije: `{st.session_state.word_embeddings.shape}`", icon=":material/done_outline:")

                    # --- Izračunavanje i Prikaz Metrika Različitosti ---
                    st.divider()
                    st.subheader("Mere semantičke različitosti")

                    # PCA Analiza
                    with st.spinner("PCA analiza..."):
                        pca_components, st.session_state.eigenvalues, st.session_state.explained_variances, st.session_state.total_variance, st.session_state.pca_error_msg = calculate_pca(st.session_state.word_embeddings)
                    
                    # Kosinusna udaljenost
                    with st.spinner("Računanje kosinusne udaljenosti..."):
                        st.session_state.avg_cos_dist, cos_err = calculate_avg_cosine_distance(st.session_state.word_embeddings)
                    
                    # Prikaz metrika
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.session_state.pca_error_msg:
                            st.error(f"PCA: {st.session_state.pca_error_msg}", icon="❌")
                        elif st.session_state.explained_variances is not None:
                            st.metric(label="Ukupna varijansa (PCA)", value=f"{st.session_state.total_variance:.4f}")
                            st.caption("Veća vrednost ≈ veći ukupan 'spread' podataka.")

                    with col2:
                        if cos_err:
                            st.error(f"Kosinus: {cos_err}", icon="❌")
                        elif st.session_state.avg_cos_dist is not None:
                            st.metric(label="Prosečna cos udaljenost", value=f"{st.session_state.avg_cos_dist:.4f}")
                            st.caption("Veća vrednost ≈ veća prosečna semantička različitost.")

                    with col3:
                        if st.session_state.pca_error_msg:
                            st.metric(label="Objašnjena varijansa (PC1)", value="N/A")
                        elif st.session_state.explained_variances is not None and len(st.session_state.explained_variances) > 0:
                            st.metric(label="Objašnjena varijansa (PC1)", value=f"{st.session_state.explained_variances[0]:.2%}")
                            st.caption("Opisan varijabilitet treba biti što raspršeniji po komponentama")

    # --- Prikaz PCA 3D scatter plot i scree plot ---
    if st.session_state.word_embeddings is not None and len(st.session_state.words) == 10 and st.session_state.pca_error_msg is None and pca_components is not None:
        st.divider()
        st.subheader("Vizuelizacija PCA komponenti")
    
        # Napravi DataFrame za plot (koristimo prve 3 PCA komponente)
        pca_df = pd.DataFrame({
            'Reč': st.session_state.words,
            'PC1': pca_components[:, 0],
            'PC2': pca_components[:, 1], 
            'PC3': pca_components[:, 2] if pca_components.shape[1] > 2 else np.zeros(len(pca_components))
        })
    
        # Kreiraj Plotly 3D scatter plot
        fig_3d = px.scatter_3d(
            pca_df,
            x='PC1',
            y='PC2',
            z='PC3',
            text='Reč',
            title='3D prikaz reči',
            labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3'},
            hover_name='Reč'
        )
        
        # Podesi izgled 3D plota
        fig_3d.update_traces(
            marker=dict(size=8),
            textposition='top center'
        )
        
        # Kreiraj scree plot
        if st.session_state.eigenvalues is not None:
            scree_fig, ax = plt.subplots(figsize=(6, 4))
            
            ax.plot(range(1, len(st.session_state.eigenvalues) + 1), 
                    st.session_state.eigenvalues, 
                    marker='o', 
                    linestyle='-')
            
            ax.set_xlabel("Glavna komponenta")
            ax.set_ylabel("Sopstvena vrednost")
            ax.set_title("Scree plot")
            ax.set_xticks(np.arange(1, len(st.session_state.eigenvalues) + 1))
            plt.tight_layout()
        
        # Prikaz u dve kolone
        col1, col2 = st.columns([3, 2])
            
        with col1:
            st.plotly_chart(fig_3d, use_container_width=True)
            st.caption(f"Prikaz uz zadržavanje {np.sum(st.session_state.explained_variances[:3]):.1%} varijabiliteta. Reči koje su bliže u ovom prostoru su semantički sličnije.")
            
        with col2:
            if st.session_state.eigenvalues is not None:
                st.pyplot(scree_fig)
                st.caption("Scree plot pokazuje važnost svake glavne komponente. Veće vrednosti znače veću varijansu koju komponenta objašnjava.")
                plt.close(scree_fig)

        # --- Prikaz PCA informacija ---
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
                
                st.caption("""
                *Tabela prikazuje varijabilitet koji objašnjava svaka od glavnih komponenti:
                - **Objašnjena varijansa**: procenat ukupne varijanse koji objašnjava svaka komponenta
                - **Kumulativna varijansa**: akumulirani procenat objašnjene varijanse*
                """)
# --- Dodatne Informacije ---
st.sidebar.header("Info")
st.sidebar.header("Model Embedinga")
st.sidebar.markdown("[classla/bcms-bertic](https://huggingface.co/classla/bcms-bertic)")
st.sidebar.header("Korišćene biblioteke")
st.sidebar.markdown("""
* Streamlit
* Transformers (Hugging Face)
* PyTorch
* Scikit-learn
* NumPy
* SciPy
* Matplotlib
* Pandas
""")
# Primeri

# Velika slova ignorise ali moram da sredim i osisanu latinicu koja vidno utice na rezultate.

# kompjuter mis tastatura ekran kabal dugme cd program internet slusalice -- cos 0.0262 | var 8.9902 |
                                                
# kompjuter miš tastatura ekran kabal dugme cd program internet slušalice -- cos 0.0188 | var 7.3553 |

# srafciger isijas grananje skalamerija socivo melanholija inicijali interpunkcija ravnodnevnica promaja -- cos 0.0505 | var 10.1457 |

# šrafciger išijas grananje skalamerija sočivo melanholija inicijali interpunkcija ravnodnevnica promaja -- cos 0.0534 | var 10.6414 |

# dete pelena vrtić škola klupa užina bojanka mama noša vaspitačica -- cos 0.0207 | var 7.4715 |

# otac sin majka deda baba ćerka brat sestra stric ujak -- cos 0.0127 | var 5.9134 |

# sin sinovljev sinčić sine dečak dete otac tata mama majka -- cos 0.0167 | var 7.0236 |

# duh duhovnost religija crkva biblija postulat vera bog otac praznik -- cos 0.0152 | var 5.7820 |

# necu nemam nisam nemoj ja ti mi vi oni ovi -- cos 0.0198 | var 6.2928 |