import streamlit as st
import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from openai import OpenAI

# Pokušaj uvoza skbio, uz "fallback" poruku
try:
    from skbio.stats.distance import permanova, DistanceMatrix
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False

# --- KONFIGURACIJA ---
client = OpenAI(api_key=st.secrets.get("openai", {}).get("api_key", "VAŠ_KLJUČ"))

st.set_page_config(page_title="Semantički Objektiv SOTA", layout="wide")

@st.cache_resource
def load_labse(): return SentenceTransformer("sentence-transformers/LaBSE")
labse_model = load_labse()

def get_embeddings(items, provider):
    if provider == "LaBSE":
        return labse_model.encode(items, convert_to_numpy=True, normalize_embeddings=True), None
    try:
        model_nm = "text-embedding-3-small" if "small" in provider else "text-embedding-3-large"
        res = client.embeddings.create(input=items, model=model_nm)
        return np.array([r.embedding for r in res.data]), None
    except Exception as e: return None, str(e)

def compute_thresholds(emb):
    N, T = emb.shape
    q = T / N
    l_max = (1 + np.sqrt(1/q))**2
    rnd_data = np.random.normal(0, 1, (N, T))
    rnd_corr = np.corrcoef(rnd_data)
    pa_thresh = la.eigvalsh(rnd_corr).max()
    beta = min(N, T) / max(N, T)
    omega = 0.56 * beta**3 - 1.28 * beta**2 + 1.10 * beta + 1.28
    gd_thresh = omega * np.median(la.eigvalsh(cosine_similarity(emb)))
    return l_max, gd_thresh, pa_thresh

def apply_manual_filter(embeddings, n_top, n_bottom):
    corr = cosine_similarity(embeddings)
    evals, evecs = la.eigh(corr)
    idx = evals.argsort()[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    evals_filt = evals.copy()
    
    if n_top > 0: evals_filt[:n_top] = 0
    if n_bottom > 0: evals_filt[max(0, len(evals)-n_bottom):] = 0
        
    clean_corr = evecs @ np.diag(evals_filt) @ evecs.T
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(clean_corr), 1e-10))
    clean_corr = np.outer(d_inv, d_inv) * clean_corr
    return clean_corr, evals, evals_filt

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Podešavanja")
    model_provider = st.selectbox("Model", ["LaBSE", "OpenAI 3-small", "OpenAI 3-large"])
    input_raw = st.text_area("Unos (rečenice odvojene novim redom):", height=200)
    
    st.divider()
    st.header("🧪 Statistička Grupa")
    enable_stats = st.checkbox("Omogući PERMANOVA", value=False)
    group_input = st.text_input("Grupisanje (npr. 10, 15)", help="Zbir mora odgovarati ukupnom broju rečenica.")
    
    analyze_btn = st.button("ANALIZIRAJ", type="primary", use_container_width=True)

# --- GLAVNA LOGIKA ---
if analyze_btn and input_raw:
    items = [l.strip() for l in input_raw.split('\n') if l.strip()]
    with st.spinner("Generisanje embeddinga..."):
        emb, err = get_embeddings(items, model_provider)
        if err: st.error(err)
        else: st.session_state.obj = {"emb": emb, "words": items}

if "obj" in st.session_state:
    d, n = st.session_state.obj, len(st.session_state.obj["words"])
    l_max, gd_t, pa_t = compute_thresholds(d["emb"])
    
    col_f, col_s = st.columns([1, 2])
    with col_f:
        st.subheader("🛠️ Spektralni Skalpel")
        c_top = st.slider("Ukloni najveće (Top-N)", 0, n-1, 1)
        c_bottom = st.slider("Ukloni najmanje (Bottom-N)", 0, n-1, 0)
        
    clean_c, ev_raw, ev_f = apply_manual_filter(d["emb"], c_top, c_bottom)
    
    # Stabilizacija matrice distance za dendrogram i PERMANOVA
    dist_m = 1.0 - np.clip(clean_c, -1.0, 1.0)
    dist_m_stable = (dist_m + dist_m.T) / 2
    np.fill_diagonal(dist_m_stable, 0)

    with col_s:
        # Scree Plot
        fig_s = go.Figure([
            go.Bar(y=ev_raw, name="Original", marker_color="lightgray"), 
            go.Bar(y=ev_f, name="Zadržano", marker_color="royalblue")
        ])
        thresholds = [l_max, gd_t, pa_t]
        colors = ["red", "orange", "green"]
        names = ["RMT (MP) Limit", "Gavish-Donoho", "Parallel Analysis"]
        for t, c, nm in zip(thresholds, colors, names):
            fig_s.add_hline(y=t, line_dash="dash", line_color=c)
            fig_s.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=c, dash='dash'), name=nm))
            
        fig_s.update_layout(height=300, margin=dict(t=50, b=20), yaxis_type="log", 
                            legend=dict(orientation="h", y=1.2, x=0.5, xanchor="center"),
                            title="Spektar eigenvrednosti")
        st.plotly_chart(fig_s, use_container_width=True)
        st.table(pd.DataFrame({"Metod": names, "Vrednost praga": [f"{t:.4f}" for t in thresholds]}))

    st.divider()
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.plotly_chart(px.imshow(pd.DataFrame(clean_c, index=d["words"], columns=d["words"]), 
                                  text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Očišćena sličnost"), use_container_width=True)
    with res_c2:
        st.subheader("🌳 Dendrogram")
        try:
            condensed = sch.distance.squareform(dist_m_stable, checks=False)
            fig_d = ff.create_dendrogram(dist_m_stable, labels=d["words"], orientation='left', 
                                         linkagefun=lambda x: sch.linkage(x, method='ward'))
            st.plotly_chart(fig_d, use_container_width=True)
        except Exception as e: st.warning(f"Dendrogram nije moguć: {e}")

    # --- PERMANOVA SEKCIJA ---
    if enable_stats and group_input:
        st.divider()
        st.subheader("🧪 Statistička Analiza Značajnosti (PERMANOVA)")
        if not SKBIO_AVAILABLE:
            st.error("Biblioteka 'scikit-bio' nije instalirana. Dodajte je u requirements.txt.")
        else:
            try:
                group_counts = [int(x.strip()) for x in group_input.split(",")]
                if sum(group_counts) != n:
                    st.error(f"Zbir grupa ({sum(group_counts)}) se ne poklapa sa brojem stavki ({n})!")
                else:
                    labels = []
                    for i, count in enumerate(group_counts): labels.extend([f"Grupa {i+1}"] * count)
                    
                    dm = DistanceMatrix(dist_m_stable, ids=d["words"])
                    res = permanova(dm, grouping=labels, permutations=999)
                    
                    p_val = res['p-value']
                    color = "green" if p_val < 0.05 else "red"
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("p-vrednost", f"{p_val:.4f}")
                    c2.metric("Pseudo-F", f"{res['test statistic']:.2f}")
                    c3.metric("Značajnost", "DA" if p_val < 0.05 else "NE")
                    
                    if p_val < 0.05:
                        st.success("Rezultat je statistički značajan nakon spektralnog pruninga!")
                    else:
                        st.info("Nema statistički značajne razlike između definisanih grupa.")
            except Exception as e:
                st.error(f"Greška u PERMANOVA testu: {e}")
