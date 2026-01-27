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

# --- INICIJALIZACIJA ---
client = OpenAI(api_key=st.secrets.get("openai", {}).get("api_key", "YOUR_OPENAI_API_KEY"))

st.set_page_config(page_title="Semantički Objektiv", layout="wide")

@st.cache_resource
def load_labse(): return SentenceTransformer("sentence-transformers/LaBSE")
labse_model = load_labse()

def get_embeddings(items, provider):
    if provider == "LaBSE":
        return labse_model.encode(items, convert_to_numpy=True, normalize_embeddings=True), None
    try:
        model_nm = "text-embedding-3-small" if "small" in provider else "text-embedding-3-large"
        res = client.embeddings.create(model=model_nm, input=items)
        return np.array([r.embedding for r in res.data]), None
    except Exception as e: return None, str(e)

def compute_thresholds(emb):
    N, T = emb.shape
    q = T / N
    # 1. RMT (Marchenko-Pastur)
    l_max = (1 + np.sqrt(1/q))**2
    # 2. Parallel Analysis (Monte Carlo)
    rnd_data = np.random.normal(0, 1, (N, T))
    rnd_corr = np.corrcoef(rnd_data)
    pa_thresh = la.eigvalsh(rnd_corr).max()
    # 3. Gavish-Donoho
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
    
    # Samo ručno nuliranje (Skalpel)
    if n_top > 0: evals_filt[:n_top] = 0
    if n_bottom > 0: evals_filt[max(0, len(evals)-n_bottom):] = 0
        
    # Rekonstrukcija
    clean_corr = evecs @ np.diag(evals_filt) @ evecs.T
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(clean_corr), 1e-10))
    clean_corr = np.outer(d_inv, d_inv) * clean_corr
    
    return clean_corr, evals, evals_filt

# --- UI ---
st.title(":rainbow[Semantički Objektiv]: Manuelna Spektroskopija")

with st.sidebar:
    st.header("Konfiguracija")
    model_provider = st.selectbox("Model", ["LaBSE", "OpenAI 3-small", "OpenAI 3-large"])
    input_raw = st.text_area("Unos podataka:", "Kralj\nKraljica\nMuškarac\nŽena\nJabuka\nKruška\nAvion\nAuto", height=250)
    analyze_btn = st.button("Analiziraj", type="primary", use_container_width=True)

if analyze_btn and input_raw:
    items = [l.strip() for l in input_raw.split('\n') if l.strip()]
    with st.spinner("Embedding..."):
        emb, err = get_embeddings(items, model_provider)
        if err: st.error(err)
        else: st.session_state.obj = {"emb": emb, "words": items}

if "obj" in st.session_state:
    d, n = st.session_state.obj, len(st.session_state.obj["words"])
    l_max, gd_t, pa_t = compute_thresholds(d["emb"])
    
    col_f, col_s = st.columns([1, 2])
    with col_f:
        st.subheader("🛠️ Ručni Skalpel")
        c_top = st.slider("Ukloni najveće (Top-N)", 0, n-1, 1, help="Ukloni PC1 da pojačaš kontrast.")
        c_bottom = st.slider("Ukloni najmanje (Bottom-N)", 0, n-1, 0, help="Gledaj zelenu liniju na grafiku za orijentaciju.")
        
    clean_c, ev_raw, ev_f = apply_manual_filter(d["emb"], c_top, c_bottom)
    
    # Matrica distance sa zaštitom od beskonačnosti
    dist_m = 1.0 - np.clip(clean_c, -1.0, 1.0)
    dist_m_s = (dist_m + dist_m.T) / 2
    np.fill_diagonal(dist_m_s, 0)
    dist_m_s = np.nan_to_num(dist_m_s, nan=1.0, posinf=2.0, neginf=0.0)

    with col_s:
        # Scree Plot sa vizuelnim granicama
        fig_s = go.Figure([
            go.Bar(y=ev_raw, name="Original", marker_color="lightgray"), 
            go.Bar(y=ev_f, name="Zadržano", marker_color="royalblue")
        ])
        
        # Pragovi kao vizuelne linije
        thresholds = [l_max, gd_t, pa_t]
        colors = ["red", "orange", "green"]
        names = ["RMT (MP) Limit", "Gavish-Donoho", "Parallel Analysis"]
        
        for t, c, nm in zip(thresholds, colors, names):
            fig_s.add_hline(y=t, line_dash="dash", line_color=c)
            fig_s.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=c, dash='dash'), name=nm))
            
        fig_s.update_layout(height=300, margin=dict(t=50, b=20), yaxis_type="log", 
                            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                            title="Spektar sa teorijskim granicama šuma")
        st.plotly_chart(fig_s, use_container_width=True)

        st.table(pd.DataFrame({"Metod": names, "Vrednost praga": [f"{t:.4f}" for t in thresholds]}))

    st.divider()
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.plotly_chart(px.imshow(pd.DataFrame(clean_c, index=d["words"], columns=d["words"]), 
                                  text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Fokusirana matrica sličnosti"), use_container_width=True)
    with res_c2:
        try:
            fig_d = ff.create_dendrogram(dist_m_s, labels=d["words"], orientation='left', linkagefun=lambda x: sch.linkage(x, method='ward'))
            fig_d.update_layout(title="Dendrogram (Ward linkage)")
            st.plotly_chart(fig_d, use_container_width=True)
        except Exception as e: st.warning(f"Dendrogram nije moguć sa trenutnim rezom: {e}")
