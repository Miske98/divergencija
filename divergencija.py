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

def apply_advanced_filter(embeddings, n_top, n_bottom, use_rmt):
    corr = cosine_similarity(embeddings)
    evals, evecs = la.eigh(corr)
    idx = evals.argsort()[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    evals_filt = evals.copy()
    
    # 1. RMT Marchenko-Pastur (Peglanje šuma)
    l_max = 0
    if use_rmt:
        T, N = embeddings.shape[1], embeddings.shape[0]
        q = T / N
        l_max = (1 + (1/q)**0.5)**2
        noise_idx = (evals < l_max) & (evals > 0)
        if np.any(noise_idx):
            evals_filt[noise_idx] = np.mean(evals[noise_idx])

    # 2. Manuelni Skalpel (Nuliranje)
    if n_top > 0: evals_filt[:n_top] = 0
    if n_bottom > 0: evals_filt[max(0, len(evals)-n_bottom):] = 0
        
    # Rekonstrukcija i normalizacija
    clean_corr = evecs @ np.diag(evals_filt) @ evecs.T
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(clean_corr), 1e-10))
    clean_corr = np.outer(d_inv, d_inv) * clean_corr
    return clean_corr, evals, evals_filt, l_max

# --- UI ---
st.title(":rainbow[Semantički Objektiv]: RMT & Skalpel")

with st.sidebar:
    st.header("Ulazni podaci")
    model_provider = st.selectbox("Model", ["LaBSE", "OpenAI 3-small", "OpenAI 3-large"])
    input_raw = st.text_area("Unesite tekst (svaki red je jedan objekat):", "Kralj\nKraljica\nMuškarac\nŽena\nJabuka\nKruška", height=300)
    analyze_btn = st.button("Analiziraj", type="primary", use_container_width=True)

items = [l.strip() for l in input_raw.split('\n') if l.strip()]

if analyze_btn and items:
    with st.spinner("Računanje..."):
        emb, err = get_embeddings(items, model_provider)
        if err: st.error(err)
        else: st.session_state.obj = {"emb": emb, "words": items}

if "obj" in st.session_state:
    d = st.session_state.obj
    n = len(d["words"])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Filteri")
        use_rmt = st.checkbox("Aktiviraj RMT čišćenje", value=True)
        c_top = st.slider("Ukloni najveće (Top-N)", 0, n-1, 0)
        c_bottom = st.slider("Ukloni najmanje (Bottom-N)", 0, n-1, 0)
    
    clean_c, ev_raw, ev_f, l_max = apply_advanced_filter(d["emb"], c_top, c_bottom, use_rmt)
    dist_m = 1.0 - np.clip(clean_c, -1.0, 1.0)
    np.fill_diagonal(dist_m, 0)

    with col2:
        fig_s = go.Figure([go.Bar(y=ev_raw, name="Original", marker_color="lightgray"), go.Bar(y=ev_f, name="Filtrirano", marker_color="royalblue")])
        if use_rmt: fig_s.add_hline(y=l_max, line_dash="dash", line_color="red", annotation_text="MP Prag")
        fig_s.update_layout(height=250, margin=dict(t=20, b=20), yaxis_type="log", title="Scree Plot (Log skala)")
        st.plotly_chart(fig_s, use_container_width=True)

    

    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.plotly_chart(px.imshow(pd.DataFrame(clean_c, index=d["words"], columns=d["words"]), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Matrica sličnosti"), use_container_width=True)
    
    with res_col2:
        st.subheader("Klasterizacija")
        try:
            # 1. Numeričko peglanje za stabilnost na Cloudu
            dist_m_stable = (dist_m + dist_m.T) / 2
            np.fill_diagonal(dist_m_stable, 0)
        
            # 2. Provera da li ima NaN vrednosti (zna da se desi kod agresivnog RMT-a)
            if np.any(np.isnan(dist_m_stable)):
                st.warning("Matrica sadrži nevalidne podatke. Smanjite intenzitet filtera.")
            else:
                # 3. Eksplicitno računanje linkage-a pre Plotly-ja
                # Ovo pomaže da vidimo gde tačno greška nastaje
                Z = sch.linkage(sch.distance.squareform(dist_m_stable), method='ward')
            
                fig_d = ff.create_dendrogram(
                    dist_m_stable, 
                    labels=d["words"], 
                    orientation='left',
                    linkagefun=lambda x: sch.linkage(x, method='ward')
                )
                fig_d.update_layout(title="Hijerarhijski klasteri", height=500)
                st.plotly_chart(fig_d, use_container_width=True)
        except Exception as e:
            st.error(f"Greška u dendrogramu: {e}")

    

    avg_d = np.mean(dist_m[np.triu_indices(n, k=1)])
    st.metric("Srednja distanca", f"{avg_d:.3f}")
