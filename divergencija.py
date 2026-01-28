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

# Pokušaj uvoza skbio za PERMANOVA test
try:
    from skbio.stats.distance import permanova, DistanceMatrix
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False

# --- CONFIG ---
st.set_page_config(page_title="Semantička analiza", layout="wide")
client = OpenAI(api_key=st.secrets.get("openai", {}).get("api_key", "YOUR_OPENAI_API_KEY"))

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

def compute_gavish_donoho(emb):
    N, T = emb.shape
    beta = min(N, T) / max(N, T)
    omega = 0.56 * beta**3 - 1.28 * beta**2 + 1.10 * beta + 1.28
    # Referenca: Medijana spektra korelacije
    return omega * np.median(la.eigvalsh(cosine_similarity(emb)))

def apply_range_filter(embeddings, range_start, range_end):
    # 1. Dekompozicija
    corr = cosine_similarity(embeddings)
    evals, evecs = la.eigh(corr)
    
    # Sortiranje (od najvećeg ka najmanjem)
    idx = evals.argsort()[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    
    # 2. Filtriranje opsega (Band-pass)
    evals_filt = np.zeros_like(evals)
    # Python slicing: range_end + 1 da bi uključio i poslednji indeks
    evals_filt[range_start : range_end + 1] = evals[range_start : range_end + 1]
    
    # 3. Rekonstrukcija
    clean_corr = evecs @ np.diag(evals_filt) @ evecs.T
    
    # Normalizacija nazad u korelaciju (zaštita od deljenja nulom)
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(clean_corr), 1e-10))
    clean_corr = np.outer(d_inv, d_inv) * clean_corr
    
    return clean_corr, evals, evals_filt

# --- UI HEADER & INPUTS ---
st.title("Spektralna analiza")

with st.sidebar:
    st.header("Podešavanja modela")
    model_provider = st.selectbox("Model embeddinga", ["LaBSE", "OpenAI 3-small", "OpenAI 3-large"])

    
col_input, col_groups = st.columns([2, 1])

with col_input:
    st.subheader("1. Unos teksta")
    input_raw = st.text_area(
        "Zalepi rečenice ovde (svaka u novom redu):", 
        height=300, # Veći textbox
        placeholder="Rečenica 1\nRečenica 2..."
    )

with col_groups:
    st.subheader("2. Definicija grupa")
    st.info("Opciono: Za statističke testove")
    group_input = st.text_area(
        "Broj rečenica po grupama:",
        height=100,
        placeholder="Primer: 10, 15\n(Znači: prvih 10 je Grupa A, sledećih 15 je Grupa B)"
    )
    analyze_btn = st.button("Analiza", type="primary", use_container_width=True)

# --- ANALIZA & RESULTS ---
if analyze_btn and input_raw:
    items = [l.strip() for l in input_raw.split('\n') if l.strip()]
    
    if len(items) < 3:
        st.error("Unesite bar 3 rečenice za analizu.")
    else:
        with st.spinner("Računam embeddinge i spektar..."):
            emb, err = get_embeddings(items, model_provider)
            
        if err: 
            st.error(err)
        else:
            st.session_state.data = {"emb": emb, "words": items, "groups": group_input}

# --- PRIKAZ REZULTATA (Samo ako postoje podaci) ---
if "data" in st.session_state:
    d = st.session_state.data
    n_samples = len(d["words"])
    
    # Računanje praga (samo jednom)
    gd_thresh = compute_gavish_donoho(d["emb"])
    
    st.divider()
    
    # --- GLAVNI INTERFEJS ZA ISTRAŽIVANJE ---
    # Levo: Vizuelizacija spektra i Slajder
    # Desno: Dendrogram i Matrica
    
    col_vis_control, col_dendo = st.columns([4, 5])
    
    with col_vis_control:
        st.subheader("Spektralni filter")
        
        # 1. RANGE SLIDER (Direktno kontroliše šta se vidi)
        # Default: Zadrži sve osim prve (ako je bias) i zadnje trećine
        sel_range = st.slider(
            "Izaberite opseg karakterističnih vrednosti za zadržavanje:",
            min_value=0,
            max_value=n_samples - 1,
            value=(0, n_samples - 1),
            step=1,
            help="Vrednosti unutar opsega se koriste za rekonstrukciju. Ostale se brišu."
        )
        
        # Primena filtera
        clean_c, ev_raw, ev_filt = apply_range_filter(d["emb"], sel_range[0], sel_range[1])
        
        # 2. SCREE PLOT (Sinhronizovan sa slajderom)
        # Bojimo stubiće: Siva = Obrisano, Plava = Zadržano
        colors = ['lightgray'] * n_samples
        for i in range(sel_range[0], sel_range[1] + 1):
            colors[i] = 'royalblue'
            
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(
            y=ev_raw,
            marker_color=colors,
            name="Eigenvrednosti",
            text=[f"EV{i}" for i in range(n_samples)],
            hovertemplate="<b>%{text}</b><br>Vrednost: %{y:.2f}<extra></extra>"
        ))
        
        # Gavish-Donoho linija (Samo ona)
        fig_s.add_hline(y=gd_thresh, line_dash="longdashdot", line_color="red", annotation_text="Gavish-Donoho Prag", annotation_position="top right")
        
        fig_s.update_layout(
            height=350,
            margin=dict(t=30, b=0, l=0, r=0),
            yaxis_type="log",
            yaxis_title="Karakteristična vrednost (Log)",
            xaxis_title="Indeks",
            showlegend=False,
            title="Scree plot (Plavo = Aktivno)"
        )
        st.plotly_chart(fig_s, use_container_width=True)
        
        # Info o zadržanoj energiji (opciono, ali korisno)
        kept_var = np.sum(ev_filt) / np.sum(ev_raw) * 100
        st.caption(f"Izolovano {kept_var:.1f}% varijabiliteta.")

        # Matrica sličnosti (manja, ispod kontrola)
        st.subheader("Matrica sličnosti")
        st.plotly_chart(px.imshow(
            pd.DataFrame(clean_c, index=d["words"], columns=d["words"]), 
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1,
            labels=dict(x="", y="")
        ), use_container_width=True)

    with col_dendo:
        st.subheader("Dendrogram")
        
        # --- STABILNA MATEMATIKA ZA DENDROGRAM ---
        # 1. Distanca
        dist_m = 1.0 - np.clip(clean_c, -1.0, 1.0)
        
        # 2. Prisilna simetrija (ključno za flow bez grešaka)
        dist_m_stable = (dist_m + dist_m.T) / 2
        np.fill_diagonal(dist_m_stable, 0)
        
        # 3. Iscrtavanje
        try:
            # Eksplicitni linkage calculation
            condensed = sch.distance.squareform(dist_m_stable, checks=False)
            linkage_matrix = sch.linkage(condensed, method='ward')
            
            fig_d = ff.create_dendrogram(
                dist_m_stable,
                labels=d["words"],
                orientation='left',
                linkagefun=lambda x: linkage_matrix
            )
            fig_d.update_layout(height=700, margin=dict(t=0, b=0)) # Viši dendrogram
            st.plotly_chart(fig_d, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Nije moguće iscrtati klastere za ovaj opseg: {e}")

    # --- PERMANOVA TABELA (SPSS STIL) ---
    if d["groups"] and SKBIO_AVAILABLE:
        st.divider()
        st.subheader("ANOVA tabela (PERMANOVA)")
        
        try:
            # Parsiranje grupa
            g_counts = [int(x.strip()) for x in d["groups"].split(",")]
            if sum(g_counts) == n_samples:
                labels = []
                for i, c in enumerate(g_counts): labels.extend([f"Grupa {i+1}"] * c)
                
                # Proračun
                dm = DistanceMatrix(dist_m_stable, ids=d["words"])
                res = permanova(dm, grouping=labels, permutations=999)
                
                # Izrada Tabele
                df_stats = pd.DataFrame({
                    "Izvor varijanse": ["Between (Model)", "Within (Rezidual)", "Total"],
                    "df": [len(g_counts)-1, n_samples-len(g_counts), n_samples-1],
                    "F": [f"{res['test statistic']:.3f}", "", ""],
                    "p": [f"{res['p-value']:.4f}", "", ""],
                    "N Permutation": [res['number of permutations'], "", ""]
                })
                
                # Checkboxovi za custom prikaz
                st.write("Prilagodi prikaz:")
                cols = st.columns(4)
                show_f = cols[0].checkbox("Prikaži F", True)
                show_p = cols[1].checkbox("Prikaži p", True)
                
                final_cols = ["Izvor varijanse", "df"]
                if show_f: final_cols.append("F")
                if show_p: final_cols.append("p")
                final_cols.append("N Permutation")
                
                st.table(df_stats[final_cols])
                
                # Export
                st.download_button("📥 Preuzmi CSV", df_stats.to_csv(index=False), "permanova.csv")
                
            else:
                st.error(f"Zbir grupa ({sum(g_counts)}) se ne slaže sa brojem rečenica ({n_samples}).")
        except Exception as e:
            st.error(f"Greška u statistici: {e}")
