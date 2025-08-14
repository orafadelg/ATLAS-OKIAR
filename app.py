# -*- coding: utf-8 -*-

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIG BÁSICA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Okiar – Demo de Produtos",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#5D6CFB"
ACCENT = "#00C2A8"
WARNING = "#FFB020"
DANGER = "#EB5757"
MUTED = "#9AA0A6"
LIGHT_BG = "#F7F8FA"

CUSTOM_CSS = f"""
/* Cards e métricas */
.metric-card {{
  background: white; border: 1px solid #EAECF0; border-radius: 16px;
  padding: 16px 18px; box-shadow: 0 1px 2px rgba(16,24,40,.05);
}}
.metric-label {{ color: {MUTED}; font-size: 12px; font-weight: 500; }}
.metric-value {{ font-size: 28px; font-weight: 700; }}
.metric-delta-up {{ color: {ACCENT}; font-weight: 600; }}
.metric-delta-down {{ color: {DANGER}; font-weight: 600; }}

.block-title {{ font-size: 16px; font-weight: 700; margin-bottom: 6px; }}
.subtle {{ color: {MUTED}; font-size: 12px; }}
.section {{ padding: 6px 0 2px 0; }}

.small-table table {{ font-size: 12px; }}

.badge {{ display:inline-block; padding:4px 8px; font-size:11px; border-radius:12px; background:{LIGHT_BG}; border:1px solid #E5E7EB; color:#374151 }}
.badge-green {{ background:#ECFDF5; color:#047857; border-color:#A7F3D0; }}
.badge-yellow {{ background:#FFFBEB; color:#92400E; border-color:#FDE68A; }}
.badge-red {{ background:#FEF2F2; color:#991B1B; border-color:#FCA5A5; }}

.alert {{ border-left: 4px solid {ACCENT}; background: white; padding: 10px 12px; border-radius: 8px; border:1px solid #EAECF0; }}
.alert-critico {{ border-left-color: {DANGER}; }}
.alert-alto {{ border-left-color: {WARNING}; }}

/* Expander */
details > summary {{ cursor: pointer; }}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RANDOM SEED
# -----------------------------------------------------------------------------
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
TODAY = date.today()

# -----------------------------------------------------------------------------
# HELPERS VISUAIS
# -----------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: float | None = None):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-up" if delta >= 0 else "metric-delta-down"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f"<div class='{cls}'>{arrow} {delta:.1f}%</div>"
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pct(x):
    return f"{x*100:.1f}%"


def line_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str, height: int = 320):
    fig = px.line(df, x=x, y=y, color=color, markers=True)
    fig.update_layout(
        title=title, height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(tickformat=",.0%")
    return fig


def bars_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str, height: int = 320):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", text_auto=".0%")
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def bars_num(df: pd.DataFrame, x: str, y: str, color: str, title: str, height: int = 320):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", text_auto=True)
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def scatter_perf_import(df: pd.DataFrame, x: str, y: str, color: str, text: str, title: str, height: int = 420):
    fig = px.scatter(df, x=x, y=y, color=color, text=text, size_max=18)
    fig.update_traces(textposition="top center")
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def radar_compare(categories: List[str], a_vals: List[float], b_vals: List[float], a_name: str, b_name: str, title: str):
    # plotly polar
    df = pd.DataFrame({
        'categoria': categories * 2,
        'score': a_vals + b_vals,
        'marca': [a_name]*len(categories) + [b_name]*len(categories)
    })
    fig = px.line_polar(df, r='score', theta='categoria', color='marca', line_close=True)
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def alert_box(titulo: str, texto: str, nivel: str = "normal"):
    cls = "alert"
    if nivel == "critico":
        cls += " alert-critico"
    elif nivel == "alto":
        cls += " alert-alto"
    st.markdown(f"<div class='{cls}'><b>{titulo}</b><br/>{texto}</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DADOS MOCK (BRAIN e MERIDIO)
# -----------------------------------------------------------------------------

BRANDS = ["Banco Áquila", "Banco Boreal", "Banco Cobalto", "Banco Delta", "Banco Épsilon"]
A_BRAND = BRANDS[0]
B_BRAND = BRANDS[1]

MEDIA_HABITS = ["TV Aberta", "TV Paga", "YouTube", "TikTok", "Instagram", "Google", "Portais", "Rádio", "Out of Home"]
FACTORS = {
    "Qualidade": ["Sabor", "Aroma", "Estética", "Durabilidade", "Acabamento"],
    "Performance": ["Preço justo", "Distribuição", "Disponibilidade", "Comunicação", "Promoções"],
    "Juventude": ["Moderna", "Alegre", "Ousada", "Divertida", "Inovadora"],
    "Segurança": ["Saudável", "Credibilidade", "Confiável", "Privacidade", "Sem falhas"],
    "Proximidade": ["Combina comigo", "Reflete meus valores", "Minha cara", "Faz sentido", "Tem a ver comigo"],
}

# 25 atributos (5 fatores x 5 atributos) – performance e importância simuladas
ATTR_ROWS = []
for f, attrs in FACTORS.items():
    for a in attrs:
        perf_a = float(np.clip(np.random.normal(0.62 if f in ["Qualidade", "Segurança"] else 0.55, 0.12), 0.2, 0.95))
        imp_a = float(np.clip(np.random.normal(0.60 if f in ["Segurança", "Proximidade"] else 0.50, 0.15), 0.1, 0.95))
        ATTR_ROWS.append({"fator": f, "atributo": a, "performance": perf_a, "importancia": imp_a})
DF_ATTR = pd.DataFrame(ATTR_ROWS)

# Habitos de mídia (BRAIN / MERIDIO – Comportamento)
DF_MEDIA = pd.DataFrame({
    "canal": MEDIA_HABITS,
    A_BRAND: np.clip(np.random.dirichlet(np.ones(len(MEDIA_HABITS))) * 1.7, 0.02, None),
    B_BRAND: np.clip(np.random.dirichlet(np.ones(len(MEDIA_HABITS))) * 1.7, 0.02, None),
})
DF_MEDIA["Geral"] = (DF_MEDIA[A_BRAND] + DF_MEDIA[B_BRAND]) / 2

# Funil (top-of-mind + 4 estágios – 5 marcas)
FUNIL_STAGES = ["Top of mind", "Lembrança", "Familiaridade", "Consideração", "Preferência"]
rows = []
for brand in BRANDS:
    base = np.clip(np.random.normal(0.20, 0.06), 0.05, 0.40)
    lemb = np.clip(base + np.random.normal(0.25, 0.05), 0.10, 0.85)
    fam = np.clip(lemb + np.random.normal(0.10, 0.04), 0.10, 0.95)
    cons = np.clip(fam - np.random.uniform(0.05, 0.20), 0.05, fam)
    pref = np.clip(cons - np.random.uniform(0.03, 0.12), 0.01, cons)
    vals = [base, lemb, fam, cons, pref]
    for s, v in zip(FUNIL_STAGES, vals):
        rows.append({"marca": brand, "etapa": s, "valor": float(v)})
DF_FUNIL = pd.DataFrame(rows)

# Imagem – fatores (scores 0-1) com hover listando atributos
IMG_ROWS = []
for brand in [A_BRAND, B_BRAND]:
    for f, attrs in FACTORS.items():
        score = float(np.clip(np.random.normal(0.62 if brand==A_BRAND else 0.58, 0.08), 0.2, 0.95))
        IMG_ROWS.append({"marca": brand, "fator": f, "score": score, "hover": ", ".join(attrs)})
DF_IMG = pd.DataFrame(IMG_ROWS)

# Equity – comparar A vs B
EQUITY_METRICS = [
    "Intenção abrir conta", "Intenção recomendar", "Satisfação",
    "Intenção contratar crédito", "Principalidade"
]
DF_EQ = pd.DataFrame({
    "kpi": EQUITY_METRICS,
    A_BRAND: np.clip(np.random.normal(0.62, 0.08, len(EQUITY_METRICS)), 0.2, 0.95),
    B_BRAND: np.clip(np.random.normal(0.57, 0.08, len(EQUITY_METRICS)), 0.2, 0.90),
})

# Estratégia – matriz perf x importância (25 atributos do DF_ATTR)
# e “Participação de Branding no resultado” (R² simulado por KPI)
BRANDING_IMPACT = pd.DataFrame({
    "kpi": ["Intenção abrir conta", "Satisfação", "Intenção recomendar", "Principalidade"],
    "participacao_branding": np.clip(np.random.normal(0.68, 0.08, 4), 0.35, 0.90)
})

# ------------------- MERIDIO -------------------
# Personas – 3 faixas (A/B/C), 3 personas por faixa
PERSONA_GROUPS = {
    "A": ["Os Apressados", "Os Visionários", "Os Refinados"],
    "B": ["Os Organizados", "Os Astutos", "Os Exploradores"],
    "C": ["Os Práticos", "Os Descolados", "Os Cautelosos"],
}
PERSONA_DESC = {
    "Os Apressados": "Valorizam rapidez e conveniência; baixa tolerância a fricção.",
    "Os Visionários": "Buscam inovação e status; adotam novidades cedo.",
    "Os Refinados": "Preferem qualidade premium e atendimento diferenciado.",
    "Os Organizados": "Planejam finanças; respondem bem a programas de fidelidade.",
    "Os Astutos": "Caçadores de valor; sensíveis a preço e benefícios.",
    "Os Exploradores": "Experimentam marcas; propensos a cross-sell.",
    "Os Práticos": "Objetivos e sensíveis a preço; pouca paciência para burocracia.",
    "Os Descolados": "Digitais, influenciados por social e creators.",
    "Os Cautelosos": "Aversos a risco; exigem provas sociais e garantias.",
}

# Hábitos de mídia / frequência / ticket variando por persona
BASE_MEDIA = DF_MEDIA.set_index("canal")["Geral"].to_dict()

def persona_adjustment(name: str, base: Dict[str, float]) -> Dict[str, float]:
    # Aplica pequenos ajustes por persona
    adj = {k: v for k, v in base.items()}
    if "Descolados" in name or "Exploradores" in name:
        for k in ["TikTok", "Instagram", "YouTube"]:
            adj[k] = min(adj[k]*1.35, 0.5)
    if "Cautelosos" in name:
        for k in ["TV Aberta", "Portais", "Google"]:
            adj[k] = min(adj[k]*1.25, 0.6)
    if "Refinados" in name:
        for k in ["TV Paga", "Out of Home", "Instagram"]:
            adj[k] = min(adj[k]*1.20, 0.55)
    s = sum(adj.values())
    return {k: v/s for k, v in adj.items()}

PERSONA_MEDIA = {p: persona_adjustment(p, BASE_MEDIA) for grp in PERSONA_GROUPS.values() for p in grp}
PERSONA_FREQ = {p: float(np.clip(np.random.normal(2.6, 0.6), 0.5, 5.0)) for p in PERSONA_MEDIA.keys()}  # vezes/mês
PERSONA_TICKET = {p: float(np.clip(np.random.normal(180, 60), 40, 600)) for p in PERSONA_MEDIA.keys()}  # R$

# Drivers – 16 atributos dermocosméticos (perf x importância)
DERMO_ATTRS = [
    "Embalagem", "Preço justo", "Força de marca", "Qualidade", "Buzz",
    "Variedade", "Ingredientes naturais", "Dermatologicamente testado",
    "Textura", "Aroma", "Benefício anti-idade", "Hidratação",
    "Disponibilidade", "Sustentabilidade", "Indicação de influenciadores", "Eficácia comprovada"
]
DRIVERS = pd.DataFrame({
    "atributo": DERMO_ATTRS,
    "performance": np.clip(np.random.normal(0.58, 0.12, len(DERMO_ATTRS)), 0.15, 0.95),
    "importancia": np.clip(np.random.normal(0.55, 0.15, len(DERMO_ATTRS)), 0.10, 0.95)
})

# Fatores agregados (4) para o simulador
AGG_FACTORS = {
    "Produto": ["Qualidade", "Textura", "Aroma", "Hidratação", "Eficácia comprovada"],
    "Marca": ["Força de marca", "Buzz", "Indicação de influenciadores", "Sustentabilidade", "Ingredientes naturais"],
    "Preço & Oferta": ["Preço justo", "Variedade", "Disponibilidade"],
    "Embalagem": ["Embalagem", "Dermatologicamente testado", "Benefício anti-idade"],
}

# Pesos fictícios para intenção de compra
AGG_WEIGHTS = {"Produto": 0.38, "Marca": 0.27, "Preço & Oferta": 0.22, "Embalagem": 0.13}
BASE_INTENCAO = 0.52

# Offers – Van Westendorp (simulado)
prices = np.linspace(20, 200, 60)
# curvas cumulativas simuladas
ppc = np.clip(np.linspace(0.02, 0.85, 60) + np.random.normal(0, 0.02, 60), 0, 1)  # too cheap
pei = np.clip(np.linspace(0.01, 0.75, 60)[::-1] + np.random.normal(0, 0.02, 60), 0, 1)  # too expensive
pip = np.clip(np.linspace(0.10, 0.90, 60) + np.random.normal(0, 0.02, 60), 0, 1)  # cheap
pex = np.clip(np.linspace(0.05, 0.80, 60)[::-1] + np.random.normal(0, 0.02, 60), 0, 1)  # expensive
VW = pd.DataFrame({"preco": prices, "muito_barato": ppc, "caro": pei, "barato": pip, "muito_caro": pex})

# Experimento – Embalagem A vs B (fatores de imagem)
PACK_FACTORS = ["Preço justo", "Qualidade", "Força de marca", "Premiumidade", "Intenção de compra", "Intenção pagar mais"]
PACK = pd.DataFrame({
    "fator": PACK_FACTORS,
    "A": np.clip(np.random.normal(70, 10, len(PACK_FACTORS)), 30, 95),
    "B": np.clip(np.random.normal(75, 10, len(PACK_FACTORS)), 30, 98),
})

# -----------------------------------------------------------------------------
# SIDEBAR GLOBAL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Okiar – Demo")
    st.caption("Frameworks de Consumer/Brand Insights, inspirados nos módulos BRAIN & MERIDIO.")
    st.divider()
    brand_a = st.selectbox("Marca A", options=BRANDS, index=0)
    brand_b = st.selectbox("Marca B", options=[b for b in BRANDS if b != brand_a], index=0)
    st.caption(":grey[Dados fictícios]")

# Atualiza rótulos A e B se usuário trocar
A_BRAND = brand_a
B_BRAND = brand_b

# -----------------------------------------------------------------------------
# ABAS
# -----------------------------------------------------------------------------
TAB_BRAIN, TAB_MERIDIO, TAB_MMX, TAB_UXM, TAB_DOMUS, TAB_EBRAIN = st.tabs([
    "BRAIN", "MERIDIO", "MMX (placeholder)", "UXM (placeholder)", "Domus (placeholder)", "e‑BRAIN (placeholder)"
])

# =====================================
# BRAIN – 5 módulos (capítulos)
# =====================================
with TAB_BRAIN:
    st.title("BRAIN – Brand+Insights")
    st.caption("Módulos: Comportamento • Memória • Imagem • Equity • Estratégia")
    st.divider()

    # ---------------- Comportamento ----------------
    st.subheader("Comportamento – Hábitos de mídia (exemplo)")
    st.caption("Distribuição de canais de mídia entre marcas selecionadas.")
    media = DF_MEDIA.melt(id_vars=["canal"], value_vars=["Geral"], var_name="amostra", value_name="share")
    fig_m = px.bar(media, x="canal", y="share", text_auto=".0%", title="Hábitos de mídia – Geral")
    fig_m.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    fig_m.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_m, use_container_width=True)

    st.divider()

    # ---------------- Memória ----------------
    st.subheader("Memória – Top of mind & Funil de 5 marcas")
    st.caption("Top of mind, lembrança, familiaridade, consideração e preferência.")
    funnel_cur = DF_FUNIL.copy()
    fig_f = bars_pct(funnel_cur, x="etapa", y="valor", color="marca", title="Funil de marcas – Exemplo")
    st.plotly_chart(fig_f, use_container_width=True)

    st.divider()

    # ---------------- Imagem ----------------
    st.subheader("Imagem – Fatores de percepção (A vs B)")
    st.caption("Passe o mouse para ver os atributos dentro de cada fator.")
    img = DF_IMG[DF_IMG["marca"].isin([A_BRAND, B_BRAND])].copy()
    fig_img = px.bar(img, x="fator", y="score", color="marca", barmode="group", text_auto=".0%",
                     hover_data={"hover": True, "fator": False, "score": ":.0%", "marca": False})
    fig_img.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y:.0%}<br>Atributos: %{customdata[0]}")
    fig_img.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    fig_img.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_img, use_container_width=True)

    st.divider()

    # ---------------- Equity ----------------
    st.subheader("Equity – KPIs (A vs B)")
    eq = DF_EQ.melt(id_vars=["kpi"], value_vars=[A_BRAND, B_BRAND], var_name="marca", value_name="score")
    fig_eq = bars_pct(eq, x="kpi", y="score", color="marca", title="KPIs de equity – demonstração", height=340)
    st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    # ---------------- Estratégia ----------------
    st.subheader("Estratégia – Matriz de prioridades e participação do Branding")

    # Matriz perf x importância
    df_pi = DF_ATTR.copy()
    df_pi["zona"] = np.where(
        (df_pi["performance"] < df_pi["performance"].median()) & (df_pi["importancia"] >= df_pi["importancia"].median()),
        "Urgência", np.where(
            (df_pi["performance"] >= df_pi["performance"].median()) & (df_pi["importancia"] >= df_pi["importancia"].median()),
            "Proteção", "Acompanhamento"
        )
    )
    fig_mat = px.scatter(df_pi, x="performance", y="importancia", color="zona", text="atributo",
                         hover_data=["fator"], title="Matriz de Priorização (Performance x Importância)")
    fig_mat.update_traces(textposition="top center")
    fig_mat.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    fig_mat.update_xaxes(tickformat=",.0%")
    fig_mat.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_mat, use_container_width=True)

    # Participação de Branding (R²)
    st.markdown("<div class='block-title'>Participação de Branding no resultado</div>", unsafe_allow_html=True)
    fig_r2 = px.bar(BRANDING_IMPACT, x="kpi", y="participacao_branding", text_auto=".0%", title="")
    fig_r2.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    fig_r2.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_r2, use_container_width=True)

# =====================================
# MERIDIO – Core + Personas + Drivers + Jornada
# =====================================
with TAB_MERIDIO:
    st.title("MERIDIO – Consumer Behavior")
    st.caption("Módulos: Comportamento • Personas • Drivers • Jornada")
    st.divider()

    # ---------------- Comportamento (igual BRAIN) ----------------
    st.subheader("Comportamento – Hábitos de mídia (exemplo)")
    media = DF_MEDIA.melt(id_vars=["canal"], value_vars=["Geral"], var_name="amostra", value_name="share")
    fig_m2 = px.bar(media, x="canal", y="share", text_auto=".0%", title="Hábitos de mídia – Geral")
    fig_m2.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig_m2.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_m2, use_container_width=True)

    st.divider()

    # ---------------- Personas ----------------
    st.subheader("Personas – por faixa de renda")
    colA, colB = st.columns((1,1))
    with colA:
        faixa = st.radio("Selecione uma faixa", options=["Todas", "A", "B", "C"], horizontal=True)
    with colB:
        st.caption(":grey[Clique em uma persona para atualizar os gráficos]")

    if faixa == "Todas":
        personas = sum(PERSONA_GROUPS.values(), [])
    else:
        personas = PERSONA_GROUPS[faixa]

    pcols = st.columns(3)
    clicked = st.session_state.get("persona_clicked", "")
    for i, p in enumerate(personas):
        if pcols[i % 3].button(p, use_container_width=True):
            st.session_state["persona_clicked"] = p
            clicked = p

    st.caption(PERSONA_DESC.get(clicked, "Sem persona selecionada – exibindo média geral."))

    # Gráficos dependentes da persona
    if clicked:
        med = pd.Series(PERSONA_MEDIA[clicked]).reset_index()
        med.columns = ["canal", "share"]
        freq = PERSONA_FREQ[clicked]
        ticket = PERSONA_TICKET[clicked]
    else:
        med = DF_MEDIA[["canal", "Geral"]].rename(columns={"Geral": "share"})
        freq = float(np.mean(list(PERSONA_FREQ.values())))
        ticket = float(np.mean(list(PERSONA_TICKET.values())))

    c1, c2 = st.columns((2,1))
    with c1:
        fig_p1 = px.bar(med, x="canal", y="share", text_auto=".0%", title="Hábitos de mídia")
        fig_p1.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        fig_p1.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_p1, use_container_width=True)
    with c2:
        st.markdown("<div class='block-title'>Frequência de compra (média/mês)</div>", unsafe_allow_html=True)
        metric_card("Frequência", f"{freq:.1f}x/mês")
        st.markdown("<div class='block-title'>Ticket médio</div>", unsafe_allow_html=True)
        metric_card("Ticket", f"R$ {ticket:,.0f}")

    st.divider()

    # ---------------- Drivers ----------------
    st.subheader("Drivers – Compra • Offers • Experimento")
    dtab1, dtab2, dtab3 = st.tabs(["Drivers de Compra", "Offers", "Experimento"])

    with dtab1:
        st.markdown("<div class='block-title'>Matriz de drivers (dermocosméticos)</div>", unsafe_allow_html=True)
        fig_d1 = px.scatter(DRIVERS, x="performance", y="importancia", text="atributo", color="importancia",
                            color_continuous_scale="Tealgrn")
        fig_d1.update_traces(textposition="top center")
        fig_d1.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), coloraxis_colorbar_title="Importância")
        fig_d1.update_xaxes(tickformat=",.0%")
        fig_d1.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_d1, use_container_width=True)

        st.markdown("<div class='block-title'>Simulador de intenção de compra (4 fatores agregados)</div>", unsafe_allow_html=True)
        sc1, sc2 = st.columns((2,1))
        with sc1:
            sliders = {}
            hover_text = {
                k: ", ".join(v) for k, v in AGG_FACTORS.items()
            }
            for k in AGG_FACTORS.keys():
                sliders[k] = st.slider(f"{k}", min_value=-20, max_value=20, value=0, step=1, help=f"Inclui: {hover_text[k]}")
        with sc2:
            # Cálculo simples: intenção = base + soma(delta% * peso)
            delta_perc = {k: sliders[k]/100 for k in sliders}
            inten = BASE_INTENCAO * (1 + sum(delta_perc[k]*AGG_WEIGHTS[k] for k in AGG_WEIGHTS))
            inten = float(np.clip(inten, 0.01, 0.99))
            # Tabela de impactos
            imp_rows = [{"Fator": k, "% Δ atributo": f"{delta_perc[k]*100:+.0f}%", "% Δ intenção": f"{delta_perc[k]*AGG_WEIGHTS[k]*100:+.1f}%"} for k in sliders]
            df_imp = pd.DataFrame(imp_rows)
            st.dataframe(df_imp, use_container_width=True, height=180)
            st.markdown("<div class='block-title'>Intenção de compra estimada</div>", unsafe_allow_html=True)
            metric_card("Intenção", pct(inten))

    with dtab2:
        st.markdown("<div class='block-title'>Escolha uma opção</div>", unsafe_allow_html=True)
        cco, cpr = st.columns((1,1))
        with cco:
            st.markdown("**Conjoint**")
            st.link_button("Abrir simulador de Conjoint", "https://simuladorconjointtdah-g3tyutubdwrqowizlogkuc.streamlit.app")
        with cpr:
            st.markdown("**Price – Van Westendorp**")
            fig_vw = go.Figure()
            fig_vw.add_trace(go.Scatter(x=VW["preco"], y=VW["muito_barato"], name="Muito barato"))
            fig_vw.add_trace(go.Scatter(x=VW["preco"], y=VW["barato"], name="Barato"))
            fig_vw.add_trace(go.Scatter(x=VW["preco"], y=VW["caro"], name="Caro"))
            fig_vw.add_trace(go.Scatter(x=VW["preco"], y=VW["muito_caro"], name="Muito caro"))
            fig_vw.update_layout(title="Curvas de sensibilidade de preço (VW)", xaxis_title="Preço", yaxis_title="% acumulado",
                                 height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_vw, use_container_width=True)
            st.caption("Faixa ideal de preço: região entre as interseções Barato×Caro e Muito barato×Muito caro (estimada no gráfico).")

    with dtab3:
        st.markdown("<div class='block-title'>Teste A/B – Embalagens</div>", unsafe_allow_html=True)
        pack_m = PACK.melt(id_vars=["fator"], value_vars=["A", "B"], var_name="embalagem", value_name="score")
        fig_pack = px.bar(pack_m, x="fator", y="score", color="embalagem", barmode="group", text_auto=True,
                          title="Desempenho por fator (A vs B)")
        fig_pack.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pack, use_container_width=True)

    st.divider()

    # ---------------- Jornada ----------------
    st.subheader("Jornada – Exemplo (macro‑etapas)")
    steps = ["Descoberta", "Pesquisa", "Avaliação", "Compra", "Pós‑compra"]
    sat = np.clip(np.random.normal(7, 1.2, len(steps)), 4, 10)  # CSAT 0–10
    drop = np.clip(np.random.normal(0.12, 0.05, len(steps)), 0.02, 0.35)
    df_j = pd.DataFrame({"etapa": steps, "CSAT": sat, "Drop-off": drop})
    jj1, jj2 = st.columns((1,1))
    with jj1:
        fig_j1 = px.bar(df_j, x="etapa", y="CSAT", text_auto=True, title="Satisfação por etapa")
        fig_j1.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_j1, use_container_width=True)
    with jj2:
        fig_j2 = px.line(df_j, x="etapa", y="Drop-off", markers=True, title="Queda por etapa")
        fig_j2.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        fig_j2.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_j2, use_container_width=True)

# =====================================
# MMX – Customer Experience (Seguros)
# =====================================
with TAB_MMX:
    st.title("MMX – Customer Experience (Seguros)")
    st.caption("Blocos: Overview • Evolução • Priorização • Mini simulador")
    st.divider()

    # ---------- Dados fictícios ----------
    np.random.seed(17)
    clientes = ["Cliente A", "Cliente B"]
    fatores_mmx = {
        "Atendimento": ["Cordialidade", "Agilidade", "Resolução no 1º contato"],
        "Canais Digitais": ["Facilidade no app", "Estabilidade", "Autoatendimento"],
        "Produto/Serviços": ["Coberturas", "Clareza de contrato", "Adequação ao perfil"],
        "Preço/Valor": ["Preço percebido", "Custo-benefício", "Transparência de reajuste"],
        "Suporte/Resolução": ["Pós-sinistro", "Prazo de retorno", "Acompanhamento do caso"],
    }

    # Score 0-1 por atributo (média por fator depois)
    def mock_attr_scores(base_shift=0.0):
        rows = []
        for fator, attrs in fatores_mmx.items():
            for a in attrs:
                score = float(np.clip(np.random.normal(0.62 + base_shift, 0.10), 0.25, 0.95))
                importancia = float(np.clip(np.random.normal(0.58, 0.12), 0.10, 0.95))
                rows.append({"fator": fator, "atributo": a, "score": score, "importancia": importancia})
        return pd.DataFrame(rows)

    mmx_A = mock_attr_scores(base_shift=0.03)   # A levemente melhor
    mmx_B = mock_attr_scores(base_shift=-0.02)  # B levemente pior

    # ---------- 1) Overview ----------
    st.subheader("Overview – Experiência por fator (Cliente A vs Cliente B)")
    fa = mmx_A.groupby("fator", as_index=False)["score"].mean().rename(columns={"score": "A"})
    fb = mmx_B.groupby("fator", as_index=False)["score"].mean().rename(columns={"score": "B"})
    fcmp = fa.merge(fb, on="fator")
    fcmp_m = fcmp.melt(id_vars=["fator"], value_vars=["A", "B"], var_name="cliente", value_name="score")

    fig_ov = px.bar(fcmp_m, x="fator", y="score", color="cliente", barmode="group",
                    text_auto=".0%", title="Média de experiência por fator")
    fig_ov.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    fig_ov.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_ov, use_container_width=True)

    # ---------- 2) Evolução ----------
    st.subheader("Evolução – Satisfação e NPS (últimos 12 meses)")
    meses = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="MS")
    evo = pd.DataFrame({
        "mes": list(meses)*2,
        "cliente": np.repeat(clientes, len(meses)),
        "satisfacao": np.clip(np.linspace(0.62, 0.70, 12) + np.random.normal(0, 0.01, 24) + np.where(np.repeat(clientes, 12)=="Cliente A", 0.02, -0.01), 0.30, 0.95),
        "nps": np.clip(np.linspace(0.28, 0.40, 12) + np.random.normal(0, 0.02, 24) + np.where(np.repeat(clientes, 12)=="Cliente A", 0.05, -0.02), -0.2, 0.9),
    })
    c1, c2 = st.columns(2)
    with c1:
        fig_s = px.line(evo, x="mes", y="satisfacao", color="cliente", markers=True, title="Satisfação (CSAT normalizado)")
        fig_s.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        fig_s.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_s, use_container_width=True)
    with c2:
        fig_n = px.line(evo, x="mes", y="nps", color="cliente", markers=True, title="NPS (escala normalizada)")
        fig_n.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        fig_n.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_n, use_container_width=True)

    # ---------- 3) Priorização ----------
    st.subheader("Priorização – Matriz Performance × Importância (15 atributos)")
    pri = pd.concat([mmx_A.assign(cliente="A"), mmx_B.assign(cliente="B")], ignore_index=True)
    # Usaremos os 15 atributos (5 fatores x 3 atributos)
    pri["zona"] = np.where(
        (pri["score"] < pri["score"].median()) & (pri["importancia"] >= pri["importancia"].median()), "Urgência",
        np.where((pri["score"] >= pri["score"].median()) & (pri["importancia"] >= pri["importancia"].median()), "Proteger", "Acompanhar")
    )
    fig_pri = px.scatter(pri, x="score", y="importancia", color="zona", symbol="cliente", text="atributo",
                         hover_data=["fator"], title="Atributos prioritários por cliente")
    fig_pri.update_traces(textposition="top center")
    fig_pri.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    fig_pri.update_xaxes(title="Performance", tickformat=",.0%")
    fig_pri.update_yaxes(title="Importância", tickformat=",.0%")
    st.plotly_chart(fig_pri, use_container_width=True)

    # ---------- 4) Mini simulador ----------
    st.subheader("Mini simulador – Impacto dos atributos em Satisfação → KPIs")
    st.caption("Ajuste os fatores (±20%). Satisfação afeta: NPS (↑), Churn (↓), Reclamações (↓), Cross-sell (↑).")

    # simularemos em nível de fator (5 sliders)
    pesos_fator = {
        "Atendimento": 0.28,
        "Canais Digitais": 0.20,
        "Produto/Serviços": 0.24,
        "Preço/Valor": 0.16,
        "Suporte/Resolução": 0.12,
    }
    colL, colR = st.columns((2,1))
    with colL:
        deltas = {}
        for f in fatores_mmx.keys():
            deltas[f] = st.slider(f"{f}", min_value=-20, max_value=20, value=0, step=1, help="Variação percentual do fator")

        base_sat = 0.66
        sat = base_sat * (1 + sum((deltas[f]/100.0)*pesos_fator[f] for f in pesos_fator))
        sat = float(np.clip(sat, 0.01, 0.99))

        # Mapeamentos simples (exemplo)
        nps = np.clip(0.2 + 0.9*sat, 0.0, 0.99)            # ↑
        churn = np.clip(0.25 - 0.25*sat, 0.01, 0.40)       # ↓
        recl = np.clip(0.30 - 0.35*sat, 0.01, 0.35)        # ↓
        cross = np.clip(0.10 + 0.8*sat, 0.02, 0.95)        # ↑

    with colR:
        metric_card("Satisfação (estimada)", f"{sat*100:.1f}%")
        metric_card("NPS (estimado)", f"{nps*100:.1f}%")
        metric_card("Churn (estimado)", f"{churn*100:.1f}%")
        metric_card("Reclamações (estimado)", f"{recl*100:.1f}%")
        metric_card("Cross-sell (estimado)", f"{cross*100:.1f}%")

    st.markdown("**Impactos por fator**")
    imp_rows = [{"Fator": f, "% Δ fator": f"{deltas[f]:+.0f}%", "Peso": f"{pesos_fator[f]*100:.0f}%"} for f in pesos_fator]
    st.dataframe(pd.DataFrame(imp_rows), use_container_width=True, height=180)


# =====================================
# UXM – Digital Experience
# =====================================
with TAB_UXM:
    st.title("UXM – Digital Experience")
    st.caption("Blocos: Overview • Evolução • Priorização • Simulador (3 colunas)")
    st.divider()

    # ---------- Dados fictícios ----------
    np.random.seed(23)
    big_five = {
        "Findability": ["Busca interna", "Arquitetura de informação", "Navegação"],
        "Usability": ["Fluxos claros", "Aprendizado rápido", "Erros recuperáveis"],
        "Performance": ["Velocidade", "Estabilidade", "Peso das páginas"],
        "Trust & Security": ["Privacidade", "Transparência", "Confiabilidade"],
        "Accessibility": ["Leitura", "Contraste", "Teclado/Screen reader"],
    }

    def mock_ux_scores(shift=0.0):
        rows = []
        for f, attrs in big_five.items():
            for a in attrs:
                s = float(np.clip(np.random.normal(0.60+shift, 0.10), 0.20, 0.95))
                w = float(np.clip(np.random.normal(0.56, 0.12), 0.10, 0.95))
                rows.append({"fator": f, "atributo": a, "score": s, "importancia": w})
        return pd.DataFrame(rows)

    uxm_A = mock_ux_scores(shift=0.03)
    uxm_B = mock_ux_scores(shift=-0.02)

    # ---------- 1) Overview ----------
    st.subheader("Overview – Big Five de UX (A vs B)")
    fa = uxm_A.groupby("fator", as_index=False)["score"].mean().rename(columns={"score": "A"})
    fb = uxm_B.groupby("fator", as_index=False)["score"].mean().rename(columns={"score": "B"})
    fcmp = fa.merge(fb, on="fator").melt(id_vars=["fator"], value_vars=["A","B"], var_name="marca", value_name="score")
    fig_ux = px.bar(fcmp, x="fator", y="score", color="marca", barmode="group", text_auto=".0%", title="Scores por fator")
    fig_ux.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    fig_ux.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_ux, use_container_width=True)

    # ---------- 2) Evolução ----------
    st.subheader("Evolução – UX Equity (últimos 12 meses)")
    meses = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="MS")
    evo = pd.DataFrame({
        "mes": list(meses)*2,
        "marca": np.repeat(["Marca A","Marca B"], len(meses)),
        "ux_equity": np.clip(np.linspace(0.55, 0.68, 12) + np.random.normal(0, 0.01, 24) + np.where(np.repeat(["Marca A","Marca B"], 12)=="Marca A", 0.02, -0.01), 0.25, 0.95)
    })
    fig_uxe = px.line(evo, x="mes", y="ux_equity", color="marca", markers=True, title="UX Equity (normalizado)")
    fig_uxe.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig_uxe.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_uxe, use_container_width=True)

    # ---------- 3) Priorização ----------
    st.subheader("Priorização – Performance × Importância (Big Five detalhado)")
    pri = pd.concat([uxm_A.assign(marca="A"), uxm_B.assign(marca="B")], ignore_index=True)
    pri["zona"] = np.where(
        (pri["score"] < pri["score"].median()) & (pri["importancia"] >= pri["importancia"].median()), "Urgência",
        np.where((pri["score"] >= pri["score"].median()) & (pri["importancia"] >= pri["importancia"].median()), "Proteger", "Acompanhar")
    )
    fig_pu = px.scatter(pri, x="score", y="importancia", color="zona", symbol="marca", text="atributo",
                        hover_data=["fator"], title="Detalhamento de atributos (Big Five)")
    fig_pu.update_traces(textposition="top center")
    fig_pu.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    fig_pu.update_xaxes(title="Performance", tickformat=",.0%")
    fig_pu.update_yaxes(title="Importância", tickformat=",.0%")
    st.plotly_chart(fig_pu, use_container_width=True)

    # ---------- 4) Simulador (3 colunas) ----------
    st.subheader("Simulador – Big Five → UX Equity → Métricas de negócio")
    st.caption("Ajuste os 5 fatores (±20%). UX Equity impacta: Conversão (↑), Retenção (↑), Suporte (↓).")

    pesos = {
        "Findability": 0.22,
        "Usability": 0.26,
        "Performance": 0.18,
        "Trust & Security": 0.20,
        "Accessibility": 0.14,
    }
    base_ux = 0.60

    col1, col2, col3 = st.columns(3)
    with col1:
        d = {}
        for f in pesos.keys():
            d[f] = st.slider(f, min_value=-20, max_value=20, value=0, step=1)
        ux_equity = base_ux * (1 + sum((d[f]/100.0)*pesos[f] for f in pesos))
        ux_equity = float(np.clip(ux_equity, 0.01, 0.99))
        metric_card("UX Equity (estimado)", f"{ux_equity*100:.1f}%")

    with col2:
        # Regras simples de negócio
        conversao = np.clip(0.08 + 0.8*ux_equity, 0.01, 0.95)   # ↑
        retencao = np.clip(0.70 + 0.3*ux_equity, 0.30, 0.99)    # ↑
        suporte = np.clip(0.35 - 0.4*ux_equity, 0.01, 0.40)     # ↓ (chamados/contatos)
        metric_card("Conversão (estimada)", f"{conversao*100:.1f}%")
        metric_card("Retenção (estimada)", f"{retencao*100:.1f}%")
        metric_card("Chamados de suporte (↓)", f"{suporte*100:.1f}%")

    with col3:
        # Tabela de impactos por fator
        imp_rows = [{"Fator": f, "% Δ fator": f"{d[f]:+.0f}%", "Peso": f"{pesos[f]*100:.0f}%"} for f in pesos]
        st.dataframe(pd.DataFrame(imp_rows), use_container_width=True, height=220)
        # Mini gráfico
        fig_k = px.bar(pd.DataFrame({
            "kpi": ["Conversão","Retenção","Suporte (↓)"],
            "valor": [conversao, retencao, suporte]
        }), x="kpi", y="valor", text_auto=".0%")
        fig_k.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
        fig_k.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_k, use_container_width=True)

with TAB_DOMUS:
    st.info("Domus será implementado no terceiro código (pilares de imagem + simulador voltado a colaboradores).")
with TAB_EBRAIN:
    st.info("e‑BRAIN será implementado no terceiro código (estrutura similar ao BRAIN para experiência dos colaboradores).")

st.divider()
st.write(":grey[Demo Okiar • Dados 100% fictícios • v0.1 – Estrutura + BRAIN + MERIDIO]")

