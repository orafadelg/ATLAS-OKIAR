# -*- coding: utf-8 -*-
"""
Okiar – Demo de Produtos (Streamlit)
------------------------------------
Abas: BRAIN • MERIDIO • MMX • UXM • Domus • e-BRAIN

Notas de implementação:
- Filtros de marca ficam **apenas na aba BRAIN** (Sidebar não tem filtros).
- Fatores/atributos adequados a bancos (sem atributos fora de contexto).
- Simuladores com efeitos mais visíveis nos sliders (pesos/curvas ajustados).
- Van Westendorp: plot maior + cálculo das interseções e destaque da faixa ideal.
- Correções de shape/broadcasting em séries temporais (ex.: uso de np.tile).
- Hovers com custom_data para textos de atributos por fator/pilar.

Como rodar:
    streamlit run app.py
"""

# =============================================================================
# Imports
# =============================================================================
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

from datetime import date, datetime
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

# =============================================================================
# Config / Tema
# =============================================================================
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

# =============================================================================
# Seeds
# =============================================================================
SEED = 11
random.seed(SEED)
np.random.seed(SEED)
TODAY = date.today()

# =============================================================================
# Helpers – UI e Matemática
# =============================================================================
def metric_card(label: str, value: str, delta: Optional[float] = None):
    """Card de métrica com delta opcional."""
    delta_html = ""
    if isinstance(delta, (int, float)):
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


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def logistic(x: float, k: float = 6.0, x0: float = 0.0) -> float:
    """Curva logística para suavizar impactos (para simuladores)."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def soft_effect(delta_pct: float, weight: float, scale: float = 1.0) -> float:
    """
    Converte uma variação percentual de fator (slider) em efeito,
    usando peso e uma leve não-linearidade.
    """
    # delta_pct em [-1.0, 1.0]
    # Curva: sinal preservado, amplitude cresce de modo crescente
    sign = 1.0 if delta_pct >= 0 else -1.0
    mag = abs(delta_pct)
    curved = (mag ** 1.15)  # leve potência para ampliar fim de curso
    return sign * curved * weight * scale


def weighted_sum(deltas: Dict[str, float], weights: Dict[str, float], scale: float = 1.0) -> float:
    """Soma ponderada de deltas com pesos e escala global."""
    return sum(soft_effect(deltas[k], weights.get(k, 0.0), scale=scale) for k in deltas)


def line_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str, height: int = 320):
    fig = px.line(df, x=x, y=y, color=color, markers=True)
    fig.update_layout(
        title=title, height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(tickformat=",.0%")
    return fig


def bars_pct(df: pd.DataFrame, x: str, y: str, color: str, title: str, height: int = 320, barmode: str = "group"):
    fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, text_auto=".0%")
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(tickformat=",.0%")
    return fig


def bars_num(df: pd.DataFrame, x: str, y: str, color: str, title: str, height: int = 320):
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", text_auto=True)
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def scatter_perf_import(df: pd.DataFrame, x: str, y: str, color: str, text: str, title: str, height: int = 420):
    fig = px.scatter(df, x=x, y=y, color=color, text=text, size_max=18)
    fig.update_traces(textposition="top center")
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(tickformat=",.0%")
    fig.update_yaxes(tickformat=",.0%")
    return fig


def alert_box(titulo: str, texto: str, nivel: str = "normal"):
    cls = "alert"
    if nivel == "critico":
        cls += " alert-critico"
    elif nivel == "alto":
        cls += " alert-alto"
    st.markdown(f"<div class='{cls}'><b>{titulo}</b><br/>{texto}</div>", unsafe_allow_html=True)


# =============================================================================
# Dados base e geradores (bancos)
# =============================================================================
BANKS = ["RedBank", "GreenBank", "GreyBank", "PurpleBank", "BlueBank"]
MEDIA_CHANNELS = ["TV Aberta", "TV Paga", "YouTube", "TikTok", "Instagram", "Google", "Portais", "Rádio", "Out of Home"]

# FATORES e atributos de IMAGEM adequados a bancos (5x5 = 25)
IMAGE_FACTORS: Dict[str, List[str]] = {
    "Confiança & Segurança": ["Confiável", "Transparência", "Segurança de dados", "Solidez financeira", "Privacidade"],
    "Experiência Digital": ["Facilidade no app", "Estabilidade", "Velocidade", "Jornadas sem fricção", "Autoatendimento"],
    "Valor & Tarifas": ["Preço justo", "Clareza de tarifas", "Benefícios", "Programas de pontos", "Custo-benefício"],
    "Atendimento & Suporte": ["Rapidez", "Cordialidade", "Resolução 1º contato", "Multicanal", "Acompanhamento"],
    "Proximidade & Marca": ["Comunicação clara", "Identificação com a marca", "Responsabilidade social", "Inovação", "Recomendação"],
}

EQUITY_KPIS = [
    "Intenção abrir conta", "Intenção recomendar", "Satisfação",
    "Intenção contratar crédito", "Principalidade"
]

FUNNEL_STAGES = ["Top of mind", "Lembrança", "Familiaridade", "Consideração", "Preferência"]


def gen_media_habits(banks: List[str], channels: List[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 100)
    cols = {b: np.clip(rng.dirichlet(np.ones(len(channels))) * 1.8, 0.02, None) for b in banks}
    df = pd.DataFrame({"canal": channels, **cols})
    df["Geral"] = df[banks].mean(axis=1)
    return df


def gen_funnel(banks: List[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 200)
    rows = []
    for brand in banks:
        base = float(np.clip(rng.normal(0.20, 0.06), 0.05, 0.40))            # Top of mind
        lemb = float(np.clip(base + rng.normal(0.25, 0.05), 0.10, 0.85))     # Lembrança
        fam = float(np.clip(lemb + rng.normal(0.10, 0.04), 0.10, 0.95))      # Familiaridade
        cons = float(np.clip(fam - rng.uniform(0.05, 0.20), 0.05, fam))      # Consideração
        pref = float(np.clip(cons - rng.uniform(0.03, 0.12), 0.01, cons))    # Preferência
        vals = [base, lemb, fam, cons, pref]
        for s, v in zip(FUNNEL_STAGES, vals):
            rows.append({"marca": brand, "etapa": s, "valor": v})
    return pd.DataFrame(rows)


def gen_image_scores(banks: List[str], factors: Dict[str, List[str]]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 300)
    rows = []
    # leve vantagem para RedBank/GreyBank em Confiança & Segurança; PurpleBank em Digital; BlueBank em Valor
    boosts = {
        "RedBank": {"Confiança & Segurança": 0.03, "Experiência Digital": 0.00, "Valor & Tarifas": 0.00, "Atendimento & Suporte": 0.01, "Proximidade & Marca": 0.01},
        "GreenBank": {"Confiança & Segurança": 0.00, "Experiência Digital": 0.01, "Valor & Tarifas": 0.01, "Atendimento & Suporte": 0.00, "Proximidade & Marca": 0.00},
        "GreyBank": {"Confiança & Segurança": 0.02, "Experiência Digital": 0.00, "Valor & Tarifas": 0.00, "Atendimento & Suporte": 0.00, "Proximidade & Marca": 0.00},
        "PurpleBank": {"Confiança & Segurança": 0.00, "Experiência Digital": 0.03, "Valor & Tarifas": 0.00, "Atendimento & Suporte": 0.00, "Proximidade & Marca": 0.01},
        "BlueBank": {"Confiança & Segurança": 0.00, "Experiência Digital": 0.00, "Valor & Tarifas": 0.03, "Atendimento & Suporte": 0.01, "Proximidade & Marca": 0.00},
    }
    for brand in banks:
        for f, attrs in factors.items():
            base = float(np.clip(rng.normal(0.58, 0.09), 0.20, 0.95))
            score = clamp(base + boosts.get(brand, {}).get(f, 0.0), 0.10, 0.97)
            rows.append({"marca": brand, "fator": f, "score": score, "hover": ", ".join(attrs)})
    return pd.DataFrame(rows)


def gen_equity(banks: List[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 400)
    data = {"kpi": EQUITY_KPIS}
    for brand in banks:
        # pequenas variações por banco
        shift = {
            "RedBank": 0.03, "GreenBank": 0.00, "GreyBank": 0.01,
            "PurpleBank": 0.01, "BlueBank": 0.00
        }.get(brand, 0.0)
        data[brand] = np.clip(rng.normal(0.58 + shift, 0.07, len(EQUITY_KPIS)), 0.20, 0.95)
    return pd.DataFrame(data)


def gen_priority_matrix(factors: Dict[str, List[str]]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 500)
    rows = []
    for f, attrs in factors.items():
        for a in attrs:
            # performance e importância com diferenças sutis por fator
            perf = float(np.clip(rng.normal({
                "Confiança & Segurança": 0.61,
                "Experiência Digital": 0.57,
                "Valor & Tarifas": 0.52,
                "Atendimento & Suporte": 0.56,
                "Proximidade & Marca": 0.55
            }[f], 0.11), 0.10, 0.97))
            imp = float(np.clip(rng.normal({
                "Confiança & Segurança": 0.62,
                "Experiência Digital": 0.60,
                "Valor & Tarifas": 0.58,
                "Atendimento & Suporte": 0.57,
                "Proximidade & Marca": 0.55
            }[f], 0.12), 0.10, 0.97))
            rows.append({"fator": f, "atributo": a, "performance": perf, "importancia": imp})
    return pd.DataFrame(rows)


def gen_branding_impact() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 600)
    return pd.DataFrame({
        "kpi": ["Intenção abrir conta", "Satisfação", "Intenção recomendar", "Principalidade"],
        "participacao_branding": np.clip(rng.normal(0.68, 0.08, 4), 0.35, 0.90)
    })


# =============================================================================
# Dados – construir base uma vez
# =============================================================================
DF_MEDIA = gen_media_habits(BANKS, MEDIA_CHANNELS)
DF_FUNIL = gen_funnel(BANKS)
DF_IMG = gen_image_scores(BANKS, IMAGE_FACTORS)
DF_EQ = gen_equity(BANKS)
DF_ATTR = gen_priority_matrix(IMAGE_FACTORS)
BRANDING_IMPACT = gen_branding_impact()

# =============================================================================
# Sidebar – sem filtros (apenas header/nota)
# =============================================================================
with st.sidebar:
    st.title("Okiar – Demo")
    st.caption("Frameworks de Consumer/Brand/EX • Dados 100% fictícios • v1.0")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.write(":grey[Use os filtros dentro de cada aba para ajustar as comparações.]")

# =============================================================================
# Abas
# =============================================================================
TAB_BRAIN, TAB_MERIDIO, TAB_MMX, TAB_UXM, TAB_DOMUS, TAB_EBRAIN = st.tabs([
    "BRAIN", "MERIDIO", "MMX", "UXM", "Domus", "e-BRAIN"
])

# =============================================================================
# BRAIN – Brand+Insights
# =============================================================================
with TAB_BRAIN:
    st.title("BRAIN – Brand+Insights")
    st.caption("Módulos: Comportamento • Memória • Imagem • Equity • Estratégia")
    st.divider()

    # ---- Filtros desta aba ----
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        brand_a = st.selectbox("Marca A", options=BANKS, index=0)
    with colB:
        brand_b = st.selectbox("Marca B", options=[b for b in BANKS if b != brand_a], index=1)
    with colC:
        compare_mode = st.radio("Comparar canais de mídia", ["Geral", f"{brand_a} vs {brand_b}"], horizontal=True)

    # ---------------- Comportamento ----------------
    st.subheader("Comportamento – Hábitos de mídia")
    if compare_mode == "Geral":
        media = DF_MEDIA.melt(id_vars=["canal"], value_vars=["Geral"], var_name="amostra", value_name="share")
        fig_m = px.bar(media, x="canal", y="share", text_auto=".0%", title="Hábitos de mídia – Geral")
        fig_m.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        fig_m.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        media = DF_MEDIA.melt(id_vars=["canal"], value_vars=[brand_a, brand_b], var_name="marca", value_name="share")
        fig_m = px.bar(media, x="canal", y="share", color="marca", barmode="group", text_auto=".0%", title="Hábitos de mídia – comparação")
        fig_m.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        fig_m.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_m, use_container_width=True)

    st.divider()

    # ---------------- Memória ----------------
    st.subheader("Memória – Top of mind & Funil de 5 marcas")
    st.caption("Top of mind, lembrança, familiaridade, consideração e preferência.")
    funnel_cur = DF_FUNIL.copy()
    fig_f = bars_pct(funnel_cur, x="etapa", y="valor", color="marca", title="Funil de marcas – Exemplo", height=360, barmode="group")
    st.plotly_chart(fig_f, use_container_width=True)

    st.divider()

    # ---------------- Imagem ----------------
    st.subheader("Imagem – Fatores de percepção (A vs B)")
    st.caption("Passe o mouse para ver os atributos dentro de cada fator.")
    img = DF_IMG[DF_IMG["marca"].isin([brand_a, brand_b])].copy()
    fig_img = px.bar(
        img, x="fator", y="score", color="marca", barmode="group", text_auto=".0%",
        title="Percepção por fator (A vs B)", custom_data=["hover"]
    )
    fig_img.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y:.0%}<br>Atributos: %{customdata[0]}")
    fig_img.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    fig_img.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_img, use_container_width=True)

    st.divider()

    # ---------------- Equity ----------------
    st.subheader("Equity – KPIs (A vs B)")
    eq = DF_EQ.melt(id_vars=["kpi"], value_vars=[brand_a, brand_b], var_name="marca", value_name="score")
    fig_eq = bars_pct(eq, x="kpi", y="score", color="marca", title="KPIs de equity – demonstração", height=340)
    st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    # ---------------- Estratégia ----------------
    st.subheader("Estratégia – Matriz de prioridades e participação do Branding")

    # Matriz perf x importância
    df_pi = DF_ATTR.copy()
    med_perf = df_pi["performance"].median()
    med_imp = df_pi["importancia"].median()
    df_pi["zona"] = np.where(
        (df_pi["performance"] < med_perf) & (df_pi["importancia"] >= med_imp),
        "Urgência",
        np.where((df_pi["performance"] >= med_perf) & (df_pi["importancia"] >= med_imp), "Proteger", "Acompanhar")
    )
    fig_mat = px.scatter(
        df_pi, x="performance", y="importancia", color="zona", text="atributo",
        hover_data=["fator"], title="Matriz de Priorização (Performance × Importância)", size_max=18
    )
    fig_mat.update_traces(textposition="top center")
    fig_mat.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    fig_mat.update_xaxes(tickformat=",.0%")
    fig_mat.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_mat, use_container_width=True)

    # Participação de Branding (R² simulado)
    st.markdown("<div class='block-title'>Participação de Branding no resultado</div>", unsafe_allow_html=True)
    fig_r2 = px.bar(BRANDING_IMPACT, x="kpi", y="participacao_branding", text_auto=".0%", title="")
    fig_r2.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    fig_r2.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_r2, use_container_width=True)


# =============================================================================
# MERIDIO – Consumer Behavior
# =============================================================================
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

    PERSONAS = sum(PERSONA_GROUPS.values(), [])
    PERSONA_MEDIA = {p: persona_adjustment(p, BASE_MEDIA) for p in PERSONAS}
    PERSONA_FREQ = {p: float(np.clip(np.random.normal(2.6, 0.6), 0.5, 5.0)) for p in PERSONAS}  # vezes/mês
    PERSONA_TICKET = {p: float(np.clip(np.random.normal(180, 60), 40, 600)) for p in PERSONAS}  # R$

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
    dtab1, dtab2, dtab3 = st.tabs(["Drivers de Compra", "Offers (Conjoint & Preço)", "Experimento"])

    # --- Drivers de compra (dermocosméticos fictício) + simulador de intenção ---
    with dtab1:
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

        fig_d1 = px.scatter(DRIVERS, x="performance", y="importancia", text="atributo", color="importancia",
                            color_continuous_scale="Tealgrn", title="Matriz de drivers (dermocosméticos)")
        fig_d1.update_traces(textposition="top center")
        fig_d1.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), coloraxis_colorbar_title="Importância")
        fig_d1.update_xaxes(tickformat=",.0%")
        fig_d1.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_d1, use_container_width=True)

        st.markdown("<div class='block-title'>Simulador de intenção de compra (4 fatores agregados)</div>", unsafe_allow_html=True)
        # Fatores agregados (4) para o simulador
        AGG_FACTORS = {
            "Produto": ["Qualidade", "Textura", "Aroma", "Hidratação", "Eficácia comprovada"],
            "Marca": ["Força de marca", "Buzz", "Indicação de influenciadores", "Sustentabilidade", "Ingredientes naturais"],
            "Preço & Oferta": ["Preço justo", "Variedade", "Disponibilidade"],
            "Embalagem": ["Embalagem", "Dermatologicamente testado", "Benefício anti-idade"],
        }
        AGG_WEIGHTS = {"Produto": 0.40, "Marca": 0.28, "Preço & Oferta": 0.22, "Embalagem": 0.10}
        BASE_INTENCAO = 0.52

        sc1, sc2 = st.columns((2,1))
        with sc1:
            sliders = {}
            hover_text = {k: ", ".join(v) for k, v in AGG_FACTORS.items()}
            for k in AGG_FACTORS.keys():
                sliders[k] = st.slider(f"{k}", min_value=-30, max_value=30, value=0, step=1, help=f"Inclui: {hover_text[k]}")
        with sc2:
            # Variação não-linear + pesos (efeitos mais fortes)
            delta_perc = {k: sliders[k]/100 for k in sliders}  # [-0.3, 0.3]
            # Escala maior para tornar efeito perceptível (ex.: 1.2)
            efeito = weighted_sum(delta_perc, AGG_WEIGHTS, scale=1.2)
            inten = clamp(BASE_INTENCAO * (1 + efeito), 0.01, 0.99)
            # Tabela de impactos
            imp_rows = [{"Fator": k, "% Δ fator": f"{delta_perc[k]*100:+.0f}%", "Peso": f"{AGG_WEIGHTS[k]*100:.0f}%"} for k in sliders]
            df_imp = pd.DataFrame(imp_rows)
            st.dataframe(df_imp, use_container_width=True, height=180)
            st.markdown("<div class='block-title'>Intenção de compra estimada</div>", unsafe_allow_html=True)
            metric_card("Intenção", pct(inten))

    # --- Offers: Conjoint (link) + Van Westendorp (grande, com faixa ideal) ---
    with dtab2:
        st.markdown("**Conjoint**")
        st.link_button("Abrir simulador de Conjoint", "https://simuladorconjointtdah-g3tyutubdwrqowizlogkuc.streamlit.app")

        # Van Westendorp maior e com cálculo do range ideal
        st.markdown("**Price – Van Westendorp**")
        prices = np.linspace(20, 200, 60)

        rng = np.random.default_rng(SEED + 700)
        muito_barato = np.clip(np.linspace(0.02, 0.85, 60) + rng.normal(0, 0.02, 60), 0, 1)
        barato =      np.clip(np.linspace(0.10, 0.90, 60) + rng.normal(0, 0.02, 60), 0, 1)
        caro =        np.clip(np.linspace(0.01, 0.75, 60)[::-1] + rng.normal(0, 0.02, 60), 0, 1)
        muito_caro =  np.clip(np.linspace(0.05, 0.80, 60)[::-1] + rng.normal(0, 0.02, 60), 0, 1)

        def _intersect_x(x, y1, y2) -> Optional[float]:
            """Retorna ponto de interseção aproximado entre y1 e y2 (por x), se houver."""
            diff = y1 - y2
            sign = np.sign(diff)
            # procura mudança de sinal
            idx = np.where(np.diff(sign) != 0)[0]
            if len(idx) == 0:
                return None
            i = idx[0]
            # interpolação linear
            x0, x1 = x[i], x[i+1]
            y0, y1v = diff[i], diff[i+1]
            if y1v == y0:
                return float((x0 + x1)/2)
            t = -y0 / (y1v - y0)
            return float(x0 + t*(x1 - x0))

        # Interseções típicas: Barato × Caro (Ponto de marginal barato/caro)
        x_bc = _intersect_x(prices, barato, caro)
        # Muito barato × Muito caro (Range exterior)
        x_mbmc = _intersect_x(prices, muito_barato, muito_caro)

        # Faixa ideal (se ambas interseções existirem): entre x_mbmc (inferior) e x_bc (superior), ordene
        faixa = None
        if x_bc and x_mbmc:
            x_low, x_high = sorted([x_mbmc, x_bc])
            faixa = (x_low, x_high)

        fig_vw = go.Figure()
        fig_vw.add_trace(go.Scatter(x=prices, y=muito_barato, name="Muito barato"))
        fig_vw.add_trace(go.Scatter(x=prices, y=barato, name="Barato"))
        fig_vw.add_trace(go.Scatter(x=prices, y=caro, name="Caro"))
        fig_vw.add_trace(go.Scatter(x=prices, y=muito_caro, name="Muito caro"))

        if faixa:
            # Sombreamento da faixa ideal
            fig_vw.add_vrect(x0=faixa[0], x1=faixa[1], fillcolor="LightGreen", opacity=0.3, layer="below", line_width=0)
            fig_vw.add_vline(x=faixa[0], line=dict(color="Green", dash="dash"), annotation_text=f"Limite inf ~ R$ {faixa[0]:.0f}", annotation_position="top left")
            fig_vw.add_vline(x=faixa[1], line=dict(color="Green", dash="dash"), annotation_text=f"Limite sup ~ R$ {faixa[1]:.0f}", annotation_position="top right")

        fig_vw.update_layout(
            title="Curvas de sensibilidade de preço (Van Westendorp)",
            xaxis_title="Preço (R$)",
            yaxis_title="% acumulado",
            height=480,  # maior
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_vw, use_container_width=True)

        if faixa:
            st.success(f"Faixa ideal (estimada): **R$ {faixa[0]:.0f} – R$ {faixa[1]:.0f}**.")
        else:
            st.warning("Não foi possível identificar uma faixa ideal clara com os dados simulados.")

    # --- Experimento: A/B de embalagem (dermocosméticos) ---
    with dtab3:
        st.markdown("<div class='block-title'>Teste A/B – Embalagens</div>", unsafe_allow_html=True)
        PACK_FACTORS = ["Preço justo", "Qualidade", "Força de marca", "Premiumidade", "Intenção de compra", "Intenção pagar mais"]
        PACK = pd.DataFrame({
            "fator": PACK_FACTORS,
            "A": np.clip(np.random.normal(70, 10, len(PACK_FACTORS)), 30, 95),
            "B": np.clip(np.random.normal(75, 10, len(PACK_FACTORS)), 30, 98),
        })
        pack_m = PACK.melt(id_vars=["fator"], value_vars=["A", "B"], var_name="Emb.", value_name="score")
        fig_pack = px.bar(pack_m, x="fator", y="score", color="Emb.", barmode="group", text_auto=True,
                          title="Desempenho por fator (A vs B)")
        fig_pack.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pack, use_container_width=True)

    st.divider()

    # ---------------- Jornada ----------------
    st.subheader("Jornada – Exemplo (macro-etapas)")
    steps = ["Descoberta", "Pesquisa", "Avaliação", "Compra", "Pós-compra"]
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

# =============================================================================
# MMX – Customer Experience (Seguros)
# =============================================================================
with TAB_MMX:
    st.title("MMX – Customer Experience (Seguros)")
    st.caption("Overview • Evolução • Priorização • Simulador")
    st.divider()

    np.random.seed(SEED + 17)
    clientes = ["Cliente A", "Cliente B"]
    fatores_mmx = {
        "Atendimento": ["Cordialidade", "Agilidade", "Resolução no 1º contato"],
        "Canais Digitais": ["Facilidade no app", "Estabilidade", "Autoatendimento"],
        "Produto/Serviços": ["Coberturas", "Clareza de contrato", "Adequação ao perfil"],
        "Preço/Valor": ["Preço percebido", "Custo-benefício", "Transparência de reajuste"],
        "Suporte/Resolução": ["Pós-sinistro", "Prazo de retorno", "Acompanhamento do caso"],
    }

    def mock_attr_scores(base_shift=0.0):
        rows = []
        for fator, attrs in fatores_mmx.items():
            for a in attrs:
                score = float(np.clip(np.random.normal(0.62 + base_shift, 0.10), 0.25, 0.95))
                importancia = float(np.clip(np.random.normal(0.60, 0.10), 0.15, 0.95))
                rows.append({"fator": fator, "atributo": a, "score": score, "importancia": importancia})
        return pd.DataFrame(rows)

    mmx_A = mock_attr_scores(base_shift=0.03)
    mmx_B = mock_attr_scores(base_shift=-0.02)

    st.subheader("Overview – Experiência por fator (Cliente A vs Cliente B)")
    fa = mmx_A.groupby("fator", as_index=False)["score"].mean().rename(columns={"score": "Cliente A"})
    fb = mmx_B.groupby("fator", as_index=False)["score"].mean().rename(columns={"score": "Cliente B"})
    fcmp = fa.merge(fb, on="fator")
    fcmp_m = fcmp.melt(id_vars=["fator"], var_name="cliente", value_name="score")
    fig_ov = px.bar(fcmp_m, x="fator", y="score", color="cliente", barmode="group", text_auto=".0%")
    fig_ov.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), title="Média de experiência por fator")
    fig_ov.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_ov, use_container_width=True)

    st.divider()

    st.subheader("Evolução – Satisfação e NPS (últimos 12 meses)")
    meses = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="MS")
    base_sat = np.linspace(0.62, 0.70, 12)
    base_nps = np.linspace(0.28, 0.40, 12)
    ruido_sat_A = np.random.normal(0, 0.01, 12)
    ruido_sat_B = np.random.normal(0, 0.01, 12)
    ruido_nps_A = np.random.normal(0, 0.02, 12)
    ruido_nps_B = np.random.normal(0, 0.02, 12)
    evo = pd.DataFrame({
        "mes": list(meses)*2,
        "cliente": ["Cliente A"]*12 + ["Cliente B"]*12,
        "satisfacao": list(np.clip(base_sat + 0.02 + ruido_sat_A, 0.30, 0.95)) + list(np.clip(base_sat - 0.01 + ruido_sat_B, 0.30, 0.95)),
        "nps": list(np.clip(base_nps + 0.05 + ruido_nps_A, -0.20, 0.95)) + list(np.clip(base_nps - 0.02 + ruido_nps_B, -0.20, 0.95)),
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

    st.divider()

    st.subheader("Priorização – Matriz Performance × Importância (15 atributos)")
    pri = pd.concat([mmx_A.assign(cliente="Cliente A"), mmx_B.assign(cliente="Cliente B")], ignore_index=True)
    med_perf = pri["score"].median()
    med_imp = pri["importancia"].median()
    pri["zona"] = np.where(
        (pri["score"] < med_perf) & (pri["importancia"] >= med_imp), "Urgência",
        np.where((pri["score"] >= med_perf) & (pri["importancia"] >= med_imp), "Proteger", "Acompanhar")
    )
    fig_pri = px.scatter(pri, x="score", y="importancia", color="zona", symbol="cliente", text="atributo",
                         hover_data=["fator"], title="Atributos prioritários por cliente", size_max=18)
    fig_pri.update_traces(textposition="top center")
    fig_pri.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    fig_pri.update_xaxes(title="Performance", tickformat=",.0%")
    fig_pri.update_yaxes(title="Importância", tickformat=",.0%")
    st.plotly_chart(fig_pri, use_container_width=True)

    st.divider()

    st.subheader("Simulador – Impacto dos fatores em Satisfação → KPIs")
    st.caption("Ajuste os 5 fatores (±30%). Satisfação impacta: NPS (↑), Churn (↓), Reclamações (↓), Cross-sell (↑).")
    pesos_fator = {
        "Atendimento": 0.28,
        "Canais Digitais": 0.22,
        "Produto/Serviços": 0.24,
        "Preço/Valor": 0.14,
        "Suporte/Resolução": 0.12,
    }
    colL, colR = st.columns((2,1))
    with colL:
        deltas = {f: st.slider(f, -30, 30, 0, 1) for f in pesos_fator}
        delta_norm = {k: v/100 for k, v in deltas.items()}
        efeito = weighted_sum(delta_norm, pesos_fator, scale=1.35)
        sat = clamp(0.66 * (1 + efeito), 0.01, 0.99)
        nps = clamp(0.10 + 1.05*sat, 0.00, 0.99)
        churn = clamp(0.30 - 0.40*sat, 0.02, 0.45)
        recl = clamp(0.32 - 0.45*sat, 0.01, 0.40)
        cross = clamp(0.08 + 0.90*sat, 0.02, 0.95)
    with colR:
        metric_card("Satisfação (estimada)", f"{sat*100:.1f}%")
        metric_card("NPS (estimado)", f"{nps*100:.1f}%")
        metric_card("Churn (estimado)", f"{churn*100:.1f}%")
        metric_card("Reclamações (estimado)", f"{recl*100:.1f}%")
        metric_card("Cross-sell (estimado)", f"{cross*100:.1f}%")
    st.markdown("**Impactos por fator**")
    imp_rows = [{"Fator": f, "% Δ fator": f"{deltas[f]:+.0f}%", "Peso": f"{pesos_fator[f]*100:.0f}%"} for f in pesos_fator]
    st.dataframe(pd.DataFrame(imp_rows), use_container_width=True, height=180)

# =============================================================================
# UXM – Digital Experience
# =============================================================================
with TAB_UXM:
    st.title("UXM – Digital Experience")
    st.caption("Overview • Evolução • Priorização • Simulador")
    st.divider()

    np.random.seed(SEED + 23)
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
                w = float(np.clip(np.random.normal(0.58, 0.10), 0.10, 0.95))
                rows.append({"fator": f, "atributo": a, "score": s, "importancia": w})
        return pd.DataFrame(rows)

    uxm_A = mock_ux_scores(shift=0.03)
    uxm_B = mock_ux_scores(shift=-0.02)

    st.subheader("Overview – Big Five de UX (A vs B)")
    fa = uxm_A.groupby("fator", as_index=False)["score"].mean().rename(columns={"score": "Marca A"})
    fb = uxm_B.groupby("fator", as_index=False)["score"].mean().rename(columns={"score": "Marca B"})
    fcmp = fa.merge(fb, on="fator").melt(id_vars=["fator"], var_name="marca", value_name="score")
    fig_ux = px.bar(fcmp, x="fator", y="score", color="marca", barmode="group", text_auto=".0%")
    fig_ux.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), title="Scores por fator")
    fig_ux.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_ux, use_container_width=True)

    st.divider()

    st.subheader("Evolução – UX Equity (últimos 12 meses)")
    meses = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="MS")
    base_uxe = np.linspace(0.55, 0.68, 12)
    ruido_A = np.random.normal(0, 0.01, 12)
    ruido_B = np.random.normal(0, 0.01, 12)
    evo = pd.DataFrame({
        "mes": list(meses)*2,
        "marca": ["Marca A"]*12 + ["Marca B"]*12,
        "ux_equity": list(np.clip(base_uxe + 0.02 + ruido_A, 0.25, 0.95)) + list(np.clip(base_uxe - 0.01 + ruido_B, 0.25, 0.95))
    })
    fig_uxe = px.line(evo, x="mes", y="ux_equity", color="marca", markers=True, title="UX Equity (normalizado)")
    fig_uxe.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    fig_uxe.update_yaxes(tickformat=",.0%")
    st.plotly_chart(fig_uxe, use_container_width=True)

    st.divider()

    st.subheader("Priorização – Performance × Importância (Big Five detalhado)")
    pri = pd.concat([uxm_A.assign(marca="A"), uxm_B.assign(marca="B")], ignore_index=True)
    med_perf = pri["score"].median()
    med_imp = pri["importancia"].median()
    pri["zona"] = np.where(
        (pri["score"] < med_perf) & (pri["importancia"] >= med_imp), "Urgência",
        np.where((pri["score"] >= med_perf) & (pri["importancia"] >= med_imp), "Proteger", "Acompanhar")
    )
    fig_pu = px.scatter(pri, x="score", y="importancia", color="zona", symbol="marca", text="atributo",
                        hover_data=["fator"], title="Detalhamento de atributos (Big Five)", size_max=18)
    fig_pu.update_traces(textposition="top center")
    fig_pu.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    fig_pu.update_xaxes(title="Performance", tickformat=",.0%")
    fig_pu.update_yaxes(title="Importância", tickformat=",.0%")
    st.plotly_chart(fig_pu, use_container_width=True)

    st.divider()

    st.subheader("Simulador – Big Five → UX Equity → Métricas de negócio")
    st.caption("Ajuste os 5 fatores (±30%). UX Equity impacta: Conversão (↑), Retenção (↑), Suporte (↓).")
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
        d = {f: st.slider(f, -30, 30, 0, 1) for f in pesos}
        d_norm = {k: v/100 for k, v in d.items()}
        efeito = weighted_sum(d_norm, pesos, scale=1.30)
        ux_equity = clamp(base_ux * (1 + efeito), 0.01, 0.99)
        metric_card("UX Equity (estimado)", f"{ux_equity*100:.1f}%")
    with col2:
        conversao = clamp(0.06 + 1.05*ux_equity, 0.01, 0.98)
        retencao = clamp(0.60 + 0.45*ux_equity, 0.20, 0.99)
        suporte = clamp(0.40 - 0.55*ux_equity, 0.01, 0.50)
        metric_card("Conversão (estimada)", f"{conversao*100:.1f}%")
        metric_card("Retenção (estimada)", f"{retencao*100:.1f}%")
        metric_card("Chamados de suporte (↓)", f"{suporte*100:.1f}%")
    with col3:
        imp_rows = [{"Fator": f, "% Δ fator": f"{d[f]:+.0f}%", "Peso": f"{pesos[f]*100:.0f}%"} for f in pesos]
        st.dataframe(pd.DataFrame(imp_rows), use_container_width=True, height=220)
        fig_k = px.bar(pd.DataFrame({
            "kpi": ["Conversão","Retenção","Suporte (↓)"],
            "valor": [conversao, retencao, suporte]
        }), x="kpi", y="valor", text_auto=".0%")
        fig_k.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
        fig_k.update_yaxes(tickformat=",.0%")
        st.plotly_chart(fig_k, use_container_width=True)

# =============================================================================
# DOMUS – Employee Experience (Colaboradores)
# =============================================================================
with TAB_DOMUS:
    st.title("Domus – Employee Experience")
    st.caption("Pilares de Imagem • Evolução • Priorização • Simulador (EX → KPIs de Pessoas)")
    st.divider()

    PILARES = {
        "Conviver": ["Cultura", "Ambiente", "Flexibilidade", "Liderança & Colegas", "Identificação"],
        "Ser": ["Carreira", "Aprendizado", "Visibilidade", "Reconhecimento", "Remuneração"],
        "Viver": ["Work-life balance", "Estabilidade", "Benefícios", "Carga de trabalho", "Autonomia"],
        "Inspirar": ["Reputação", "Diversidade & ESG", "Inovação", "Propósito", "Polêmica"],
    }

    np.random.seed(SEED + 101)
    pilares_rows = []
    for pilar, attrs in PILARES.items():
        for contexto in ["Colaboradores", "Mercado de talentos"]:
            score = float(np.clip(np.random.normal(0.64 if contexto=="Colaboradores" else 0.58, 0.08), 0.20, 0.95))
            pilares_rows.append({"pilar": pilar, "contexto": contexto, "score": score, "hover": ", ".join(attrs)})
    df_pilares = pd.DataFrame(pilares_rows)

    st.subheader("Pilares de imagem (Colaboradores × Mercado)")
    fig_domus_img = px.bar(
        df_pilares, x="pilar", y="score", color="contexto", barmode="group",
        text_auto=".0%", hover_data={"hover": True, "pilar": False, "score":":.0%", "contexto": False},
        title="Conviver • Ser • Viver • Inspirar"
    )
    fig_domus_img.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y:.0%}<br>Atributos: %{customdata[0]}")
    fig_domus_img.update_yaxes(tickformat=",.0%")
    fig_domus_img.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_domus_img, use_container_width=True)

    st.divider()

    st.subheader("Evolução – EX (eNPS) e Turnover (últimos 12 meses)")
    meses = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq="MS")
    eA = np.clip(np.linspace(0.30, 0.45, 12) + np.random.normal(0, 0.01, 12), -0.2, 0.95)
    eB = np.clip(np.linspace(0.22, 0.35, 12) + np.random.normal(0, 0.01, 12), -0.2, 0.95)
    tA = np.clip(np.linspace(0.20, 0.16, 12) + np.random.normal(0, 0.005, 12), 0.02, 0.40)
    tB = np.clip(np.linspace(0.24, 0.21, 12) + np.random.normal(0, 0.005, 12), 0.02, 0.40)
    ex_evo = pd.DataFrame({
        "mes": list(meses)*2,
        "grupo": ["Colaboradores"]*12 + ["Mercado de talentos"]*12,
        "eNPS": list(eA) + list(eB),
        "turnover": list(tA) + list(tB),
    })
    c1, c2 = st.columns(2)
    with c1:
        fig_enps = px.line(ex_evo, x="mes", y="eNPS", color="grupo", markers=True, title="eNPS (normalizado)")
        fig_enps.update_yaxes(tickformat=",.0%")
        fig_enps.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_enps, use_container_width=True)
    with c2:
        fig_to = px.line(ex_evo, x="mes", y="turnover", color="grupo", markers=True, title="Turnover (↓ é melhor)")
        fig_to.update_yaxes(tickformat=",.0%")
        fig_to.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_to, use_container_width=True)

    st.divider()

    st.subheader("Priorização – Performance × Importância (Pilares e seus atributos)")
    pr_rows = []
    for pilar, attrs in PILARES.items():
        for a in attrs:
            perf = float(np.clip(np.random.normal(0.60, 0.12), 0.10, 0.95))
            imp = float(np.clip(np.random.normal(0.58, 0.15), 0.10, 0.95))
            pr_rows.append({"pilar": pilar, "atributo": a, "performance": perf, "importancia": imp})
    df_pr = pd.DataFrame(pr_rows)
    med_perf = df_pr["performance"].median()
    med_imp = df_pr["importancia"].median()
    df_pr["zona"] = np.where(
        (df_pr["performance"] < med_perf) & (df_pr["importancia"] >= med_imp), "Desenvolver agora!",
        np.where((df_pr["performance"] >= med_perf) & (df_pr["importancia"] >= med_imp), "Comunicar agora!", "Acompanhar")
    )
    fig_pr = px.scatter(df_pr, x="performance", y="importancia", color="zona", text="atributo", hover_data=["pilar"],
                        title="Matriz de Prioridades (EX)", size_max=18)
    fig_pr.update_traces(textposition="top center")
    fig_pr.update_xaxes(tickformat=",.0%", title="Performance")
    fig_pr.update_yaxes(tickformat=",.0%", title="Importância")
    fig_pr.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()

    st.subheader("Simulador – EX (pilares) → KPIs de Pessoas")
    st.caption("Ajuste os 4 pilares (±30%). EX impacta eNPS (↑), Turnover (↓), Reclamações internas (↓), Produtividade (↑).")
    pesos = {"Conviver": 0.28, "Ser": 0.26, "Viver": 0.22, "Inspirar": 0.24}
    base_ex = 0.60
    cL, cM, cR = st.columns((1,1,1))
    with cL:
        deltas = {p: st.slider(p, -30, 30, 0, 1, help=", ".join(PILARES[p])) for p in pesos}
        d_norm = {k: v/100 for k, v in deltas.items()}
        efeito = weighted_sum(d_norm, pesos, scale=1.30)
        ex_equity = clamp(base_ex * (1 + efeito), 0.01, 0.99)
        metric_card("EX (estimado)", f"{ex_equity*100:.1f}%")
    with cM:
        enps = clamp(0.05 + 1.10*ex_equity, -0.20, 0.98)
        turnover = clamp(0.32 - 0.50*ex_equity, 0.02, 0.45)
        reclama = clamp(0.30 - 0.45*ex_equity, 0.01, 0.45)
        prod = clamp(0.55 + 0.60*ex_equity, 0.10, 0.99)
        metric_card("eNPS (↑)", f"{enps*100:.1f}%")
        metric_card("Turnover (↓)", f"{turnover*100:.1f}%")
        metric_card("Reclamações (↓)", f"{reclama*100:.1f}%")
        metric_card("Produtividade (↑)", f"{prod*100:.1f}%")
    with cR:
        imp_rows = [{"Pilar": p, "% Δ pilar": f"{deltas[p]:+.0f}%", "Peso": f"{pesos[p]*100:.0f}%"} for p in pesos]
        st.dataframe(pd.DataFrame(imp_rows), use_container_width=True, height=212)
        fig_kpis = px.bar(pd.DataFrame({
            "kpi": ["eNPS (↑)", "Turnover (↓)", "Reclamações (↓)", "Produtividade (↑)"],
            "valor": [enps, turnover, reclama, prod]
        }), x="kpi", y="valor", text_auto=".0%")
        fig_kpis.update_yaxes(tickformat=",.0%")
        fig_kpis.update_layout(height=212, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_kpis, use_container_width=True)

# =============================================================================
# e-BRAIN – Employer Brand Insights (marca empregadora)
# =============================================================================
with TAB_EBRAIN:
    st.title("e-BRAIN – Employer Brand Insights")
    st.caption("Comportamento • Memória • Imagem • Equity • Estratégia")
    st.divider()

    PILARES_E = {
        "Conviver": ["Cultura", "Ambiente", "Flexibilidade", "Liderança & Colegas", "Identificação"],
        "Ser": ["Carreira", "Aprendizado", "Visibilidade", "Reconhecimento", "Remuneração"],
        "Viver": ["Work-life balance", "Estabilidade", "Benefícios", "Carga de trabalho", "Autonomia"],
        "Inspirar": ["Reputação", "Diversidade & ESG", "Inovação", "Propósito", "Polêmica"],
    }

    st.subheader("Comportamento – Onde talentos buscam e avaliam empregadores")
    canais = ["LinkedIn", "Google", "Glassdoor", "Instagram", "YouTube", "Comunidades", "Indicações", "Portais de Vagas"]
    share = np.clip(np.random.dirichlet(np.ones(len(canais))) * 1.8, 0.03, None)
    df_cb = pd.DataFrame({"canal": canais, "share": share})
    fig_cb = px.bar(df_cb, x="canal", y="share", text_auto=".0%", title="Canais de busca/avaliação")
    fig_cb.update_yaxes(tickformat=",.0%")
    fig_cb.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_cb, use_container_width=True)

    st.divider()

    st.subheader("Memória – Funil de marca empregadora")
    etapas = ["Lembrança", "Familiaridade", "Consideração", "Preferência"]
    marcas = ["Marca A", "Marca B", "Marca C", "Marca D", "Marca E"]
    funil_rows = []
    for m in marcas:
        base = np.clip(np.random.normal(0.25, 0.06), 0.08, 0.40)
        fam = np.clip(base + np.random.normal(0.20, 0.05), 0.12, 0.90)
        cons = np.clip(fam - np.random.uniform(0.05, 0.18), 0.05, fam)
        pref = np.clip(cons - np.random.uniform(0.03, 0.12), 0.01, cons)
        vals = [base, fam, cons, pref]
        for e, v in zip(etapas, vals):
            funil_rows.append({"marca": m, "etapa": e, "valor": float(v)})
    df_funil_e = pd.DataFrame(funil_rows)
    fig_f = px.bar(df_funil_e, x="etapa", y="valor", color="marca", barmode="group", text_auto=".0%",
                   title="Lembrança • Familiaridade • Consideração • Preferência")
    fig_f.update_yaxes(tickformat=",.0%")
    fig_f.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_f, use_container_width=True)

    st.divider()

    st.subheader("Imagem – Pilares de percepção (A vs B)")
    marcas_comp = ["Marca A", "Marca B"]
    img_rows = []
    for marca in marcas_comp:
        for pilar, attrs in PILARES_E.items():
            score = float(np.clip(np.random.normal(0.62 if marca=="Marca A" else 0.58, 0.08), 0.20, 0.95))
            img_rows.append({"marca": marca, "pilar": pilar, "score": score, "hover": ", ".join(attrs)})
    df_img_e = pd.DataFrame(img_rows)
    fig_i = px.bar(df_img_e, x="pilar", y="score", color="marca", barmode="group", text_auto=".0%",
                   hover_data={"hover": True, "pilar": False, "score":":.0%", "marca": False},
                   title="Conviver • Ser • Viver • Inspirar")
    fig_i.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y:.0%}<br>Atributos: %{customdata[0]}")
    fig_i.update_yaxes(tickformat=",.0%")
    fig_i.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_i, use_container_width=True)

    st.divider()

    st.subheader("Equity – KPIs (A vs B)")
    kpis = ["Me candidataria", "Aceitaria oferta menor", "Recomendaria", "Não trocaria de empresa", "É meu sonho trabalhar lá", "Daria meu máximo", "eNPS"]
    eq_vals = pd.DataFrame({
        "kpi": kpis,
        "Marca A": np.clip(np.random.normal(0.60, 0.08, len(kpis)), 0.20, 0.95),
        "Marca B": np.clip(np.random.normal(0.54, 0.08, len(kpis)), 0.20, 0.92),
    })
    eqm = eq_vals.melt(id_vars=["kpi"], value_vars=["Marca A","Marca B"], var_name="marca", value_name="score")
    fig_eq = px.bar(eqm, x="kpi", y="score", color="marca", barmode="group", text_auto=".0%", title="KPIs de Employer Brand")
    fig_eq.update_yaxes(tickformat=",.0%")
    fig_eq.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    st.subheader("Estratégia – Priorização & EBIX (Participação de Branding no resultado)")
    pr_rows = []
    for pilar, attrs in PILARES_E.items():
        for a in attrs:
            perf = float(np.clip(np.random.normal(0.58, 0.12), 0.10, 0.95))
            imp = float(np.clip(np.random.normal(0.60, 0.15), 0.10, 0.95))
            pr_rows.append({"pilar": pilar, "atributo": a, "performance": perf, "importancia": imp})
    df_pi_e = pd.DataFrame(pr_rows)
    med_perf = df_pi_e["performance"].median()
    med_imp = df_pi_e["importancia"].median()
    df_pi_e["zona"] = np.where(
        (df_pi_e["performance"] < med_perf) & (df_pi_e["importancia"] >= med_imp), "Desenvolver agora!",
        np.where((df_pi_e["performance"] >= med_perf) & (df_pi_e["importancia"] >= med_imp), "Comunicar agora!", "Acompanhar")
    )
    fig_pi = px.scatter(df_pi_e, x="performance", y="importancia", color="zona", text="atributo", hover_data=["pilar"],
                        title="Matriz de Prioridades (Employer Brand)", size_max=18)
    fig_pi.update_traces(textposition="top center")
    fig_pi.update_xaxes(tickformat=",.0%", title="Performance")
    fig_pi.update_yaxes(tickformat=",.0%", title="Importância")
    fig_pi.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_pi, use_container_width=True)

    kpis_res = ["Atração (me candidaria)", "Retenção (não trocaria)", "Aceite de ofertas", "Engajamento (daria meu máximo)", "eNPS"]
    ebix = pd.DataFrame({
        "kpi": kpis_res,
        "participacao_branding": np.clip(np.random.normal(0.64, 0.10, len(kpis_res)), 0.30, 0.92)
    })
    fig_ebix = px.bar(ebix, x="kpi", y="participacao_branding", text_auto=".0%", title="EBIX – Participação de Branding no resultado")
    fig_ebix.update_yaxes(tickformat=",.0%")
    fig_ebix.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_ebix, use_container_width=True)

st.divider()
st.write(":grey[Okiar • Dados 100% fictícios • v1.0 – BRAIN • MERIDIO • MMX • UXM • Domus • e-BRAIN]")
