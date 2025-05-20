"""
app.py – Streamlit front‑end for NeuroTrader Pro
Rev 2025‑05‑14: new sidebar menu, live metrics, grouped modules,
                better Risk‑/Theme‑/Collective‑views, graceful fall‑backs
"""

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.figure_factory import create_annotated_heatmap   # ← keep this line
from sklearn.decomposition import PCA
from streamlit.components.v1 import html
from streamlit_elements import elements, dashboard, mui
from utils import get_json, put_json
# ───────────────────────────────────────────  Page config & CSS
import streamlit as st
import requests

API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="Trading‑AI Dashboard")

# ---------------------- cached REST wrappers ------------------------------- #
@st.cache_data(ttl=2.0)  # replaces experimental_memo
def get_status():
    return requests.get(f"{API_BASE}/status", timeout=2).json()


@st.cache_data(ttl=30.0)
def get_vol_profile():
    return requests.get(f"{API_BASE}/volatility_profile", timeout=5).json()

st.set_page_config(
    page_title="NeuroTrader Pro",
    layout="wide",
    page_icon="🧠",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {font-size:1.45rem !important;}
    [data-testid="stMetricLabel"] {opacity:.65;font-size:.85rem !important;}
    [data-testid="stSidebar"]     {background-image:linear-gradient(#2b313e,#1a1d24);}
    .sidebar-menu button          {text-align:left;width:100%;padding:.5rem .75rem;
                                   border:none;background:transparent;color:#e5e7eb;font-weight:500;}
    .sidebar-menu button:hover    {background:rgba(255,255,255,.06);}
    .sidebar-menu .active         {background:rgba(79,70,229,.35);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────  Sidebar nav

SIDEBAR_PAGES = [
    ("🧠 Neural Dashboard", "dashboard"),
    ("📈 Market Hologram", "hologram"),
    ("⚙️ Adaptive Modules", "modules"),
    ("📊 Quantum Ledger", "ledger"),
    ("🔍 Insight Engine", "insight"),
    ("🌐 Market Reality", "reality"),
    ("🛡️ Risk Matrix", "risk"),
    ("👥 Opponent Arena", "arena"),
]

with st.sidebar:
    st.title("NeuroTrader Pro")
    st.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)
    if "page" not in st.session_state:
        st.session_state.page = SIDEBAR_PAGES[0][1]

    for label, key in SIDEBAR_PAGES:
        if st.button(
            label,
            key=f"nav-{key}",
            type="primary" if st.session_state.page == key else "secondary",
        ):
            st.session_state.page = key
    st.markdown("</div>", unsafe_allow_html=True)

    # Trading personality selector
    st.subheader("Trading Personality")
    mode = st.selectbox("Risk Profile", ["safe", "aggressive", "extreme"], index=1)
    try:
        if put_json("/mode", {"mode": mode}):
            st.success(f"Switched to {mode.upper()} mode")
    except Exception as e:
        st.error(f"Mode change failed: {e}")
    st.markdown(
        f'<span style="opacity:.8">Current:</span> <strong>{mode.upper()}</strong>',
        unsafe_allow_html=True,
    )

page = st.session_state.page

# ───────────────────────────────────────────  Helper live metrics

def get_active_positions() -> int:
    trades = get_json("/trades") or []
    return len([t for t in trades if not t.get("exit_reason")])

def get_risk_pct() -> float:
    rx = get_json("/risk_exposure") or {}
    return round(sum(rx.get("values", [])) * 100, 2)

# ───────────────────────────────────────────  Neural Dashboard

if page == "dashboard":
    st.markdown("## Neural Trading Dashboard")

    # summary metrics
    status = get_json("/status") or {}
    cols = st.columns([2, 1, 1, 1, 1, 2])
    with cols[0]:
        st.metric("Neuro‑Balance", f"${status.get('balance',0):,.2f}")
    with cols[1]:
        st.metric("Drawdown", f"{status.get('drawdown',0)*100:.2f}%")
    with cols[2]:
        st.metric("Clusters", status.get("clusters", 0))
    with cols[3]:
        st.metric("Playbook", status.get("playbook_size", 0))
    with cols[4]:
        st.metric("Live PnL", f"${status.get('pnl',0):,.2f}")

    col1, col2 = st.columns([3, 2], gap="large")

    # Market consciousness (quantum graph)
# ── Market Consciousness (quantum graph) ────────────────────────────────
    with col1:
        st.subheader("Market Consciousness")
        data = get_json("/quantum_map") or {}
        if data.get("nodes"):
            G = nx.DiGraph()
            for n in data["nodes"]:
                G.add_node(n["id"], size=n["size"] * 2, color=n["color"])
            for e in data["links"]:
                G.add_edge(e["source"], e["target"], weight=e["value"] * 2)

            pos = nx.spring_layout(G, seed=42, k=0.5)

            # edges
            ex, ey = [], []
            for a, b in G.edges():
                x0, y0 = pos[a]
                x1, y1 = pos[b]
                ex += [x0, x1, None]
                ey += [y0, y1, None]
            edge_trace = go.Scatter(
                x=ex, y=ey,
                mode="lines",
                line=dict(width=1.5, color="rgba(79,70,229,0.35)"),
                hoverinfo="none",
            )

            # nodes
            nx_, ny_, sizes, colors, labels = [], [], [], [], []
            for n in G.nodes():
                x, y = pos[n]
                nx_.append(x); ny_.append(y)
                sizes.append(G.nodes[n]["size"] * 14)
                colors.append(G.nodes[n]["color"])
                labels.append(n)
            node_trace = go.Scatter(
                x=nx_, y=ny_,
                mode="markers+text",
                text=labels,
                textposition="middle center",
                marker=dict(size=sizes, color=colors, line_width=2),
                textfont=dict(color="white", size=12),
                hoverinfo="text",
            )

            fig = go.Figure(data=[edge_trace, node_trace])   # ← no “+” operator
            fig.update_layout(
                height=500,
                margin=dict(t=30, b=20, l=5, r=5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)


    # Neuro activity
    with col2:
        st.subheader("Neuro Activity")
        activity = get_json("/neuro_activity") or {}
        if activity.get("gradients"):
            grad_df = pd.DataFrame(
                {"Layer": list(activity["gradients"].keys()),
                 "Gradient": [abs(v) for v in activity["gradients"].values()]}
            )
            fig = px.bar(grad_df, x="Layer", y="Gradient")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Neuro activity data not available.")

# ───────────────────────────────────────────  Adaptive Modules

elif page == "modules":
    st.markdown("## Adaptive Neuro Modules")

    # helper: group names
    GROUPS = {
        "🛰 Data & Feature": [
            "MultiScaleFeatureEngine", "MarketThemeDetector", "LiquidityHeatmapLayer"
        ],
        "🛡️ Risk Management": [
            "PortfolioRiskSystem", "TimeAwareRiskScaling", "DynamicRiskController"
        ],
        "🔬 Strategy Analytics": [
            "StrategyIntrospector", "RegimePerformanceMatrix",
            "TradeThesisTracker", "PositionManager"
        ],
        "🎓 Learning & Bias": [
            "CurriculumPlannerPlus", "MemoryBudgetOptimizer",
            "BiasAuditor", "ThesisEvolutionEngine"
        ],
        "🤺 Opponent / Meta": [
            "OpponentModeEnhancer"
        ],
    }

    mods = {m["name"]: m for m in get_json("/modules") or []}

    for group, names in GROUPS.items():
        with st.expander(group, expanded=True):
            for n in names:
                if n not in mods:
                    continue
                m = mods[n]
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{n}**")
                toggle_key = f"{n}-toggle"
                new_state = c2.toggle(
                    "Enabled",
                    value=m["enabled"],
                    key=toggle_key,
                    label_visibility="collapsed",
                )
                if new_state != m["enabled"]:
                    put_json(f"/modules/{n}", {"enabled": new_state})
                    st.experimental_rerun()

    # Collective intelligence weight table
    with st.expander("🧠 Collective Intelligence", expanded=True):
        votes_hist = get_json("/votes_history") or []
        if votes_hist and isinstance(votes_hist[-1], dict):
            df_v = (
                pd.DataFrame(votes_hist[-1].items(), columns=["Module", "Weight"])
                .sort_values("Weight", ascending=False)
            )
            st.dataframe(df_v, use_container_width=True)
        else:
            st.info("No weight distribution yet.")

# ───────────────────────────────────────────  Quantum Ledger

elif page == "ledger":
    st.markdown("## Quantum Transaction Ledger")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Active Positions", get_active_positions())
        st.metric("Risk Exposure", f"{get_risk_pct():.1f}%")

    with col2:
        trades = get_json("/trades") or []
        if trades:
            df = pd.DataFrame(trades)
            st.data_editor(
                df,
                column_config={
                    "pnl": st.column_config.ProgressColumn(
                        "PnL", format="$%.2f", min_value=-10000, max_value=10000
                    )
                },
                use_container_width=True,
                height=600,
            )

# ───────────────────────────────────────────  Insight Engine

elif page == "insight":
    st.markdown("## Insight Engine")

    tab1, tab2, tab3 = st.tabs(["🧭 Reasoning Trace", "📜 Error Genome", "🔬 Market X‑Ray"])

    with tab1:
        trace = get_json("/reasoning_trace") or []
        st.code("\n".join(trace[-100:][::-1]), language="neuro")

    with tab2:
        genome = get_json("/genome_metrics") or {}
        if genome:
            df_g = pd.DataFrame({"metric": list(genome.keys()), "value": list(genome.values())})
            fig = px.line_polar(df_g, r="value", theta="metric", line_close=True)
            fig.update_traces(fill="toself")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        bars = get_json("/debug/bars") or {}
        if bars:
            inst = st.selectbox("Instrument", list(bars.keys()))
            if inst:
                df_b = pd.DataFrame(bars[inst]["H1"])
                df_b.set_index("time", inplace=True)
                st.area_chart(df_b["close"])

# ───────────────────────────────────────────  Market Reality

elif page == "reality":
    st.markdown("## Market Reality Matrix")

    with elements("reality_board"):
        layout = [
            dict(i="theme", x=0, y=0, w=6, h=4),
            dict(i="fractal", x=6, y=0, w=6, h=4),
            dict(i="liquidity", x=0, y=4, w=8, h=4),
            dict(i="world", x=8, y=4, w=4, h=4),
        ]
        with dashboard.Grid(layout):

            # Theme clusters (PCA 2‑D)
            with mui.Paper(key="theme", sx={"p": 2}):
                st.subheader("Market Theme Clusters")
                themes = get_json("/market_themes") or {}
                if themes:
                    X = pd.DataFrame(themes["cluster_centers"]).values
                    comps = PCA(n_components=2).fit_transform(X)
                    df_c = pd.DataFrame(comps, columns=["PC‑1", "PC‑2"])
                    df_c["cluster"] = [f"C{i}" for i in range(len(df_c))]
                    fig = px.scatter(df_c, x="PC‑1", y="PC‑2", text="cluster")
                    st.plotly_chart(fig, use_container_width=True)

            # Fractals
            with mui.Paper(key="fractal", sx={"p": 2}):
                st.subheader("Fractal Confirmation")
                fp = get_json("/fractal_patterns") or {}
                if fp:
                    fig = go.Figure(
                        go.Candlestick(
                            x=fp["time"],
                            open=fp["open"],
                            high=fp["high"],
                            low=fp["low"],
                            close=fp["close"],
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=fp["fractal_times"],
                            y=fp["fractal_prices"],
                            mode="markers",
                            marker=dict(size=8),
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Liquidity heat‑map
            with mui.Paper(key="liquidity", sx={"p": 2}):
                st.subheader("Liquidity Heat‑Map")
                liq = get_json("/liquidity_map") or {}
                if liq and liq.get("bids") and liq.get("asks"):
                    z   = [liq["bids"], liq["asks"]]
                    fig = create_annotated_heatmap(
                        z=z,
                        x=liq["price_levels"],
                        y=["bids", "asks"],
                        colorscale="Turbo",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No depth data available.")


            # World model forecast
            with mui.Paper(key="world", sx={"p": 2}):
                st.subheader("World Model Forecast")
                preds = get_json("/world_predictions") or {}
                if preds:
                    df_w = pd.DataFrame(
                        {"Actual": preds["actual"], "Predicted": preds["predicted"]}
                    )
                    st.line_chart(df_w)

# ───────────────────────────────────────────  Risk Matrix

elif page == "risk":
    st.markdown("## Portfolio Risk Ecosystem")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Exposure Limits")
        limits = get_json("/risk_limits") or {}
        if limits:
            df_lim = pd.DataFrame(limits.items(), columns=["Instrument", "Limit"])
            st.dataframe(df_lim.style.format({"Limit": "${:.2f}"}))

    with col2:
        st.subheader("Risk Distribution")
        rx = get_json("/risk_exposure") or {}
        if rx:
            df_rx = pd.DataFrame({"Instrument": rx["labels"], "Exposure": rx["values"]})
            fig = px.bar(df_rx, x="Instrument", y="Exposure", text_auto=".2%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No live exposure yet.")

# ───────────────────────────────────────────  Opponent Arena

elif page == "arena":
    st.markdown("## Adversarial Strategy Arena")
    tab1, tab2 = st.tabs(["Strategy Profiles", "Shadow Trading"])

    with tab1:
        st.subheader("Detected Opponent Signatures")
        opponents = get_json("/opponent_profiles") or {}
        if opponents:
            fig = px.line_polar(
                r=opponents["features"],
                theta=opponents["labels"],
                line_close=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Parallel Universe Performance")
        shadows = get_json("/shadow_performance") or []
        if shadows:
            df_sh = pd.DataFrame(shadows).set_index("step")
            st.area_chart(df_sh)

# ───────────────────────────────────────────  Auto‑refresh JS

html(
    """
    <script>
    setInterval(() => window.location.reload(), 15000);
    </script>
    """
)
