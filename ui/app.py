# ui/app.py
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.figure_factory import create_annotated_heatmap
from sklearn.decomposition import PCA
from streamlit_elements import elements, dashboard, mui  # ensure `pip install streamlit-elements`
from streamlit.components.v1 import html
from utils import get_json, put_json

# ─────────────── App Config ───────────────
st.set_page_config(page_title="NeuroTrader Pro", layout="wide")
API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")

# ─────────────── Global Style ───────────────
st.markdown("""
<style>
body, .css-18e3th9 {
    background-color: #0d0d0d;
    color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background-color: #111111;
    color: #FFFFFF;
}
[data-testid="stMetricValue"] { font-size:1.5rem !important; }
[data-testid="stMetricLabel"] { opacity:.75; font-size:.9rem !important; }
.stButton>button {
    background-color: transparent;
    color: #FFFFFF;
    border: none;
    text-align: left;
    width: 100%;
    padding: 0.6rem 1rem;
    font-weight: 500;
}
.stButton>button:hover {
    background-color: rgba(255,255,255,0.1);
}
.mui-Paper {
    background-color: #1a1a1a !important;
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────── Sidebar ───────────────
st.sidebar.title("NeuroTrader Pro")
pages = ["Dashboard", "Market Hologram", "Modules", "Ledger", "Insights", "Reality", "Risk", "Arena"]
page = st.sidebar.radio("Navigation", pages, index=0)
st.sidebar.markdown("---")

# Mode info
mode_stats = get_json("/mode") or {}
mode = mode_stats.get("mode", "UNKNOWN").upper()
auto = mode_stats.get("auto", False)
reason = mode_stats.get("reason", "")
# Colors for modes
mode_colors = {"SAFE":"#4CAF50", "NORMAL":"#FFFFFF", "AGGRESSIVE":"#FF9800", "EXTREME":"#F44336"}
mode_color = mode_colors.get(mode, "#888888")
st.sidebar.markdown(f"**Mode:** <span style='color:{mode_color}'>{mode}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Auto:** {'✅' if auto else '❌'}")
if auto and reason:
    st.sidebar.markdown(f"<small>{reason}</small>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# ─────────────── Top Metrics ───────────────
def render_topbar():
    status = get_json("/status") or {}
    cols = st.columns([1.5,1,1,1,1,1])
    metrics = [
        ("Balance", f"${status.get('balance',0):,.2f}"),
        ("Drawdown", f"{status.get('drawdown',0)*100:.2f}%"),
        ("Clusters", status.get('clusters',0)),
        ("Live PnL", f"${status.get('pnl',0):,.2f}"),
        ("Open Positions", status.get('active_positions',0)),
        ("Risk", f"{status.get('risk_exposure',0):.2f}%")
    ]
    for col,(label,val) in zip(cols, metrics):
        col.metric(label, val)

# ─────────────── Page Definitions ───────────────
if page == "Dashboard":
    st.header("Trading Dashboard")
    render_topbar()
    col1,col2 = st.columns([3,2], gap="large")
    with col1:
        st.subheader("Market Consciousness")
        data = get_json("/quantum_map") or {}
        if data.get('nodes'):
            G=nx.DiGraph()
            for n in data['nodes']:
                G.add_node(n['id'], size=n['size']*10, color=n['color'])
            for e in data['links']:
                G.add_edge(e['source'], e['target'], weight=e['value']*2)
            pos=nx.spring_layout(G, seed=42)
            edge_x,edge_y=[],[]
            for u,v in G.edges():
                x0,y0=pos[u]; x1,y1=pos[v]
                edge_x+= [x0,x1,None]; edge_y+=[y0,y1,None]
            edge_trace=go.Scatter(x=edge_x,y=edge_y, mode='lines', line=dict(width=1, color='rgba(255,255,255,0.2)'), hoverinfo='none')
            node_x,node_y,node_size,node_color,node_text=[],[],[],[],[]
            for node in G.nodes():
                x,y=pos[node]; node_x.append(x); node_y.append(y)
                node_size.append(G.nodes[node]['size']); node_color.append(G.nodes[node]['color']); node_text.append(node)
            node_trace=go.Scatter(x=node_x,y=node_y, mode='markers+text', text=node_text, textposition='top center', marker=dict(size=node_size, color=node_color, line_width=1), hoverinfo='text')
            fig=go.Figure(data=[edge_trace,node_trace])
            fig.update_layout(height=500,margin=dict(t=20,b=20,l=20,r=20),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("No quantum map data.")
    with col2:
        st.subheader("Neuro Activity")
        act = get_json("/neuro_activity") or {}
        if act.get('gradients'):
            df=pd.DataFrame({'Layer':list(act['gradients'].keys()), 'Gradient':[abs(v) for v in act['gradients'].values()]})
            fig=px.bar(df,x='Layer',y='Gradient')
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("No neuro activity data.")

elif page == "Market Hologram":
    st.header("Market Hologram")
    render_topbar()
    data = get_json("/quantum_map") or {}
    if data.get('nodes'):
        # Build the network graph similar to Dashboard, but color by volatility
        G = nx.DiGraph()
        vols = {n['id']: n['size'] for n in data['nodes']}
        for n in data['nodes']:
            G.add_node(n['id'], size=n['size']*10, color=n['color'])
        for e in data['links']:
            G.add_edge(e['source'], e['target'], weight=e['value']*2)
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='rgba(255,255,255,0.2)'), hoverinfo='none')
        node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            node_size.append(G.nodes[node]['size'])
            # scale volatility to color intensity (lighter for higher vol)
            v = vols.get(node, 1)
            intensity = min(255, int(50 + v * 20))
            node_color.append(f"rgb({intensity},{intensity},{intensity})")
            node_text.append(f"{node} ({v:.2f})")
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text, textposition='top center',
            marker=dict(size=node_size, color=node_color, line_width=1), hoverinfo='text'
        )
        fig_holo = go.Figure(data=[edge_trace, node_trace])
        fig_holo.update_layout(height=520, margin=dict(t=20,b=20,l=20,r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_holo, use_container_width=True)
    else:
        st.info("No quantum map data.")

elif page == "Modules":
    st.header("Adaptive Modules")
    mods = get_json("/modules/status") or []
    if mods:
        df=pd.DataFrame(mods)
        df['Status']=df['enabled'].apply(lambda e: '🟢 Online' if e else '🔴 Offline')
        st.table(df[['name','Status','confidence']].rename(columns={'name':'Module','confidence':'Confidence'}))
        for m in mods:
            new=st.checkbox(f"Enable {m['name']}",value=m['enabled'])
            if new!=m['enabled']:
                put_json(f"/modules/{m['name']}",{"enabled":new})
                st.experimental_rerun()
    else:
        st.error("Modules data unavailable.")

elif page == "Ledger":
    st.header("Quantum Ledger")
    render_topbar()
    col1,col2=st.columns([1,2])
    with col1:
        st.metric("Active Positions", get_json("/trades") and len([t for t in get_json("/trades") if not t.get('exit_reason')]) or 0)
        st.metric("Risk Exposed", f"{sum(get_json('/risk_exposure')['values'])*100:.2f}%")
    with col2:
        trades=get_json("/trades") or []
        if trades:
            df=pd.DataFrame(trades)
            st.data_editor(df,use_container_width=True,height=500)
        else:
            st.info("No trades data.")

elif page == "Insights":
    st.header("Insight Engine")
    tabs=st.tabs(["Reasoning Trace","Error Genome","Market X‑Ray"])
    with tabs[0]:
        trace=get_json("/reasoning_trace") or []
        st.code("\n".join(trace[::-1][:100]))
    with tabs[1]:
        genome=get_json("/genome_metrics") or {}
        if genome:
            df=pd.DataFrame({'metric':list(genome.keys()),'value':list(genome.values())})
            fig=px.line_polar(df,r='value',theta='metric',line_close=True)
            fig.update_traces(fill='toself')
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("No genome data.")
    with tabs[2]:
        bars=get_json("/debug/bars") or {}
        if bars:
            inst=st.selectbox("Instrument",list(bars.keys()))
            if inst:
                rec=bars[inst]['H1']
                df=pd.DataFrame({'time':rec['time'],'close':rec['close']}).set_index('time')
                st.area_chart(df,use_container_width=True)
        else:
            st.info("No market x-ray data.")

elif page == "Reality":
    st.header("Market Reality Matrix")
    with elements("board"):
        layout=[{'i':'theme','x':0,'y':0,'w':6,'h':4},{'i':'fractal','x':6,'y':0,'w':6,'h':4},{'i':'liquidity','x':0,'y':4,'w':8,'h':4},{'i':'world','x':8,'y':4,'w':4,'h':4}]
        with dashboard.Grid(layout):
            with mui.Paper(key='theme',sx={'p':2}):
                st.subheader("Theme Clusters")
                themes=get_json("/market_themes") or {}
                if themes:
                    X=pd.DataFrame(themes['cluster_centers']).values
                    comps=PCA(n_components=2).fit_transform(X)
                    dfc=pd.DataFrame(comps,columns=['PC1','PC2'])
                    dfc['cluster']=[f"C{i}" for i in range(len(dfc))]
                    fig=px.scatter(dfc,x='PC1',y='PC2',text='cluster')
                    st.plotly_chart(fig,use_container_width=True)
                else:
                    st.info("No themes.")
            with mui.Paper(key='fractal',sx={'p':2}):
                st.subheader("Fractal Patterns")
                fp=get_json("/fractal_patterns") or {}
                if fp:
                    fig=go.Figure(data=[go.Candlestick(x=fp['time'],open=fp['open'],high=fp['high'],low=fp['low'],close=fp['close'])])
                    fig.add_trace(go.Scatter(x=fp['fractal_times'],y=fp['fractal_prices'],mode='markers'))
                    st.plotly_chart(fig,use_container_width=True)
                else:
                    st.info("No fractal data.")
            with mui.Paper(key='liquidity',sx={'p':2}):
                st.subheader("Liquidity Heatmap")
                liq=get_json("/liquidity_map") or {}
                if liq.get('bids') and liq.get('asks'):
                    z=[liq['bids'],liq['asks']]
                    fig=create_annotated_heatmap(z,x=liq['price_levels'],y=['bids','asks'],colorscale='Turbo')
                    st.plotly_chart(fig,use_container_width=True)
                else:
                    st.info("No depth data.")
            with mui.Paper(key='world',sx={'p':2}):
                st.subheader("World Forecast")
                preds=get_json("/world_predictions") or {}
                if preds:
                    dfw=pd.DataFrame({'Actual':preds['actual'],'Predicted':preds['predicted']})
                    st.line_chart(dfw,use_container_width=True)
                else:
                    st.info("No world model data.")

elif page == "Risk":
    st.header("Portfolio Risk Ecosystem")
    col1,col2=st.columns([1,2])
    with col1:
        st.subheader("Exposure Limits")
        limits=get_json("/risk_limits") or {}
        if limits:
            df=pd.DataFrame(limits.items(),columns=['Instrument','Limit'])
            st.dataframe(df.style.format({'Limit':'${:.2f}'}),use_container_width=True)
        else:
            st.info("No limits data.")
    with col2:
        st.subheader("Risk Distribution")
        rx=get_json("/risk_exposure") or {}
        if rx.get('labels'):
            dfr=pd.DataFrame({'Instrument':rx['labels'],'Exposure':rx['values']})
            fig=px.bar(dfr,x='Instrument',y='Exposure',text_auto='.2%')
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("No exposure data.")

elif page == "Arena":
    st.header("Adversarial Arena")
    tab1,tab2=st.tabs(["Strategy Profiles","Shadow Trading"])
    with tab1:
        st.subheader("Opponent Signatures")
        opp=get_json("/opponent_profiles") or {}
        if opp:
            dfop=pd.DataFrame({'feature':opp['labels'],'value':opp['features']})
            fig=px.line_polar(dfop,r='value',theta='feature',line_close=True)
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("No opponent data.")
    with tab2:
        st.subheader("Parallel Performance")
        shadow=get_json("/shadow_performance") or []
        if shadow:
            dfsh=pd.DataFrame(shadow).set_index('step')
            st.area_chart(dfsh,use_container_width=True)
        else:
            st.info("No shadow data.")

# ─────────────── Auto-refresh ───────────────
html("""
<script>setInterval(()=>window.location.reload(),15000);</script>
""")
