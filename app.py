"""
GEO Research Lab - Interactive Dashboard
"""
import json
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

RESULTS_DIR = Path("results")
st.set_page_config(page_title="GEO Research Lab", page_icon="🧬", layout="wide")

# Theme toggle
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def t(dark_val, light_val):
    return dark_val if st.session_state.theme == "dark" else light_val

BG = t("#0f1117", "#ffffff")
BG2 = t("#161b22", "#f6f8fa")
TEXT = t("#c9d1d9", "#24292f")
TEXT2 = t("#8b949e", "#57606a")
BORDER = t("rgba(255,255,255,0.06)", "rgba(0,0,0,0.08)")
ACCENT = t("#818cf8", "#6366f1")
ACCENT2 = t("#67e8f9", "#0891b2")
CARD_BG = t("rgba(255,255,255,0.03)", "rgba(0,0,0,0.02)")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp {{ background: {BG}; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {TEXT}; }}
.stSidebar {{ background: {BG2} !important; border-right: 1px solid {BORDER}; }}
h1 {{ color: {TEXT} !important; font-weight: 700 !important; font-size: 1.9rem !important; letter-spacing: -0.02em; }}
h2 {{ color: {TEXT} !important; font-weight: 600 !important; font-size: 1.25rem !important; }}
h3 {{ color: {ACCENT} !important; font-weight: 600 !important; font-size: 0.95rem !important; text-transform: uppercase; letter-spacing: 0.06em; }}
[data-testid="stMetric"] {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 20px; }}
[data-testid="stMetricValue"] {{ color: {TEXT} !important; font-size: 1.8rem !important; font-weight: 700 !important; }}
[data-testid="stMetricLabel"] {{ color: {TEXT2} !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }}
.card {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; padding: 20px; margin: 12px 0; line-height: 1.7; color: {TEXT2}; }}
.card strong {{ color: {TEXT}; }}
.insight {{ background: {CARD_BG}; border-left: 3px solid {ACCENT}; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 14px 0; color: {TEXT2}; line-height: 1.6; }}
.insight strong {{ color: {TEXT}; }}
.finding {{ background: {CARD_BG}; border-left: 3px solid #34d399; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 14px 0; color: {TEXT2}; line-height: 1.6; }}
.finding strong {{ color: {TEXT}; }}
.big-num {{ font-size: 2.8rem; font-weight: 800; color: {ACCENT}; line-height: 1; }}
.sub {{ color: {TEXT2}; font-size: 0.85rem; margin-top: -6px; }}
hr {{ border-color: {BORDER} !important; }}
.stSelectbox > div > div {{ background: {BG2} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important; }}
.stMultiSelect > div > div {{ background: {BG2} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; background: {BG2}; border-radius: 10px; padding: 3px; border: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 8px; color: {TEXT2}; font-weight: 500; font-size: 0.85rem; }}
.stTabs [aria-selected="true"] {{ background: {CARD_BG} !important; color: {ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)

PLT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=t("rgba(15,17,23,0.5)", "rgba(246,248,250,0.5)"),
    font=dict(family="Inter", color=TEXT2, size=11),
    title_font=dict(size=14, color=TEXT),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    margin=dict(t=50, b=40, l=50, r=20),
    colorway=[ACCENT, ACCENT2, "#f472b6", "#34d399", "#fbbf24", "#f87171", "#a78bfa"],
)
CC = {"A: Baseline": t("#475569","#94a3b8"), "B: RAG+Optimized": ACCENT, "C: RAG+Neutral": ACCENT2}

def fig_layout(fig, **kw):
    merged = {**PLT, **kw}
    fig.update_layout(**merged)
    return fig

# Data loading
@st.cache_data(ttl=30)
def load_latest(pattern):
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    return json.loads(files[0].read_text()) if files else []

@st.cache_data(ttl=30)
def load_all():
    bl = load_latest("baseline_*.json")
    ps = load_latest("pseudo_brand_*.json")
    fb = load_latest("feedback_loop_*.json")
    ro, rb = [], []
    for f in sorted(RESULTS_DIR.glob("rag_*.json"), key=lambda ff: ff.stat().st_mtime, reverse=True):
        d = json.loads(f.read_text())
        if not d or len(d) < 20: continue
        v = d[0].get("version") or d[0].get("rag_version") or ""
        if v == "optimized" and not ro: ro = d
        elif v == "baseline" and not rb: rb = d
        if ro and rb: break
    return bl, ro, rb, ps, fb

BRANDS = ["HubSpot","Salesforce","Copper","Less Annoying CRM","Freshsales","Jira","Asana","Notion","ClickUp"]

def mention_df(results, cond):
    rows = []
    for r in results:
        if r.get("status") != "success" or not r.get("response"): continue
        lo = r["response"].lower()
        for b in BRANDS:
            rows.append({"condition":cond, "brand":b, "model":r.get("model_key","?"), "specificity":r.get("specificity","?"), "mentioned":b.lower() in lo})
    return pd.DataFrame(rows)

baseline, rag_opt, rag_base, pseudo, feedback = load_all()
dfs = []
if baseline: dfs.append(mention_df(baseline, "A: Baseline"))
if rag_opt: dfs.append(mention_df(rag_opt, "B: RAG+Optimized"))
if rag_base: dfs.append(mention_df(rag_base, "C: RAG+Neutral"))
df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Sidebar
with st.sidebar:
    st.markdown(f"<div style='text-align:center;padding:16px 0 8px'><span style='font-size:1.5rem;font-weight:700;color:{TEXT}'>GEO Lab</span><br><span style='color:{TEXT2};font-size:0.72rem'>Generative Engine Optimization</span></div>", unsafe_allow_html=True)
    st.markdown("---")
    if st.toggle("Light mode", value=st.session_state.theme=="light"):
        if st.session_state.theme != "light":
            st.session_state.theme = "light"
            st.rerun()
    else:
        if st.session_state.theme != "dark":
            st.session_state.theme = "dark"
            st.rerun()
    page = st.selectbox("Navigate", ["Command Center","Condition Deep-Dive","Model Intelligence","NexaCRM Experiment","Feedback Loop","Embedding Analysis","Response Explorer"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"<div style='text-align:center;color:{TEXT2};font-size:0.7rem;line-height:1.8'><strong>CS6180 - Spring 2026</strong><br>Divit Pratap Singh<br>Uday Sonawane<br>Shantanu Shashank Dharmadhikari</div>", unsafe_allow_html=True)


if page == "Command Center":
    st.markdown("# Command Center")
    st.markdown(f"<p class='sub'>Overview of all GEO experiments across 9 brands, 4 LLMs, and 3 conditions</p>", unsafe_allow_html=True)

    ps_ok = [r for r in pseudo if r.get("status")=="success"]
    ps_hit = [r for r in ps_ok if r.get("nexacrm_position") is not None]
    total_q = sum(len([r for r in d if r.get("status")=="success"]) for d in [baseline, rag_opt, rag_base])
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Brands", "9", "CRM + PM")
    c2.metric("LLMs", "4", "via OpenRouter")
    c3.metric("Queries", str(total_q), "3 conditions")
    c4.metric("NexaCRM", f"{len(ps_hit)/max(len(ps_ok),1)*100:.0f}%", "pseudo-brand hit rate")
    c5.metric("Max Lift", "+77.5pp", "Copper CRM")

    st.markdown("---")
    st.markdown("""<div class='card'>
        <h2 style='margin-top:0'>What is this dashboard?</h2>
        This dashboard presents results from our research on <strong>Generative Engine Optimization (GEO)</strong>,
        the practice of optimizing content so brands appear prominently in AI-generated recommendations.
        <br><br>
        When you ask ChatGPT or Claude <em>"What is the best CRM?"</em>, the answer shapes real purchasing decisions.
        We study what determines which brands get recommended and whether injecting optimized content via
        <strong>RAG (Retrieval-Augmented Generation)</strong> can change those rankings.
        <br><br>
        <strong>Three experimental conditions:</strong><br>
        - <strong>Condition A (Baseline):</strong> Ask LLMs directly with no extra context. Pure parametric knowledge.<br>
        - <strong>Condition C (RAG + Neutral):</strong> Inject plain factual brand descriptions into the prompt via RAG.<br>
        - <strong>Condition B (RAG + Optimized):</strong> Inject descriptions enriched with statistics, expert quotes, and citations (per KDD 2024 GEO strategies).
    </div>""", unsafe_allow_html=True)

    if not df.empty:
        rates = df.groupby(["condition","brand"])["mentioned"].mean().reset_index()
        rates["pct"] = rates["mentioned"]*100
        fig = px.bar(rates, x="brand", y="pct", color="condition", barmode="group", color_discrete_map=CC,
                     labels={"pct":"Mention Rate (%)","brand":""},
                     text=rates["pct"].round(0).astype(int).astype(str)+"%")
        fig.update_traces(textposition="outside", textfont_size=9)
        fig_layout(fig, height=500, title="Brand Mention Rate Across All Conditions",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class='insight'>
            <strong>How to read this:</strong> Each group of bars shows how often an LLM mentions that brand in its recommendations.
            Taller bars = more visibility. The gap between grey (baseline) and the colored bars shows how much RAG helped.
        </div>""", unsafe_allow_html=True)

    # lift + radar
    left, right = st.columns(2)
    with left:
        st.markdown("## Visibility Lift: Baseline to Optimized")
        st.markdown("*How many percentage points did each brand gain from optimized RAG content?*")
        if not df.empty:
            a_r = df[df["condition"]=="A: Baseline"].groupby("brand")["mentioned"].mean()
            b_r = df[df["condition"]=="B: RAG+Optimized"].groupby("brand")["mentioned"].mean()
            common = a_r.index.intersection(b_r.index)
            lift = ((b_r[common]-a_r[common])*100).dropna().sort_values(ascending=True)
            if not lift.empty:
                colors = [ACCENT if v>0 else "#f87171" for v in lift.values]
                fig = go.Figure(go.Bar(x=lift.values, y=lift.index, orientation="h", marker_color=colors,
                                       text=[f"+{v:.0f}pp" for v in lift.values], textposition="outside", textfont=dict(color=TEXT,size=11)))
                fig_layout(fig, height=400, title="Percentage-Point Change (A to B)", xaxis_title="Change (pp)", margin=dict(l=140,t=50,b=40,r=20))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""<div class='finding'><strong>Key finding:</strong> Every brand improved. Low-visibility brands gained the most. Copper went from 15% to 92.5% (+77.5pp).</div>""", unsafe_allow_html=True)

    with right:
        st.markdown("## Model Consistency Radar")
        st.markdown("*Do all 4 LLMs respond equally to optimized content?*")
        if not df.empty:
            bd = df[df["condition"]=="B: RAG+Optimized"]
            if not bd.empty:
                mr = bd.groupby("model")["mentioned"].mean()*100
                cats, vals = list(mr.index), list(mr.values)
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
                                               fillcolor=t("rgba(129,140,248,0.1)","rgba(99,102,241,0.08)"),
                                               line=dict(color=ACCENT, width=2), marker=dict(size=7, color=ACCENT)))
                fig_layout(fig, height=400, title="Avg Mention Rate by Model (Condition B)",
                           polar=dict(bgcolor=t("rgba(15,17,23,0.4)","rgba(246,248,250,0.4)"),
                                      radialaxis=dict(gridcolor=BORDER, range=[0,105], tickfont=dict(size=9)),
                                      angularaxis=dict(gridcolor=BORDER)))
                st.plotly_chart(fig, use_container_width=True)

    # headline cards
    st.markdown("---")
    st.markdown("## Key Takeaways")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class='card' style='text-align:center'>
            <div class='big-num'>+77.5pp</div>
            <div style='font-weight:600;margin:6px 0;color:{TEXT}'>Copper CRM</div>
            <div style='font-size:0.82rem'>From 15% baseline to 92.5% with optimized RAG. Copper was nearly invisible to LLMs before optimization.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='card' style='text-align:center'>
            <div class='big-num'>95%</div>
            <div style='font-weight:600;margin:6px 0;color:{TEXT}'>NexaCRM (Fictional)</div>
            <div style='font-size:0.82rem'>A made-up brand with zero training data ranked #1 in 95% of queries, outranking HubSpot and Salesforce every time.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='card' style='text-align:center'>
            <div class='big-num'>3x</div>
            <div style='font-weight:600;margin:6px 0;color:{TEXT}'>More Consistent</div>
            <div style='font-size:0.82rem'>RAG makes LLM responses 3x more consistent across models (cosine distance drops from 0.37 to 0.13).</div>
        </div>""", unsafe_allow_html=True)



elif page == "Condition Deep-Dive":
    st.markdown("# Condition Deep-Dive")
    st.markdown(f"<p class='sub'>Detailed comparison of all three experimental conditions</p>", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
        <strong>What are the three conditions?</strong><br><br>
        <strong>Condition A (Baseline):</strong> We ask LLMs product recommendation questions with no extra context.
        The LLM relies entirely on what it learned during training. This reveals its "default" brand preferences.<br><br>
        <strong>Condition C (RAG + Neutral):</strong> Before sending the query, we retrieve plain brand descriptions from a vector database (ChromaDB)
        and prepend them to the prompt. This tests whether simply providing information helps.<br><br>
        <strong>Condition B (RAG + Optimized):</strong> Same as C, but the descriptions are rewritten using three proven strategies from the
        KDD 2024 GEO paper: adding statistics (+33% visibility), expert quotations (+41%), and source citations (+31%).
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Brand Comparison", "Query Specificity", "Heatmap"])

    with tab1:
        sel = st.multiselect("Filter brands", BRANDS, default=BRANDS)
        if not df.empty:
            f = df[df["brand"].isin(sel)]
            rates = f.groupby(["condition","brand"])["mentioned"].mean().reset_index()
            rates["pct"] = rates["mentioned"]*100
            fig = px.bar(rates, x="brand", y="pct", color="condition", barmode="group", color_discrete_map=CC,
                         text=rates["pct"].round(0).astype(int).astype(str)+"%")
            fig.update_traces(textposition="outside", textfont_size=9)
            fig_layout(fig, height=500, title="Mention Rate by Brand and Condition")
            st.plotly_chart(fig, use_container_width=True)

            pivot = rates.pivot(index="brand", columns="condition", values="pct").reset_index()
            if "A: Baseline" in pivot.columns and "B: RAG+Optimized" in pivot.columns:
                pivot["Lift (pp)"] = (pivot["B: RAG+Optimized"] - pivot["A: Baseline"]).round(1)
                pivot = pivot.sort_values("Lift (pp)", ascending=False)
                st.dataframe(pivot.style.background_gradient(subset=["Lift (pp)"], cmap="RdYlGn", vmin=-10, vmax=80)
                             .format("{:.1f}", subset=[c for c in pivot.columns if c != "brand"]), use_container_width=True)

    with tab2:
        st.markdown("""<div class='insight'>
            <strong>What is query specificity?</strong> We test three types of queries:<br>
            - <strong>Vague:</strong> "What is the best CRM?" (favors dominant brands)<br>
            - <strong>Medium:</strong> "Best CRM for startups" (slightly more targeted)<br>
            - <strong>Specific:</strong> "Best CRM for a 50-person B2B startup under $20K/year with Slack integration" (opens opportunities for niche brands)
        </div>""", unsafe_allow_html=True)
        if not df.empty:
            sp = df.groupby(["condition","specificity","brand"])["mentioned"].mean().reset_index()
            sp["pct"] = sp["mentioned"]*100
            fig = px.line(sp, x="specificity", y="pct", color="brand", facet_col="condition", markers=True,
                          category_orders={"specificity":["vague","medium","specific"],"condition":["A: Baseline","C: RAG+Neutral","B: RAG+Optimized"]})
            fig_layout(fig, height=450, title="Brand Visibility by Query Specificity")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("""<div class='insight'><strong>How to read this:</strong> Each cell shows how often a specific model mentions a specific brand (as a percentage). Bright cells mean high visibility, dark cells mean the brand is rarely recommended by that model.</div>""", unsafe_allow_html=True)
        cond_sel = st.selectbox("Select condition", list(CC.keys()))
        if not df.empty:
            cd = df[df["condition"]==cond_sel]
            hm = cd.groupby(["model","brand"])["mentioned"].mean().reset_index()
            hm["pct"] = (hm["mentioned"]*100).round(0)
            piv = hm.pivot(index="model", columns="brand", values="pct").fillna(0)
            fig = px.imshow(piv, text_auto=".0f", color_continuous_scale=t(["#0f1117","#1e1b4b","#818cf8","#c7d2fe"],["#ffffff","#e0e7ff","#818cf8","#4f46e5"]),
                            labels=dict(color="Mention %"), aspect="auto")
            fig_layout(fig, height=350, title=f"Model x Brand Heatmap: {cond_sel}")
            st.plotly_chart(fig, use_container_width=True)



elif page == "Model Intelligence":
    st.markdown("# Model Intelligence")
    st.markdown(f"<p class='sub'>How each LLM responds differently to the same content</p>", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
        <strong>Which LLMs did we test?</strong><br><br>
        We used 4 models via OpenRouter API:<br>
        - <strong>GPT-4o-mini</strong> (OpenAI) - fast, affordable, widely used<br>
        - <strong>Llama 4 Maverick</strong> (Meta) - open-weight, free tier available<br>
        - <strong>Mistral Large</strong> (Mistral AI) - European model, strong reasoning<br>
        - <strong>DeepSeek V3.2</strong> (DeepSeek) - Chinese model, very cost-effective<br><br>
        Each model has different training data and alignment, so the same brand can be invisible on one model and dominant on another.
    </div>""", unsafe_allow_html=True)

    if not df.empty:
        models = sorted(df["model"].unique())
        cols = st.columns(len(models))
        for i, m in enumerate(models):
            bd = df[(df["model"]==m) & (df["condition"]=="B: RAG+Optimized")]
            avg = bd["mentioned"].mean()*100 if not bd.empty else 0
            ad = df[(df["model"]==m) & (df["condition"]=="A: Baseline")]
            a_avg = ad["mentioned"].mean()*100 if not ad.empty else 0
            cols[i].metric(m, f"{avg:.0f}%", f"+{avg-a_avg:.0f}pp vs baseline")

        st.markdown("---")
        mc = df.groupby(["model","condition"])["mentioned"].mean().reset_index()
        mc["pct"] = mc["mentioned"]*100
        fig = px.bar(mc, x="model", y="pct", color="condition", barmode="group", color_discrete_map=CC,
                     text=mc["pct"].round(0).astype(int).astype(str)+"%")
        fig.update_traces(textposition="outside", textfont_size=10)
        fig_layout(fig, height=450, title="Average Mention Rate per Model Across Conditions")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class='finding'><strong>Finding:</strong> All 4 models show the same pattern: Baseline < Neutral RAG < Optimized RAG. GEO strategies work across model families, not just one vendor.</div>""", unsafe_allow_html=True)



elif page == "NexaCRM Experiment":
    st.markdown("# NexaCRM: The Pseudo-Brand Experiment")
    st.markdown(f"<p class='sub'>Can a brand that does not exist rank #1 in AI recommendations?</p>", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
        <strong>What is NexaCRM?</strong><br><br>
        NexaCRM is a CRM product we completely made up. It has no website, no users, no reviews, and zero mentions on the internet.
        No LLM has ever encountered it during training.<br><br>
        <strong>What did we do?</strong><br>
        We wrote an optimized description using the same KDD GEO strategies (fake but credible statistics,
        fabricated publication quotes, invented expert endorsements) and injected it via RAG alongside real CRM brands like HubSpot and Salesforce.<br><br>
        <strong>Why?</strong><br>
        To isolate the effect of content quality from parametric knowledge. If NexaCRM ranks highly, it proves that what is in the RAG context
        matters more than what the model "knows" from training.
    </div>""", unsafe_allow_html=True)

    if pseudo:
        ok = [r for r in pseudo if r.get("status")=="success"]
        hit = [r for r in ok if r.get("nexacrm_position") is not None]
        avg_p = sum(r["nexacrm_position"] for r in hit)/len(hit) if hit else 0

        st.markdown("---")
        c1,c2,c3 = st.columns(3)
        c1.metric("Mention Rate", f"{len(hit)/len(ok)*100:.0f}%", f"{len(hit)} of {len(ok)} queries")
        c2.metric("Avg Position", f"#{avg_p:.1f}", "when mentioned")
        c3.metric("Beat Real Brands?", "Yes", "#1 every time")

        st.markdown(f"""<div class='finding'>
            <strong>Result:</strong> NexaCRM was mentioned in {len(hit)/len(ok)*100:.0f}% of queries and ranked #1 every single time,
            beating HubSpot, Salesforce, and all real competitors. A brand with zero training data outranked
            products used by millions of people, purely through optimized RAG content.
        </div>""", unsafe_allow_html=True)

        # per-model
        st.markdown("## Results by Model")
        md = {}
        for r in ok:
            m = r.get("model_key","?")
            if m not in md: md[m] = {"total":0,"hit":0}
            md[m]["total"] += 1
            if r.get("nexacrm_position") is not None: md[m]["hit"] += 1
        mdf = pd.DataFrame([{"Model":m,"Queries":d["total"],"Mentioned":d["hit"],"Rate":d["hit"]/d["total"]*100} for m,d in md.items()])
        fig = go.Figure(go.Bar(x=mdf["Model"], y=mdf["Rate"], marker_color=[ACCENT if r==100 else ACCENT2 for r in mdf["Rate"]],
                                text=mdf["Rate"].apply(lambda x: f"{x:.0f}%"), textposition="outside"))
        fig_layout(fig, height=380, title="NexaCRM Mention Rate by Model", yaxis_range=[0,115], yaxis_title="Mention Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Individual Query Results")
        st.markdown("*Expand any row to see the full LLM response and which brands were recommended in what order.*")
        for r in ok:
            brands = list(r.get("brand_positions",{}).keys()) if r.get("brand_positions") else []
            pos = r.get("nexacrm_position")
            icon = "🏆" if pos==1 else ("✅" if pos else "—")
            rank = f"#{pos}" if pos else "Not mentioned"
            chain = " > ".join(brands[:6])
            with st.expander(f"{icon}  {r.get('model_key')}  |  {r.get('prompt','')[:50]}  |  NexaCRM: {rank}"):
                st.markdown(f"**Ranking order:** {chain}")
                st.markdown(r.get("response","")[:800])



elif page == "Feedback Loop":
    st.markdown("# Feedback Loop: LLM-as-Optimizer")
    st.markdown(f"<p class='sub'>Using one LLM to iteratively rewrite content that improves visibility on other LLMs</p>", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
        <strong>How does the feedback loop work?</strong><br><br>
        1. Measure a brand's current visibility (mention rate + position)<br>
        2. Show the current description and results to GPT-4o-mini as an "optimizer"<br>
        3. Ask it to rewrite the description to maximize recommendation likelihood<br>
        4. Re-embed the new description into ChromaDB<br>
        5. Re-test across all 4 models<br>
        6. Repeat<br><br>
        This creates a closed-loop system where content improves automatically over iterations.
    </div>""", unsafe_allow_html=True)

    if feedback:
        iters = [h["iteration"] for h in feedback]
        rates = [h["mention_rate"]*100 for h in feedback]
        positions = [h.get("avg_position") for h in feedback]

        c1,c2,c3 = st.columns(3)
        c1.metric("Iterations", len(feedback)-1)
        c2.metric("Mention Rate", f"{rates[0]:.0f}% to {rates[-1]:.0f}%")
        p0, pn = positions[0], positions[-1]
        c3.metric("Position", f"#{pn:.1f}" if pn else "N/A", f"{pn-p0:+.1f}" if p0 and pn else "")

        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=iters, y=rates, mode="lines+markers+text", name="Mention Rate (%)",
                                  line=dict(color=ACCENT,width=3), marker=dict(size=12),
                                  text=[f"{r:.0f}%" for r in rates], textposition="top center"), secondary_y=False)
        if all(p for p in positions):
            fig.add_trace(go.Scatter(x=iters, y=positions, mode="lines+markers", name="Avg Position",
                                      line=dict(color=ACCENT2,width=3,dash="dot"), marker=dict(size=12)), secondary_y=True)
        fig_layout(fig, height=420, title="Copper: Optimization Across Iterations")
        fig.update_yaxes(title_text="Mention Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Avg Position (lower = better)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## Content Evolution")
        st.markdown("*See how the LLM rewrote the description at each iteration.*")
        for h in feedback:
            label = "Original (hand-crafted)" if h["iteration"]==0 else f"LLM Rewrite v{h['iteration']}"
            with st.expander(f"Iteration {h['iteration']}: {label} - {h['mention_rate']*100:.1f}% mention rate"):
                st.code(h.get("content","N/A")[:700], language=None)
    else:
        st.info("No feedback loop data yet. Run: python run_feedback_loop.py --brand Copper --iterations 2")



elif page == "Embedding Analysis":
    st.markdown("# Embedding Space Analysis")
    st.markdown(f"<p class='sub'>How RAG changes the semantic distribution of LLM responses</p>", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
        <strong>What is this analysis?</strong><br><br>
        We took every LLM response from all conditions, embedded them as vectors using sentence-transformers (all-MiniLM-L6-v2),
        and projected them into 2D using t-SNE. This shows how "similar" or "different" the responses are.<br><br>
        <strong>Why does it matter?</strong><br>
        If RAG changes what LLMs say, the responses from different conditions should form distinct clusters.
        If models agree more under RAG, the cluster should be tighter (lower internal distance).
    </div>""", unsafe_allow_html=True)

    f5 = Path("paper/figures/fig5_embedding_tsne.png")
    f6 = Path("paper/figures/fig6_embedding_by_model.png")
    c1,c2 = st.columns(2)
    if f5.exists(): c1.image(str(f5), caption="Colored by experimental condition", use_container_width=True)
    if f6.exists(): c2.image(str(f6), caption="Colored by LLM model", use_container_width=True)

    st.markdown("## Distance Matrix")
    st.markdown("*Cosine distance between response embeddings. Higher = more different.*")
    st.dataframe(pd.DataFrame([
        {"Comparison":"Baseline vs Neutral RAG","Distance":0.3683,"What it means":"RAG moderately shifts response content"},
        {"Comparison":"Baseline vs Optimized RAG","Distance":0.3496,"What it means":"Similar shift magnitude as neutral"},
        {"Comparison":"Neutral vs Optimized RAG","Distance":0.1430,"What it means":"Very similar since both use RAG mechanism"},
        {"Comparison":"Baseline vs NexaCRM","Distance":0.4476,"What it means":"Largest shift: injecting a new brand rewrites responses"},
    ]), use_container_width=True, hide_index=True)

    st.markdown("## Model Agreement (Response Coherence)")
    st.markdown("*Lower internal distance = models agree more with each other.*")
    st.dataframe(pd.DataFrame([
        {"Condition":"A: Baseline","Internal Distance":0.3712,"Meaning":"Models disagree substantially on what to recommend"},
        {"Condition":"C: RAG + Neutral","Internal Distance":0.1660,"Meaning":"2x more consistent with RAG"},
        {"Condition":"B: RAG + Optimized","Internal Distance":0.1286,"Meaning":"3x more consistent. Optimized content aligns all models."},
    ]), use_container_width=True, hide_index=True)

    st.markdown("""<div class='finding'>
        <strong>Key finding:</strong> Without RAG, each model recommends different brands (high disagreement).
        With optimized RAG, all 4 models converge on the same recommendations. RAG does not just boost brands,
        it standardizes the recommendation landscape.
    </div>""", unsafe_allow_html=True)



elif page == "Response Explorer":
    st.markdown("# Response Explorer")
    st.markdown(f"<p class='sub'>Browse raw LLM outputs from every experiment</p>", unsafe_allow_html=True)

    st.markdown("""<div class='insight'><strong>How to use:</strong> Select a dataset, optionally filter by brand mention or model, then expand any row to read the full LLM response.</div>""", unsafe_allow_html=True)

    src = st.selectbox("Dataset", ["Baseline (A)","RAG + Optimized (B)","RAG + Neutral (C)","Pseudo-brand (NexaCRM)"])
    data = {"Baseline (A)":baseline,"RAG + Optimized (B)":rag_opt,"RAG + Neutral (C)":rag_base,"Pseudo-brand (NexaCRM)":pseudo}[src]
    ok = [r for r in data if r.get("status")=="success"]
    st.write(f"**{len(ok)}** successful responses")

    brand_f = st.text_input("Filter by brand mention", "")
    model_f = st.multiselect("Filter by model", sorted(set(r.get("model_key","?") for r in ok)))

    shown = 0
    for r in ok:
        if brand_f and brand_f.lower() not in r.get("response","").lower(): continue
        if model_f and r.get("model_key") not in model_f: continue
        if shown >= 30: break
        shown += 1
        with st.expander(f"{r.get('model_key','?')}  |  {r.get('prompt','')[:55]}"):
            st.markdown(r.get("response","N/A"))
