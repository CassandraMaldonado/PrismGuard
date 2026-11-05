"""Interactive Streamlit decision brief for PrismGuard."""

# I use Streamlit to make the evaluation easier to explore. The analysis stays
# in the package, while this file helps someone understand the tradeoffs behind
# the model rollout decision.

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prism_guard.analysis import analyze

st.set_page_config(page_title="PrismGuard", page_icon="◈", layout="wide")
st.markdown("""<style>@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');.stApp{background:#f3f1ec;color:#111817;font-family:Manrope,sans-serif}.block-container{max-width:1220px;padding-top:2.5rem}.eyebrow{font:9px 'DM Mono';letter-spacing:.12em;color:#66716b}.hero h1{font-size:5.1rem;line-height:.94;letter-spacing:-.07em;margin:.2rem 0 1.3rem}.teal{color:#008f79}.intro{max-width:680px;color:#68716e;line-height:1.7}.memo{background:#111817;color:white;padding:1.8rem;min-height:270px;box-shadow:12px 12px 0 #d8ff59}.memo small{font:9px 'DM Mono';color:#93a09a}.memo strong{display:block;color:#d8ff59;font-size:2.5rem;margin:2.5rem 0 .6rem;letter-spacing:-.05em}.memo p{font-size:.82rem;line-height:1.55;color:#c1cbc7}.rule{border-top:1px solid #d6d4cc;margin:2.5rem 0 1.8rem}div[data-testid='stMetric']{background:#fbfaf5;border:1px solid #d6d4cc;padding:1.1rem}.guard{display:grid;grid-template-columns:25px 1fr auto;gap:10px;align-items:center;border-top:1px solid #d6d4cc;padding:13px 0}.guard i{display:grid;place-items:center;width:20px;height:20px;border-radius:50%;background:#dff4b8;color:#27663a;font-style:normal}.guard.fail i{background:#fed7cf;color:#a44030}.guard b{font-size:.8rem}.guard span{font:9px 'DM Mono';color:#68716e}.stButton>button{width:100%;background:#008f79;color:white;border:0;border-radius:0;font-weight:700}.stButton>button:hover{background:#111817;color:white}</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def run(seed, samples): return analyze(seed, samples)

with st.sidebar:
    st.markdown("## ◈ PrismGuard")
    st.caption("AI RELIABILITY LAB")
    st.divider()
    seed=st.number_input("Replication seed",0,1_000_000,41)
    samples=st.slider("Evaluation samples",400,10_000,3_000,100)
    rerun=st.button("Rerun evaluation ↗",type="primary")
    st.divider();st.caption("I compare a candidate retrieval and response policy with the existing policy across quality, safety, calibration, latency, cost, and traffic drift.")
if rerun: run.clear()
with st.spinner("Evaluating candidate policy…"): r=run(int(seed),int(samples))
left,right=st.columns([1.5,.72],gap="large")
with left:
    st.markdown('<div class="hero"><p class="eyebrow">AI PRODUCT EVALUATION / POLICY 041</p><h1>Better answers.<br><span class="teal">Evidence first.</span></h1><p class="intro">A rollout decision should reflect more than a quality score. This lab evaluates whether a candidate AI policy is better, safe enough, calibrated, and worth its latency and cost tradeoffs.</p></div>',unsafe_allow_html=True)
with right: st.markdown(f'<div class="memo"><small>ROLLOUT RECOMMENDATION</small><strong>{r["decision"]["status"]}</strong><p>{r["decision"]["summary"]}</p><p>{r["decision"]["next_action"]}</p></div>',unsafe_allow_html=True)
st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
a,b,c,d=st.columns(4);a.metric("HUMAN QUALITY",f'{r["metrics"]["quality"]["relative_effect"]:+.1%}',"relative lift");b.metric("GROUNDEDNESS",f'{r["metrics"]["grounded"]["effect"]:+.1%}',"absolute lift");c.metric("LATENCY",f'{r["metrics"]["latency_ms"]["effect"]:+.0f} ms',"candidate vs control");d.metric("CALIBRATION",f'{r["calibration"]["ece"]:.3f}',"expected calibration error")
st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
col1,col2=st.columns([1.05,.95],gap="large")
with col1:
    st.markdown('<p class="eyebrow">01 / CALIBRATION</p>',unsafe_allow_html=True);st.subheader("Do confidence scores match human-rated quality?")
bins=pd.DataFrame(r["calibration"]["bins"]);fig=go.Figure();fig.add_trace(go.Scatter(x=bins.confidence,y=bins.accuracy,mode="lines+markers",name="Observed accuracy",line={"color":"#008f79","width":3}));fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect calibration",line={"color":"#9aa4a0","dash":"dash"}));fig.update_layout(height=330,margin={"l":10,"r":10,"t":20,"b":10},paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_title="Model confidence",yaxis_title="Human-rated quality",legend={"orientation":"h","y":1.12});fig.update_xaxes(range=[0,1]);fig.update_yaxes(range=[0,1],gridcolor="#dedcd4");st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
with col2:
    st.markdown('<p class="eyebrow">02 / RELEASE GUARDRAILS</p>',unsafe_allow_html=True);st.subheader("What needs to be true before shipping?")
    for g in r["guardrails"]:st.markdown(f'<div class="guard {"" if g["passed"] else "fail"}"><i>{"✓" if g["passed"] else "!"}</i><b>{g["name"]}</b><span>{g["threshold"]}</span></div>',unsafe_allow_html=True)
st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
st.markdown('<p class="eyebrow">03 / PERFORMANCE SLICES</p>',unsafe_allow_html=True);st.subheader("Where does the candidate help most?")
rows=[{"Task":k.title(),"Quality lift":v["quality"]["effect"],"Groundedness lift":v["grounded"]["effect"],"Quality p-value":v["quality"]["p_value"]} for k,v in r["task_slices"].items()];st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,column_config={"Quality lift":st.column_config.NumberColumn(format="%.3f"),"Groundedness lift":st.column_config.NumberColumn(format="%.3f"),"Quality p-value":st.column_config.NumberColumn(format="%.3f")})
st.markdown('<div class="rule"></div>',unsafe_allow_html=True)
st.markdown('<p class="eyebrow">04 / DRIFT CHECK</p>',unsafe_allow_html=True);x,y=st.columns(2);x.metric("POPULATION STABILITY INDEX",f'{r["drift"]["psi"]:.3f}',r["drift"]["status"]);y.markdown("#### Why this matters\nEven a strong offline evaluation can be misleading if live traffic differs from the data used to evaluate it. I use this check to decide whether the rollout needs additional monitoring.")

