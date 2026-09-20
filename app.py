import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart EV Charging GNN", page_icon="⚡", layout="wide")
I=pd.read_csv("interactions.csv"); S=pd.read_csv("stations.csv")
st.title("⚡ Smart EV Charging Recommendation")
st.caption("Graph Neural Network style prototype: driver–station graph + context-aware Top-N recommendation")

driver=st.sidebar.selectbox("Driver",sorted(I.driver_id.unique()))
ctype=st.sidebar.selectbox("Charging type",["Fast","Slow"])
provider=st.sidebar.selectbox("Preferred provider",["Any"]+sorted(S.provider.unique()))
hour=st.sidebar.slider("Time of day",0,23,18)
N=st.sidebar.slider("Top-N",3,6,5)

G=nx.Graph()
for d in I.driver_id.unique(): G.add_node(d,kind="driver")
for s in S.station_id: G.add_node(s,kind="station")
for _,r in I.iterrows(): G.add_edge(r.driver_id,r.station_id)

st.subheader("1. Driver–Station Graph")
fig,ax=plt.subplots(figsize=(11,4))
pos=nx.spring_layout(G,seed=7)
dn=[n for n,d in G.nodes(data=True) if d["kind"]=="driver"]
sn=[n for n,d in G.nodes(data=True) if d["kind"]=="station"]
nx.draw_networkx_nodes(G,pos,nodelist=dn,node_size=900,node_color="#D9F4EE",edgecolors="#159A89",ax=ax)
nx.draw_networkx_nodes(G,pos,nodelist=sn,node_size=1000,node_color="#E7EEFF",edgecolors="#2B65C8",ax=ax)
nx.draw_networkx_edges(G,pos,alpha=.35,ax=ax); nx.draw_networkx_labels(G,pos,ax=ax)
ax.axis("off"); st.pyplot(fig)

rows=[]
for _,s in S.iterrows():
    h=I[I.station_id==s.station_id]
    success=h.successful.mean()
    prov=h.preferred_provider.mean()
    time=np.mean(abs(h.hour-hour)<=2)
    dist=1/(1+s.distance_km)
    fast=1 if ctype=="Slow" or s.fast==1 else 0
    pref=1 if provider=="Any" or s.provider==provider else 0
    gnn=.35*success+.20*prov+.20*time+.15*dist+.10*fast
    final=.50*gnn+.15*s.availability+.15*dist+.10*fast+.05*pref+.05*time
    rows.append([s.station_id,s["name"],s.provider,s.distance_km,s.availability,gnn,final])
R=pd.DataFrame(rows,columns=["id","station","provider","distance","availability","gnn","score"]).sort_values("score",ascending=False)
R["rank"]=range(1,len(R)+1)

st.subheader("2. Top-N Recommended Stations")
V=R.head(N).copy()
V["Distance"]=V.distance.round(1).astype(str)+" km"
V["Availability"]=(V.availability*100).round().astype(int).astype(str)+"%"
V["GNN score"]=V.gnn.round(3); V["Final score"]=V.score.round(3)
st.dataframe(V[["rank","station","provider","Distance","Availability","GNN score","Final score"]],hide_index=True,use_container_width=True)

b=V.iloc[0]
st.success(f"⭐ Recommended: **{b.station}** — {b.distance:.1f} km, final score {b.score:.3f}")
st.info("GNN idea: connected nodes exchange information. Historical driver–station interactions create relationship signals; distance, availability, charging type, provider and time refine the final ranking.")
