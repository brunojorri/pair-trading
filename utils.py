# utils.py
import plotly.graph_objects as go
import pandas as pd
from typing import List
from trading_calendar import get_session_boundaries
import streamlit as st

def plot_pair_analysis(data: pd.DataFrame, pair: tuple, beta: float):
    s1, s2 = pair
    spread = data[s1] - beta * data[s2]
    zscore = (spread - spread.mean()) / spread.std()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values, name="Spread", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore.values, name="Z-Score", line=dict(color="red", dash="dot")))
    
    fig.update_layout(
        title=f"Spread e Z-Score: {s1} vs {s2}",
        xaxis_title="Data",
        yaxis_title="Valor",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)