from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis.rrg import RRGPoint


def generate_line_chart_html(series: pd.Series, title: str, ylabel: str = "", height: int = 350) -> str:
    """
    Generates a Plotly line chart HTML string with institutional styling.
    """
    fig = go.Figure()

    # Maroon for institutional feel
    line_color = "#8B0021" 

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        name=title,
        line=dict(color=line_color, width=2.5),
        hovertemplate='%{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Outfit, sans-serif", size=16, color="#1A1A1A", weight=600),
            x=0,
            xanchor='left'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif", color="#5A5A5A"),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor='#E5E7EB',
            zeroline=False,
            showline=False,
            tickformat='%b %d'
        ),
        yaxis=dict(
            title=ylabel,
            showgrid=True,
            gridcolor='#E5E7EB',
            zeroline=False,
            showline=False
        ),
        hovermode="x unified"
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})


def generate_rrg_chart_html(points: List[RRGPoint], title: str, height: int = 500) -> str:
    """
    Generates an institutional-style Relative Rotation Graph.
    """
    fig = go.Figure()

    # Create scatter plot with refined colors
    for p in points:
        # Professional Institutional Palette
        if p.rs >= 100 and p.mom >= 100:
            color = "#065F46"  # Leading (Emerald)
        elif p.rs < 100 and p.mom < 100:
            color = "#991B1B"  # Lagging (Crimson)
        elif p.rs < 100 and p.mom >= 100:
            color = "#3730A3"  # Improving (Indigo)
        else:
            color = "#B45309"  # Weakening (Gold)

        fig.add_trace(go.Scatter(
            x=[p.rs],
            y=[p.mom],
            mode='markers+text',
            name=p.label,
            text=[p.label],
            textposition="top center",
            marker=dict(size=14, color=color, line=dict(width=1.5, color="white")),
            hovertemplate=f"<b>{p.label}</b><br>RS: %{{x:.2f}}<br>Mom: %{{y:.2f}}<extra></extra>"
        ))

    # Add Quadrant Lines
    fig.add_hline(y=100, line_width=1.5, line_color="#D1D1D1", line_dash="solid")
    fig.add_vline(x=100, line_width=1.5, line_color="#D1D1D1", line_dash="solid")

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Outfit, sans-serif", size=18, color="#1A1A1A", weight=600),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font=dict(family="Outfit, sans-serif", color="#5A5A5A"),
        height=height,
        margin=dict(l=60, r=60, t=80, b=60),
        xaxis=dict(
            title=dict(
                text="RELATIVE STRENGTH (RS)",
                font=dict(size=10)
            ),
            showgrid=True,
            gridcolor='#F3F4F6',
            zeroline=False,
            range=[95, 105] if not points else None
        ),
        yaxis=dict(
            title=dict(
                text="MOMENTUM",
                font=dict(size=10)
            ),
            showgrid=True,
            gridcolor='#F3F4F6',
            zeroline=False,
            range=[95, 105] if not points else None
        )
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
