import re

with open(r'c:\Users\abhyu\OneDrive\Desktop\AI Risk Spike\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'import plotly.graph_objects as go' not in content:
    content = content.replace('import pandas as pd', 'import pandas as pd\nimport plotly.graph_objects as go')

line_chart_pattern = r'st\.line_chart\(\s*line_chart_data,\s*color=\[\"#FB7185\", \"#38BDF8\", \"#FCD34D\"\]\s*\)'
line_chart_replace = '''fig = go.Figure()
    colors = ["#FB7185", "#38BDF8", "#FCD34D"]
    for i, col in enumerate(["Suspicious Rate", "Model Baseline", "Upper Threshold"]):
        fig.add_trace(go.Scatter(
            x=line_chart_data.index, y=line_chart_data[col],
            mode='lines+markers',
            marker=dict(size=6, symbol="circle-open", line=dict(width=2)),
            line=dict(color=colors[i], width=3),
            name=col
        ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, zeroline=False, color='#9CA3AF'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, color='#9CA3AF'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#D1D5DB")),
        font=dict(family="'Source Code Pro', monospace")
    )
    st.plotly_chart(fig, use_container_width=True)'''
content = re.sub(line_chart_pattern, line_chart_replace, content)

z_chart_pattern = r'st\.bar_chart\(\s*z_chart\.set_index\(\"relative_hour\"\)\[\[\"Z-Score\"\]\],\s*color=\[\"#FCD34D\"\]\s*\)'
z_chart_replace = '''fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=z_chart["relative_hour"], y=z_chart["Z-Score"],
        marker_color="#FCD34D",
        marker_line_color="rgba(252, 211, 77, 0.5)",
        marker_line_width=2
    ))
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, zeroline=False, color='#9CA3AF'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, color='#9CA3AF'),
        font=dict(family="'Source Code Pro', monospace")
    )
    st.plotly_chart(fig2, use_container_width=True)'''
content = re.sub(z_chart_pattern, z_chart_replace, content)

money_chart_pattern = r'st\.bar_chart\(\s*money_chart\.set_index\(\"relative_hour\"\)\[\[\"High Risk Amount \(\$\)\"\]\],\s*color=\[\"#F43F5E\"\]\s*\)'
money_chart_replace = '''fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=money_chart["relative_hour"], y=money_chart["High Risk Amount ($)"],
        marker_color="#F43F5E",
        marker_line_color="rgba(244, 63, 94, 0.5)",
        marker_line_width=2
    ))
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, zeroline=False, color='#9CA3AF'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, color='#9CA3AF'),
        font=dict(family="'Source Code Pro', monospace")
    )
    st.plotly_chart(fig3, use_container_width=True)'''
content = re.sub(money_chart_pattern, money_chart_replace, content)

vol_chart_pattern = r'st\.bar_chart\(\s*vol_chart\.set_index\(\"relative_hour\"\)\[\[\"Transactions\"\]\],\s*color=\[\"#06B6D4\"\]\s*\)'
vol_chart_replace = '''fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=vol_chart["relative_hour"], y=vol_chart["Transactions"],
        marker_color="#06B6D4",
        marker_line_color="rgba(6, 182, 212, 0.5)",
        marker_line_width=2
    ))
    fig4.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=False, zeroline=False, color='#9CA3AF'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, color='#9CA3AF'),
        font=dict(family="'Source Code Pro', monospace")
    )
    st.plotly_chart(fig4, use_container_width=True)'''
content = re.sub(vol_chart_pattern, vol_chart_replace, content)

with open(r'c:\Users\abhyu\OneDrive\Desktop\AI Risk Spike\app.py', 'w', encoding='utf-8') as f:
    f.write(content)
