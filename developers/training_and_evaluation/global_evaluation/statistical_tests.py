import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
import plotly.graph_objs as go


def wilcoxon_equivalence(data1, data2, delta):
    """Perform Wilcoxon signed-rank test for equivalence of two paired samples"""
    assert len(data1) == len(data2)
    data1 = np.array(data1)
    data2 = np.array(data2)
    _, p_value_1 = wilcoxon(data1, data2 - delta, alternative="greater")
    _, p_value_2 = wilcoxon(data1, data2 + delta, alternative="less")
    p_value = max(p_value_1, p_value_2)
    if p_value < 0.05:
        message = f"Wilcoxon signed-rank test:\n p-value = {round(p_value, 3)}\n -> Distribution are {delta} frames-close "
    else:
        message = f"Wilcoxon signed-rank test:\n p-value = {round(p_value, 3)}\n -> No conclusion"
    print(message)
    return message


def mannwhitneyu_equivalence(data1, data2, delta):
    data1 = np.array(data1)
    data2 = np.array(data2)
    _, p_value_1 = mannwhitneyu(data1, data2 - delta, alternative="greater")
    _, p_value_2 = mannwhitneyu(data1, data2 + delta, alternative="less")
    p_value = max(p_value_1, p_value_2)
    if p_value < 0.05:
        message = f"Mann-Withney U test:\n p-value = {round(p_value, 3)}\n -> Distribution are {delta} frames-close "
    else:
        message = f"Mann-Withney U test:\n p-value = {round(p_value, 3)}\n -> No conclusion"
    print(message)
    return message


def create_plot(data, names, title):
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
    ]

    # Create traces for each column with specified colors
    traces = [
        go.Box(
            y=data[i],
            name=names[i],
            marker=dict(color=colors[i % len(colors)]),
            x=[i] * len(data[i]),
        )
        for i in range(len(data))
    ]
    layout = go.Layout(
        xaxis=dict(
            title=dict(text=title, font=dict(size=20)),
        ),
        # plot_bgcolor="rgba(0,0,0,0)",  # Set the background color to transparent
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig
