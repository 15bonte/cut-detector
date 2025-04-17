import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
import plotly.graph_objs as go

def print_v(txt, verbose):
    if verbose:
        print(txt)

def wilcoxon_equivalence(data1, data2, delta, verbose):
    """Perform Wilcoxon signed-rank test for equivalence of two paired samples"""
    assert len(data1) == len(data2)
    data1 = np.array(data1)
    data2 = np.array(data2)
    _, p_value_1 = wilcoxon(data1, data2 - delta, alternative="greater")
    _, p_value_2 = wilcoxon(data1, data2 + delta, alternative="less")
    p_value = max(p_value_1, p_value_2)
    if p_value < 0.05:
        message = f"Wilcoxon signed-rank test: p-value = {round(p_value, 3)} -> Distribution are {delta} frames-close "
    else:
        message = f"Wilcoxon signed-rank test: p-value = {round(p_value, 3)} -> No conclusion"
    print_v(message, verbose)
    return message, p_value


def mannwhitneyu_equivalence(data1, data2, delta, verbose):
    data1 = np.array(data1)
    data2 = np.array(data2)
    _, p_value_1 = mannwhitneyu(data1, data2 - delta, alternative="greater")
    _, p_value_2 = mannwhitneyu(data1, data2 + delta, alternative="less")
    p_value = max(p_value_1, p_value_2)
    if p_value < 0.05:
        message = f"Mann-Withney U test: p-value = {round(p_value, 3)} -> Distribution are {delta} frames-close "
    else:
        message = f"Mann-Withney U test: p-value = {round(p_value, 3)} -> No conclusion"
    print_v(message, verbose)
    return message, p_value


def mannwhitneyu_difference(data1, data2, verbose):
    _, p_value = mannwhitneyu(data1, data2)
    if p_value < 0.05:
        message = f"Mann-Withney U test: p-value = {round(p_value, 3)} -> Distributions are different"
    else:
        message = f"Mann-Withney U test: p-value = {round(p_value, 3)} -> No conclusion"
    print_v(message, verbose)
    return message, p_value

def wilcoxon_difference(data1, data2, verbose):
    _, p_value = wilcoxon(data1, data2)
    if p_value < 0.05:
        message = f"Wilcoxon signed-rank test: p-value = {round(p_value, 3)} -> Distributions are different"
    else:
        message = f"Wilcoxon signed-rank test: p-value = {round(p_value, 3)} -> No conclusion"
    print_v(message, verbose)
    return message, p_value


def create_plot(data, names, title, mode):
    assert mode in ["box", "cumulative"]
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
    ]

    # Create traces for each column with specified colors
    traces = []
    for i, local_data in enumerate(data):
        if mode == "box":
            trace = go.Box(
                y=local_data,
                name=names[i],
                marker=dict(color=colors[i % len(colors)]),
                x=[i] * len(local_data),
            )
        else:
            sorted_data = np.sort(local_data)
            cumulative = np.linspace(0, 1, len(sorted_data))

            trace = go.Scatter(
                x=sorted_data,
                y=cumulative,
                name=names[i],
                mode="lines",
                marker=dict(color=colors[i % len(colors)]),
            )

        traces.append(trace)

    layout = go.Layout(
        xaxis=dict(
            title=dict(text=title, font=dict(size=20)),
        ),
        # plot_bgcolor="rgba(0,0,0,0)",  # Set the background color to transparent
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig
