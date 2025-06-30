import plotly.graph_objects as go
import numpy as np

x = np.array([32, 64, 256])

# single = {
#     "dp3": np.array([0.175, 0.25, 0.2]),
#     "dp3_gt": np.array([0.525, 0.775, 0.825]),
#     "dp3_gt_tax3d": np.array([0.45, 0.55, 0.525]),
#     "dp3_gt_tax3d_2": np.array([0.5, 0.925, 0.8]),
#     "tax3d": np.array([0.45, 0.45, 0.45]),
#     "tax3d_2": np.array([0.95, 0.95, 0.95]),
# }

# double = {
#     "DP3, Vanilla": np.array([0.3625, 0.4, 0.325]),
#     "DP3 + TAX3D": np.array([0.5, 0.575, 0.5875]),
#     "DP3 + Ours": np.array([0.775, 0.8375, 0.8875]),
#     "DP3 + Oracle": np.array([0.7, 0.8375, 0.85]),
#     #"tax3d": np.array([0.6125, 0.6125, 0.6125]),
#     #"tax3d_2": np.array([0.9, 0.9, 0.9]),
# }

single_hole_mean_success_rate = {
    "DP3, Vanilla": np.array([0.125, 0.230, 0.2]),
    "DP3 + TAX3D": np.array([0.4, 0.45, 0.49]),
    "DP3 + Ours": np.array([0.395, 0.7, 0.91]),
    "DP3 + Ours++": np.array([0.435, 0.715, 0.85]),
    "DP3 + Oracle": np.array([0.385, 0.67, 0.84]),
}

double_hole_mean_success_rate = {
    "DP3, Vanilla": np.array([0.3225, 0.3075, 0.2375]),
    "DP3 + TAX3D": np.array([0.52, 0.5825, 0.58]),
    "DP3 + Ours": np.array([0.645, 0.7425, 0.8425]),
    "DP3 + Ours++": np.array([0.6125, 0.8, 0.86]),
    "DP3 + Oracle": np.array([0.6875, 0.8, 0.925]),
}

def plot_methods(method_dict, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=method_dict["dp3"],
            name="DP3",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=method_dict["dp3_gt"],
            name="DP3 GT",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=method_dict["dp3_gt_tax3d"],
            name="DP3 GT + Tax3D",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=method_dict["dp3_gt_tax3d_2"],
            name="DP3 GT + Tax3D 2",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=method_dict["tax3d"],
            name="Tax3D",
            mode="lines+markers",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=method_dict["tax3d_2"],
            name="Tax3D 2",
            mode="lines+markers",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Num. Training Cloths",
        yaxis_title="Success Rate",
        width=800,
        height=600,
        legend=dict(
            x=0.5,
            y=0.2,
            orientation="h",
            xanchor="center",
            yanchor="top",
        ),
    )
    fig.show()

#task_avg = {k: (single[k] + double[k])/2 for k in single.keys()}


def plot_bar(method_dict, title):
    fig = go.Figure()
    for key, value in method_dict.items():
        fig.add_trace(
            go.Bar(
                x=["32", "64", "256"],
                y=value,
                name=key,
                textposition="auto",
                text=[f"{v:.2f}" for v in value],
            )
        )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="# Training Cloths",
        yaxis_title="Success Rate",
        width=1100,
        height=600,
        legend=dict(
            x=0.5,
            y=-0.2,
            orientation="h",
            xanchor="center",
            yanchor="top",
        ),
    )
    # make background white
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=24, family="Times New Roman"),
        legend=dict(
            title_font=dict(size=24, family="Times New Roman"),
            font=dict(size=24),
        ),
    )
    fig.show()

#plot_methods(single, title="Single Hole")
# plot_methods(double, title="Double Hole")
#plot_methods(task_avg, title="Task Average")

plot_bar(double_hole_mean_success_rate, title="Double Hole")
plot_bar(single_hole_mean_success_rate, title="Single Hole")