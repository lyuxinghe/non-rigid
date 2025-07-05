import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

single_hole_coverage = np.array([
    0.822, 0.828, 0.940, 1.723, 2.587, 3.350, 
])
single_hole_coverage_error = np.array([
    0.046, 0.041, 0.153, 0.405, 0.378, 0.316,
])

single_hole_precision = np.array([
    0.949, 0.927, 1.077, 2.028, 3.330, 5.566,
])
single_hole_precision_error = np.array([
    0.028, 0.017, 0.117, 0.825, 1.115, 1.402,
])

single_hole_small_success = np.array([
    0.4, 0.425, 0.5, 0.525, 0.575, 0.4,
])
single_hole_medium_success = np.array([
    0.675, 0.7, 0.775, 0.75, 0.6, 0.425,
])
single_hole_big_success = np.array([
    0.9, 0.875, 0.875, 0.825, 0.625, 0.55,
])


double_hole_coverage = np.array([
    0.496, 0.536, 0.605, 0.823, 1.080, 1.452,
])
double_hole_coverage_error = np.array([
    0.015, 0.025, 0.049, 0.137, 0.171, 0.336,
])

double_hole_precision = np.array([
    0.584, 0.611, 0.667, 0.934, 1.428, 2.160,
])
double_hole_precision_error = np.array([
    0.012, 0.017, 0.047, 0.176, 0.339, 0.503,
])

double_hole_small_success = np.array([
    0.65, 0.6375, 0.65, 0.5875, 0.55, 0.4
])
double_hole_medium_success = np.array([
    0.7375, 0.7875, 0.7875, 0.775, 0.6625, 0.6125
])
double_hole_big_success = np.array([
    0.8375, 0.7875, 0.8, 0.7875, 0.6875, 0.5875
])


fig_x = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
fig = make_subplots(rows=1, cols=2, subplot_titles=("Point Prediction Errors", "Policy Evaluations"))
################# PLOTTING COVERAGE AND PRECISION #################
# sfig = go.Figure()
# first, just plotting coverage and precision
# fig.add_trace(
#     go.Scatter(
#         x=fig_x,
#         y=single_hole_coverage,
#         name="Coverage RMSE",
#         mode="lines+markers",
#         line=dict(color='rgba(255, 128, 0, 1.0)'),
#     ),
#     row=1, col=1
# )
# fig.add_trace(
#     go.Scatter(
#         x=fig_x,
#         y=single_hole_precision,
#         name="Precision RMSE",
#         mode="lines+markers",
#         line=dict(color='rgba(0, 128, 255, 1.0)'),
#     ),
#     row=1, col=1
# )
fig.add_trace(
    go.Scatter(
        x=fig_x,
        y=double_hole_coverage,
        name="Coverage RMSE",
        mode="lines+markers",
        line=dict(color='rgba(255, 128, 0, 1.0)'),
        legendgroup="point",
        legendgrouptitle_text="Point Errors",
        showlegend=True,
        # showlegend=False,  # Hide legend for double hole coverage
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=fig_x,
        y=double_hole_precision,
        name="Precision RMSE",
        mode="lines+markers",
        line=dict(color='rgba(0, 128, 255, 1.0)'),
        legendgroup="point",
        legendgrouptitle_text="Point Errors",
        showlegend=True,
        # showlegend=False,  # Hide legend for double hole precision
    ),
    row=1, col=1
)

# then, adding error bars
# single_hole_coverage_upper = single_hole_coverage + single_hole_coverage_error
# single_hole_coverage_lower = single_hole_coverage - single_hole_coverage_error
# fig.add_trace(
#     go.Scatter(
#         x=np.concatenate([fig_x, fig_x[::-1]]),
#         y=np.concatenate([single_hole_coverage_upper, single_hole_coverage_lower[::-1]]),
#         fill='toself',
#         fillcolor='rgba(255, 128, 0, 0.1)',
#         line=dict(color='rgba(0, 0, 0, 0.0)', width=0),
#         hoverinfo='skip',
#         showlegend=False,
#     ),
#     row=1, col=1
# )
# single_hole_precision_upper = single_hole_precision + single_hole_precision_error
# single_hole_precision_lower = single_hole_precision - single_hole_precision_error
# fig.add_trace(
#     go.Scatter(
#         x=np.concatenate([fig_x, fig_x[::-1]]),
#         y=np.concatenate([single_hole_precision_upper, single_hole_precision_lower[::-1]]),
#         fill='toself',
#         fillcolor='rgba(0, 128, 255, 0.1)',
#         line=dict(color='rgba(0, 0, 0, 0.0)', width=0),
#         hoverinfo='skip',
#         showlegend=False,
#     ),
#     row=1, col=1
# )
double_hole_coverage_upper = double_hole_coverage + double_hole_coverage_error
double_hole_coverage_lower = double_hole_coverage - double_hole_coverage_error
fig.add_trace(
    go.Scatter(
        x=np.concatenate([fig_x, fig_x[::-1]]),
        y=np.concatenate([double_hole_coverage_upper, double_hole_coverage_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 128, 0, 0.1)',
        line=dict(color='rgba(0, 0, 0, 0.0)', width=0),
        hoverinfo='skip',
        showlegend=False,
    ),
    row=1, col=1
)
double_hole_precision_upper = double_hole_precision + double_hole_precision_error
double_hole_precision_lower = double_hole_precision - double_hole_precision_error
fig.add_trace(
    go.Scatter(
        x=np.concatenate([fig_x, fig_x[::-1]]),
        y=np.concatenate([double_hole_precision_upper, double_hole_precision_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 128, 255, 0.1)',
        line=dict(color='rgba(0, 0, 0, 0.0)', width=0),
        hoverinfo='skip',
        showlegend=False,
    ),
    row=1, col=1
)


################### PLOTTING SUCCESS RATES #################
# fig.add_trace(
#     go.Scatter(
#         x=fig_x,
#         y=single_hole_small_success,
#         name="Success Rate (32 Training Cloths)",
#         mode="lines+markers",
#         line=dict(color='rgba(255, 0, 0, 1.0)'),
#     ),
#     row=2, col=1
# )
# fig.add_trace(
#     go.Scatter(
#         x=fig_x,
#         y=single_hole_medium_success,
#         name="Success Rate (64 Training Cloths)",
#         mode="lines+markers",
#         line=dict(color='rgba(0, 255, 0, 1.0)'),
#     ),
#     row=2, col=1
# )
# fig.add_trace(
#     go.Scatter(
#         x=fig_x,
#         y=single_hole_big_success,
#         name="Success Rate (256 Training Cloths)",
#         mode="lines+markers",
#         line=dict(color='rgba(0, 0, 255, 1.0)'),
#     ),
#     row=2, col=1
# )

fig.add_trace(
    go.Scatter(
        x=fig_x,
        y=double_hole_small_success,
        name="128 Training Demos",
        mode="lines+markers",
        line=dict(color='rgba(255, 0, 0, 1.0)'),
        legendgroup="policy",
        legendgrouptitle_text="Policy Evaluations",
        showlegend=True,
        # showlegend=False,  # Hide legend for double hole small success
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=fig_x,
        y=double_hole_medium_success,
        name="256 Training Demos",
        mode="lines+markers",
        line=dict(color='rgba(0, 255, 0, 1.0)'),
        legendgroup="policy",
        legendgrouptitle_text="Policy Evaluations",
        showlegend=True,
        # showlegend=False,  # Hide legend for double hole medium success
    ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(
        x=fig_x,
        y=double_hole_big_success,
        name="1024 Training Demos",
        mode="lines+markers",
        line=dict(color='rgba(0, 0, 255, 1.0)'),
        legendgroup="policy",
        legendgrouptitle_text="Policy Evaluations",
        showlegend=True,
        # showlegend=False,  # Hide legend for double hole big success
    ),
    row=1, col=2
)

fig.update_layout(
    legend=dict(
        x=1.3, y=0.95, xanchor='right', yanchor='top',
        tracegroupgap=20,  # Distance between legend groups
        font=dict(size=24),
    )
)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title="GMM Noise Level",
    xaxis2_title="GMM Noise Level",
    yaxis_title="RMSE",
    yaxis2_title="Success Rate",
    width=1200,
    height=480,
    font=dict(
        color="black",
        size=24,
      
        family="Times New Roman"
    ),
)

# Change the font size of each subplot title
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=24)  # 🔧 Adjust size here

fig.update_xaxes(visible=True, showline=True, linewidth=2, linecolor='black', gridcolor='lightgray')
fig.update_yaxes(visible=True, showline=True, linewidth=2, linecolor='black', gridcolor='lightgray')


fig.show()