import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math

class PlotlyFigure:
    def __init__(self, fig):
        self.fig = fig
        self.show = False

    def set_color(self, color):
        for it in self.fig.data:
            it.line.color = color
        if self.show:
            self.fig.show()

    def show(self):
        self.fig.show()

def plot(x, y, color='blue', dash='solid', legend='', title='Untitled', xaxis_title='x', yaxis_title='y', show=True):
    # Create a Plotly figure
    fig = go.Figure()
    pfig = PlotlyFigure(fig)

    # Add the scatter plot
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=legend,
        line=dict(color=color, dash=dash)
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True if len(legend) > 0 else False
    )
    if show: fig.show()
    return pfig

def scatter(x, y, color='blue', marker_symbol='circle', legend='', title='Untitled', xaxis_title='x', yaxis_title='y', show=True, classes=None):
    # Create a Plotly figure
    fig = go.Figure()
    pfig = PlotlyFigure(fig)

    # Add the scatter plot
    if classes is not None and len(classes) == len(x):
        # If classes are provided and match the length of x and y, plot each class with a different color
        for cls in set(classes):
            idx = [i for i, c in enumerate(classes) if c == cls]
            fig.add_trace(go.Scatter(
                x=[x[i] for i in idx],
                y=[y[i] for i in idx],
                mode='markers',
                name=f'{legend} {cls}',
                marker=dict(symbol=marker_symbol)
            ))
    else:
        # If classes are not provided or don't match in length, plot all points with the default color
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name=legend,
            marker=dict(color=color, symbol=marker_symbol)
        ))


    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True if len(legend) > 0 else False
    )
    if show: fig.show()
    return pfig

def plot_heatmaps(weight_images, grid_size=None, title="Heatmaps", fig_size=(15, 20), show=True):
    """
    Plots the weights as heatmaps using Plotly.

    Parameters:
    - weight_images: A list or array of weight images to plot.
    - fig_size: Tuple (width, height) for the size of the figure.
    - grid_size: Tuple (num_rows, num_cols) for the grid size of subplots, or None to infer.
    - title: Title of the entire figure.
    """

    # Calculate the number of rows and columns for subplots
    if grid_size is None:
        num_images = len(weight_images)
        num_cols = min(8, math.ceil(math.sqrt(num_images)))
        num_rows = math.ceil(num_images / num_cols)
    else:
        num_rows, num_cols = grid_size

    # Create a subplot figure
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f'Neuron {i}' for i in range(num_rows * num_cols)])

    # Add each weight image as a subplot
    for i in range(num_rows * num_cols):
        if i < len(weight_images):  # Check to avoid index error if number of images is less than grid slots
            row = i // num_cols + 1
            col = i % num_cols + 1
            fig.add_trace(
                go.Heatmap(z=np.abs(weight_images[i]), colorscale='Greys', showscale=False),
                row=row, col=col
            )

    # Update layout
    fig.update_layout(height=fig_size[1], width=fig_size[0], title_text=title, showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    if show: fig.show()
    pfig = PlotlyFigure(fig)
    return pfig
