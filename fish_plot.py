import os
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update, exceptions

# Read the CSV file from the same directory as app.py
file_path = os.path.join(os.path.dirname(__file__), 'for plotly.csv')
df = pd.read_csv(file_path)

# Calculate average loss during fasting for each fish
def calculate_average_loss(df):
    df['Loss1'] = ((df['W2_postfast1'] - df['W1_Prefast1']) / df['W1_Prefast1']) * 100
    df['Loss2'] = ((df['W4_postfast2'] - df['W3_prefast2']) / df['W3_prefast2']) * 100
    df['Average Loss'] = (df['Loss1'] + df['Loss2']) / 2
    df['Rank'] = df['Average Loss'].rank(ascending=False)  # Rank from lowest loss to highest
    return df

df = calculate_average_loss(df)

# Create Dash app
app = Dash(__name__)
server = app.server
# Create figure
fig = go.Figure()

# Define time points
time_points = ['Initial', 'Prefast1', 'Postfast1', 'Prefast2', 'Postfast2']
weights = ['Initial Weigh', 'W1_Prefast1', 'W2_postfast1', 'W3_prefast2', 'W4_postfast2']
lengths = ['Initial Length', 'L1_Prefast1', 'L2_postfast1', 'L3_prefast2', 'L4_postfast2']
symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up']
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Store original marker properties
original_markers = []

# Add scatter plots for each sampling point
for i, (time, weight, length, symbol, color) in enumerate(zip(time_points, weights, lengths, symbols, colors)):
    valid_data = df[(df[weight] > 0) & (df[length] > 0)]
    scatter = go.Scatter(
        x=valid_data[weight],
        y=valid_data[length],
        mode='markers',
        name=time,
        marker=dict(symbol=symbol, size=10, color=color),
        customdata=valid_data['Tag'],
        hovertemplate='Tag: %{customdata}<br>Weight: %{x}<br>Length: %{y}'
    )
    fig.add_trace(scatter)
    original_markers.append({'symbol': symbol, 'color': color, 'size': 10})

# Add a trace for the connecting line
fig.add_trace(
    go.Scatter(
        x=[], y=[], mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(size=10),
        showlegend=False,
        hoverinfo='text',
        visible=False
    )
)

# Initialize layout with uirevision to maintain zoom/pan state
fig.update_layout(
    title={
        'text': 'Fish weight vs. length in fasting trial 2024',
        'x': 0.5,  # Center the title horizontally
        'y': 0.98,  # Position the title inside the plot area
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 14}  # Reduced font size
    },
    xaxis_title='Weight (g)',
    yaxis_title='Length (cm)',
    hovermode='closest',
    clickmode='event+select',
    height=860,  # Set the height of the plot
    margin=dict(l=40, r=20, t=40, b=60),  # Adjusted margins for better fit
    legend=dict(x=1, y=1),  # Position the legend
    uirevision='constant'  # This ensures the zoom state is preserved
)

app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Div([
        html.Label("Number of Top-Ranked Fish to Display:", htmlFor='top-rank-input', style={'font-size': '18px', 'margin-right': '10px'}),
        dcc.Input(id='top-rank-input', type='number', value=len(df), min=1, max=len(df), step=1, style={'margin-right': '20px'}),
        html.Button("Reset", id="reset-button", n_clicks=0)
    ], style={'margin-bottom': '20px', 'display': 'flex', 'align-items': 'center'}),
    dcc.Store(id='selected-tag', data=None),
    dcc.Store(id='initial-zoom', data={'xaxis.range': [None, None], 'yaxis.range': [None, None]}),
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('selected-tag', 'data'),
    Output('initial-zoom', 'data'),
    Output('top-rank-input', 'value'),  # Update the value of the input field
    Input('scatter-plot', 'clickData'),
    Input('scatter-plot', 'relayoutData'),
    Input('reset-button', 'n_clicks'),
    Input('top-rank-input', 'value'),
    State('selected-tag', 'data'),
    State('initial-zoom', 'data')
)
def update_figure(click_data, relayout_data, n_clicks, top_rank, selected_tag, initial_zoom):
    ctx = callback_context
    fig_update = go.Figure(fig)  # Create a copy of the figure to update

    if not ctx.triggered:
        raise exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Function to reset opacity to full brightness
    def reset_opacity(fig_update):
        for i in range(len(time_points)):
            fig_update.data[i].update(
                marker=dict(
                    opacity=1,  # Reset to full visibility
                    symbol=original_markers[i]['symbol'],
                    color=original_markers[i]['color'],
                    size=original_markers[i]['size']
                )
            )

    # Handle reset button click
    if triggered_id == 'reset-button':
        reset_opacity(fig_update)
        selected_tag = None
        initial_zoom = {'xaxis.range': [None, None], 'yaxis.range': [None, None]}  # Reset zoom state
        fig_update.update_layout(annotations=[], shapes=[])  # Remove any annotations or shapes
        top_rank = len(df)  # Reset top-rank input to show all data points
        return fig_update, selected_tag, initial_zoom, top_rank

    if triggered_id in ['top-rank-input', 'scatter-plot']:
        # Apply the top-rank filter to the data
        filtered_df = df[df['Rank'] <= top_rank]
        for i in range(len(time_points)):
            valid_data = filtered_df[(filtered_df[weights[i]] > 0) & (filtered_df[lengths[i]] > 0)]
            fig_update.data[i].update(
                x=valid_data[weights[i]],
                y=valid_data[lengths[i]],
                customdata=valid_data['Tag']
            )

    if triggered_id == 'scatter-plot' and click_data:
        clicked_tag = click_data['points'][0]['customdata']

        # Toggle selection
        if selected_tag == clicked_tag:
            selected_tag = None  # Deselect if the same fish is clicked again
            reset_opacity(fig_update)  # Reset opacity if deselecting
        else:
            selected_tag = clicked_tag  # Select the new fish

            # Gather x and y values for the connecting line
            if selected_tag:
                x = []
                y = []
                hover_text = []
                line_marker_symbols = []
                line_marker_colors = []
                for i in range(len(time_points)):
                    weight = weights[i]
                    length = lengths[i]
                    point_data = df[df['Tag'] == selected_tag]
                    if not point_data.empty and weight in point_data.columns and length in point_data.columns:
                        if not point_data[weight].isna().any() and not point_data[length].isna().any():
                            x.append(point_data[weight].values[0])
                            y.append(point_data[length].values[0])
                            hover_text.append(f'{time_points[i]}<br>Tag: {selected_tag}<br>Weight: {point_data[weight].values[0]}<br>Length: {point_data[length].values[0]}')
                            line_marker_symbols.append(original_markers[i]['symbol'])
                            line_marker_colors.append(original_markers[i]['color'])

                fig_update.data[-1].update(
                    x=x, 
                    y=y, 
                    visible=True, 
                    customdata=[selected_tag]*len(x), 
                    hovertext=hover_text, 
                    hoverinfo='text',
                    marker=dict(symbol=line_marker_symbols, color=line_marker_colors, size=10)
                )

            # Update opacity for each point while maintaining the color and shape
            for i in range(len(time_points)):
                marker_opacity = [1 if tag == selected_tag else 0.1 for tag in fig_update.data[i].customdata]
                fig_update.data[i].update(marker=dict(
                    opacity=marker_opacity,
                    symbol=original_markers[i]['symbol'],
                    color=original_markers[i]['color'],
                    size=original_markers[i]['size']
                ))

            if selected_tag:
                # Display additional information for the selected fish with weight differences
                tag_info = df[df['Tag'] == selected_tag]
                if not tag_info.empty:
                    fasting_percentages = []
                    for i in range(1, len(time_points), 2):  # Calculate percentages for Postfast1 and Postfast2
                        prefast_weight = tag_info[weights[i]].values[0]
                        postfast_weight = tag_info[weights[i + 1]].values[0]
                        percent_diff = (postfast_weight - prefast_weight) / prefast_weight * 100
                        fasting_percentages.append(percent_diff)

                    avg_fasting_loss = sum(fasting_percentages) / len(fasting_percentages)
                    
                    # Calculate percentage changes between time points
                    weight_changes = []
                    for i in range(1, len(weights)):
                        weight_diff = (tag_info[weights[i]].values[0] - tag_info[weights[i-1]].values[0]) / tag_info[weights[i-1]].values[0] * 100
                        weight_changes.append(round(weight_diff, 1))

                    tag_display = f"<b>Selected Tag:</b> {selected_tag}<br>"
                    tag_display += f"<b>Rank:</b> {int(tag_info['Rank'].values[0])}<br>"
                    tag_display += f"Initial W: {tag_info[weights[0]].values[0]}, L: {tag_info[lengths[0]].values[0]}<br>"
                    tag_display += f"Prefast1 W: {tag_info[weights[1]].values[0]} ({weight_changes[0]:+.1f}%), L: {tag_info[lengths[1]].values[0]}<br>"
                    tag_display += f"Postfast1 W: {tag_info[weights[2]].values[0]} ({weight_changes[1]:+.1f}%), L: {tag_info[lengths[2]].values[0]}<br>"
                    tag_display += f"Prefast2 W: {tag_info[weights[3]].values[0]} ({weight_changes[2]:+.1f}%), L: {tag_info[lengths[3]].values[0]}<br>"
                    tag_display += f"Postfast2 W: {tag_info[weights[4]].values[0]} ({weight_changes[3]:+.1f}%), L: {tag_info[lengths[4]].values[0]}<br>"
                    tag_display += f"<b>Average Fasting Loss:</b> {avg_fasting_loss:.2f}%"

                    # Add annotation or shape with the selected fish information in the upper-left corner
                    fig_update.update_layout(
                        annotations=[dict(
                            xref='paper', yref='paper', x=0.05, y=0.95,  # Position in the upper-left corner
                            text=tag_display,
                            showarrow=False,
                            font=dict(size=12),
                            align='left',
                            bordercolor='black',
                            borderwidth=1,
                            bgcolor='white',
                            opacity=0.8
                        )]
                    )
            else:
                fig_update.data[-1].update(visible=False)
                fig_update.update_layout(annotations=[])

    elif triggered_id == 'scatter-plot' and not click_data:
        # Reset to show all points when clicking on an empty space
        selected_tag = None
        reset_opacity(fig_update)
        fig_update.data[-1].update(visible=False)
        fig_update.update_layout(annotations=[])

    # Handle zoom/pan state
    if triggered_id == 'scatter-plot' and relayout_data:
        # Store the new zoom/pan state
        initial_zoom['xaxis.range'] = relayout_data.get('xaxis.range', initial_zoom['xaxis.range'])
        initial_zoom['yaxis.range'] = relayout_data.get('yaxis.range', initial_zoom['yaxis.range'])

    # Apply zoom state if it exists
    fig_update.update_layout(
        xaxis=dict(range=initial_zoom['xaxis.range']),
        yaxis=dict(range=initial_zoom['yaxis.range'])
    )

    return fig_update, selected_tag, initial_zoom, no_update


if __name__ == '__main__':
    app.run_server(debug=True)
