import os
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Read the CSV file from the same directory as app.py
file_path = os.path.join(os.path.dirname(__file__), 'for plotly.csv')
df = pd.read_csv(file_path)

# Calculate the necessary metrics
def calculate_metrics(df):
    df['Loss1'] = ((df['W2_postfast1'] - df['W1_Prefast1']) / df['W1_Prefast1']) * 100
    df['Loss2'] = ((df['W4_postfast2'] - df['W3_prefast2']) / df['W3_prefast2']) * 100
    df['Average Loss'] = (df['Loss1'] + df['Loss2']) / 2
    df['Change1'] = ((df['W1_Prefast1'] - df['Initial Weigh']) / df['Initial Weigh']) * 100
    df['Change2'] = ((df['W2_postfast1'] - df['W1_Prefast1']) / df['W1_Prefast1']) * 100
    df['Change3'] = ((df['W3_prefast2'] - df['W2_postfast1']) / df['W2_postfast1']) * 100
    df['Change4'] = ((df['W4_postfast2'] - df['W3_prefast2']) / df['W3_prefast2']) * 100
    df['Average Feeding Change'] = (df['Change1'] + df['Change3']) / 2
    df['Whole Trial Average'] = (df[['Change1', 'Change2', 'Change3', 'Change4']].mean(axis=1))
    return df

df = calculate_metrics(df)

# Rank the percentage changes for all fish
df['RankChange1'] = df['Change1'].rank(ascending=False, na_option='bottom').fillna(len(df)).astype(int)
df['RankChange2'] = df['Change2'].rank(ascending=False, na_option='bottom').fillna(len(df)).astype(int)
df['RankChange3'] = df['Change3'].rank(ascending=False, na_option='bottom').fillna(len(df)).astype(int)
df['RankChange4'] = df['Change4'].rank(ascending=False, na_option='bottom').fillna(len(df)).astype(int)
df['RankAverageFeedingChange'] = df['Average Feeding Change'].rank(ascending=False, na_option='bottom').fillna(len(df)).astype(int)
df['RankWholeTrialAverage'] = df['Whole Trial Average'].rank(ascending=False, na_option='bottom').fillna(len(df)).astype(int)

# Create Dash app
app = Dash(__name__)

# Layout with title, input fields, and sorting dropdown
app.layout = html.Div([
    html.Div("Fish Weights Across Time Points in Fasting Trial 2024", style={'font-weight': 'bold', 'text-align': 'center', 'font-size': '24px'}),
    html.Div([
        html.Label("Top Ranked Fish from ", style={'margin-right': '10px'}),
        dcc.Input(id='rank-start', type='number', value=1, min=1, max=len(df), step=1, style={'width': '40px', 'margin-right': '5px'}),
        html.Label(" to ", style={'margin-right': '10px'}),
        dcc.Input(id='rank-end', type='number', value=24, min=1, max=len(df), step=1, style={'width': '40px', 'margin-right': '5px'}),
        html.Label(" (Weight Change during:", style={'margin-right': '5px'}),
        dcc.Dropdown(
            id='sort-criteria',
            options=[
                {'label': 'Both Fastings Average', 'value': 'Average Loss'},
                {'label': 'First Feeding', 'value': 'Change1'},
                {'label': 'First Fasting', 'value': 'Change2'},
                {'label': 'Second Feeding', 'value': 'Change3'},
                {'label': 'Second Fasting', 'value': 'Change4'},
                {'label': 'Both Feeding Average', 'value': 'Average Feeding Change'},
                {'label': 'Whole Trial Average', 'value': 'Whole Trial Average'}
            ],
            value='Average Loss',
            clearable=False,
            style={'width': '250px', 'margin-right': '10px'}
        ),
        html.Label(")", style={'margin-right': '5px'})
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-bottom': '0px'}),
    dcc.Graph(id='bar-plot')
])

@app.callback(
    Output('bar-plot', 'figure'),
    Input('rank-start', 'value'),
    Input('rank-end', 'value'),
    Input('sort-criteria', 'value')
)
def update_figure(rank_start, rank_end, sort_criteria):
    # Sort the DataFrame based on the selected sorting criteria
    df_sorted = df.sort_values(by=sort_criteria, ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = df_sorted.index + 1  # Recalculate ranks based on the new sorting

    # Select the range of fish to display
    df_top = df_sorted[(df_sorted['Rank'] >= rank_start) & (df_sorted['Rank'] <= rank_end)]

    # Create a bar chart
    bar_fig = go.Figure()

    shapes = []
    # Add bars for each fish's absolute weight values at each time point in correct order
    for index, row in df_top.iterrows():
        if pd.isna(row['Rank']):
            continue  # Skip rows where Rank is NaN
        
        fish_tag = row['Tag']
        rank = int(row['Rank'])  # Convert Rank to integer if it's not NaN

        # Add the bars from the last to the first time point
        bar_fig.add_trace(go.Bar(
            y=[rank],  # Align the bar for the first weight point (Initial Weigh to W1_Prefast1)
            x=[row['W4_postfast2'] - row['W3_prefast2']] if row['W4_postfast2'] != row['W3_prefast2'] else [0.5],
            orientation='h',
            width=0.2,  # Thinner bars
            base=row['W3_prefast2'],  # Start the bar at the "W3_prefast2" value
            marker=dict(color='purple'),
            name='W3_prefast2 to W4_postfast2',
            hovertemplate=f'Tag: {fish_tag}<br>From: {row["W3_prefast2"]} g<br>To: {row["W4_postfast2"]} g'
                          f'<br>Change: {row["Change4"]:.1f}%<br>Rank: {row["RankChange4"]}'
        ))

        bar_fig.add_trace(go.Bar(
            y=[rank - 0.25],  # Slight offset for the second bar (W2_postfast1 to W3_prefast2)
            x=[row['W3_prefast2'] - row['W2_postfast1']] if row['W3_prefast2'] != row['W2_postfast1'] else [0.5],
            orientation='h',
            width=0.2,  # Thinner bars
            base=row['W2_postfast1'],  # Start the bar at the "W2_postfast1" value
            marker=dict(color='red'),
            name='W2_postfast1 to W3_prefast2',
            hovertemplate=f'Tag: {fish_tag}<br>From: {row["W2_postfast1"]} g<br>To: {row["W3_prefast2"]} g'
                          f'<br>Change: {row["Change3"]:.1f}%<br>Rank: {row["RankChange3"]}'
        ))

        bar_fig.add_trace(go.Bar(
            y=[rank - 0.5],  # Slight offset for the third bar (W1_Prefast1 to W2_postfast1)
            x=[row['W2_postfast1'] - row['W1_Prefast1']] if row['W2_postfast1'] != row['W1_Prefast1'] else [0.5],
            orientation='h',
            width=0.2,  # Thinner bars
            base=row['W1_Prefast1'],  # Start the bar at the "W1_Prefast1" value
            marker=dict(color='green'),
            name='W1_Prefast1 to W2_postfast1',
            hovertemplate=f'Tag: {fish_tag}<br>From: {row["W1_Prefast1"]} g<br>To: {row["W2_postfast1"]} g'
                          f'<br>Change: {row["Change2"]:.1f}%<br>Rank: {row["RankChange2"]}'
        ))

        bar_fig.add_trace(go.Bar(
            y=[rank - 0.75],  # Position the last bar at the lowest position for this rank (Initial Weigh to W1_Prefast1)
            x=[row['W1_Prefast1'] - row['Initial Weigh']] if row['W1_Prefast1'] != row['Initial Weigh'] else [0.5],
            orientation='h',
            width=0.2,  # Thinner bars
            base=row['Initial Weigh'],  # Start the bar at the "Initial Weigh" value
            marker=dict(color='blue'),
            name='Initial Weigh to W1_Prefast1',
            hovertemplate=f'Tag: {fish_tag}<br>From: {row["Initial Weigh"]} g<br>To: {row["W1_Prefast1"]} g'
                          f'<br>Change: {row["Change1"]:.1f}%<br>Rank: {row["RankChange1"]}'
        ))

        # Add a horizontal line to separate each fish
        shapes.append(
            dict(
                type="line",
                x0=0,
                x1=1,
                xref="paper",
                y0=rank - 0.88,
                y1=rank - 0.88,
                line=dict(color="lightgray", width=1.5, dash='solid')  # Faded but visible line
            )
        )

    # Update bar chart layout with the annotations (lines) and a legend containing small bars and descriptors
    bar_fig.update_layout(
        xaxis_title='Weight (g)',
        yaxis_title='Rank (Top Rank at the Top)',
        yaxis=dict(
            tickmode='array',
            tickvals=[i + 0.6 for i in range(len(df_top))],  # Adjusted tickvals to center labels
            ticktext=[f'#{i + 1} - ...{str(row["Tag"])[-5:]}' for i, (_, row) in enumerate(df_top.iterrows(), start=0)],
            autorange='reversed',  # Reverse the y-axis so top rank is at the top
            dtick=1,  # Ensure ticks are placed every 1 unit
            range=[len(df_top) + 0.5, 0.5],  # Ensure the y-axis starts at 1
            showgrid=False  # Hide the y-axis grid lines
        ),
        hovermode='closest',
        showlegend=False,  # Hide the default legend
        height=860,  # Set the height of the plot to 860
        margin=dict(l=100, r=20, t=25, b=40),  # Adjust the margins
        shapes=shapes + [
            # Blue bar legend
            dict(
                type="rect", x0=0.95, x1=0.99, y0=0.98, y1=0.99, xref="paper", yref="paper",
                line=dict(width=0), fillcolor="blue"
            ),
            # Green bar legend
            dict(
                type="rect", x0=0.95, x1=0.99, y0=0.95, y1=0.96, xref="paper", yref="paper",
                line=dict(width=0), fillcolor="green"
            ),
            # Red bar legend
            dict(
                type="rect", x0=0.95, x1=0.99, y0=0.92, y1=0.93, xref="paper", yref="paper",
                line=dict(width=0), fillcolor="red"
            ),
            # Purple bar legend
            dict(
                type="rect", x0=0.95, x1=0.99, y0=0.89, y1=0.90, xref="paper", yref="paper",
                line=dict(width=0), fillcolor="purple"
            )
        ],
        annotations=[
            dict(
                x=0.991, y=0.984, xref='paper', yref='paper',
                text='First Feeding',
                showarrow=False, font=dict(size=10), align='left'
            ),
            dict(
                x=0.991, y=0.954, xref='paper', yref='paper',
                text='First Fasting',
                showarrow=False, font=dict(size=10), align='left'
            ),
            dict(
                x=0.991, y=0.924, xref='paper', yref='paper',
                text='Second Feeding',
                showarrow=False, font=dict(size=10), align='left'
            ),
            dict(
                x=0.991, y=0.894, xref='paper', yref='paper',
                text='Second Fasting',
                showarrow=False, font=dict(size=10), align='left'
            )
        ]
    )

    return bar_fig



server = app.server ## This is Flask App. This is Important.


if __name__ == '__main__':
    app.run_server(debug=True)


# Add this line at the end:
server = app.server