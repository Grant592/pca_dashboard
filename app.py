import pandas as pd
import numpy as np
import dash
import dash_table
import dash_core_components as dcc
import  dash_html_components as html
from dash.dependencies import Input, Output
import datetime
from PCA_class import PCA_class

pca = PCA_class('whole_period_gps.csv', 'wellness_data.csv')
app = dash.Dash(__name__)

server = app.server
app.config['suppress_callback_exceptions'] = False

app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for dropdown items
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.Img(
                            className="logo", src=app.get_asset_url("dash.png"),
                        ),
                        html.H2("PCA - Annual Loads"),
                        html.P(
                            "Select a player from the dropdown below"
                        ),
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                dcc.Dropdown(
                                    id="player-dropdown",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in ['Player1', 'Player2', 'Player3', 'Player4']
                                    ],
                                    placeholder="Select a player",
                                )
                            ],
                        ),
                    ]
                ),
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id="pca-details",
                                 #figure=pca.plot_graphs()
                                 ),
                        dcc.Graph(id="pca-annual",
                                 # figure=pca.plot_annual()
                                 )
                    ],
                ),
            ],
        ),
    ],
)

@app.callback(
    Output('pca-details', 'figure'),
    [Input('player-dropdown', 'value')])
def update_figure(player_name):
    pca= PCA_class('whole_period_gps.csv', 'wellness_data.csv')
    pca.create_EWM_(player_name)
    pca.scale_fit_transform_()
    return pca.plot_graphs()

@app.callback(
    Output('pca-annual', 'figure'),
    [Input('player-dropdown', 'value')])
def update_figure(player_name):
    pca = PCA_class('whole_period_gps.csv', 'wellness_data.csv')
    pca.create_EWM_(player_name)
    pca.scale_fit_transform_()
    fig =pca.plot_annual()
    return fig



if __name__ == '__main__':
    app.run_server(debug=True)
