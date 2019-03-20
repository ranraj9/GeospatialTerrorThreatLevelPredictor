## Evolving from the basics

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import pandas as pd

import plotly
from plotly import graph_objs as go
from plotly.graph_objs import *
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)
server = app.server
app.title = 'THREAT MAP'

# API keys and datasets
plotly.tools.set_credentials_file(username='amber.sa97', api_key='oGPcE8EP76oYLaSfm1y5')
mapbox_access_token = 'pk.eyJ1IjoiYW1iZXItc2F4ZW5hIiwiYSI6ImNqdDk1d3p5NzAydXEzeW1kMTYyMG9xZmsifQ.OoExrgP7MJoOvCBduoNy1w'
df = pd.read_csv("dash.csv")
df1=df.drop(['Unnamed: 0','iyear','latitude','provstate','longitude','city'],axis=1)
df.drop(['Unnamed: 0'], axis = 1, inplace = True)

# Selecting only required columns

# Boostrap CSS.
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

k = []
for i in range(101):
    r = 2.55*i
    b = 2.55*(100-i)
    g = 0
    l = [i/100, "rbg({}, {}, {})".format(r, g, b)]
    k.append(l)

#  Layouts

layout_table = dict(
    autosize=True,
    height=500,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor='#fffcfc',
    paper_bgcolor='#fffcfc',
    legend=dict(font=dict(size=10), orientation='h'),
)
layout_table['font-size'] = '12'
layout_table['margin-top'] = '20'

layout_map = dict(
    autosize=True,
    height=500,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor='#fffcfc',
    paper_bgcolor='#fffcfc',
    legend=dict(font=dict(size=10), orientation='h'),
    title='THREAT LEVEL',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(
            lon=80,
            lat=28
        ),
        zoom=3,
    )
)

# functions
def gen_map(df):
    # groupby returns a dictionary mapping the values of the first field
    # 'classification' onto a list of record dictionaries with that
    # classification value.
    return {
        "data": [{
                "type": "scattermapbox",
                "lat": list(df['latitude']),
                "lon": list(df['longitude']),
                "hoverinfo": "text",
                "hovertext": [["city: {} <br>Threat: {} <br>provstate: {}".format(i,j,k)]
                                for i,j,k in zip(df['city'], df['Threat'],df['provstate'])],
                "mode": "markers",
                "name": list(df['city']),
                "marker": {
                    "colorscale" : k,
                    "color" : df['Threat'],
                    "size": 12,
                    "opacity": 0.4
                }
        }],
        "layout": layout_map
    }
# #
app.layout = html.Div(
    html.Div([
        # html.Div(
        #     [
        #         html.H1(children='Maps and Tables',
        #                 className='nine columns'),
        #         html.Img(
        #             src="https://www.fulcrumanalytics.com/wp-content/uploads/2017/12/cropped-site-logo-1.png",
        #             className='three columns',
        #             style={
        #                 'height': '16%',
        #                 'width': '16%',
        #                 'float': 'right',
        #                 'position': 'relative',
        #                 'padding-top': 12,
        #                 'padding-right': 0
        #             },
        #         ),
        #         html.Div(children='''
        #                 Dash Tutorial video 04: Working with tables and maps.
        #                 ''',
        #                 className='nine columns'
        #         )
        #     ], className="row"
        # ),

        # Selectors
        # html.Div(
        #     [
        #         html.Div(
        #             [
        #                 html.P('Choose Borroughs:'),
        #                 dcc.Checklist(
        #                         id = 'boroughs',
        #                         options=[
        #                             {'label': 'Manhattan', 'value': 'MN'},
        #                             {'label': 'Bronx', 'value': 'BX'},
        #                             {'label': 'Queens', 'value': 'QU'},
        #                             {'label': 'Brooklyn', 'value': 'BK'},
        #                             {'label': 'Staten Island', 'value': 'SI'}
        #                         ],
        #                         values=['MN', 'BX', "QU",  'BK', 'SI'],
        #                         labelStyle={'display': 'inline-block'}
        #                 ),
        #             ],
        #             className='six columns',
        #             style={'margin-top': '10'}
        #         ),
        #         html.Div(
        #             [
        #                 html.P('Threat:'),
        #                 dcc.Dropdown(
        #                     id='type',
        #                     options= [{'label': str(item),
        #                                           'value': str(item)}
        #                                          for item in set(df['Threat'])],
        #                     multi=True,
        #                     value=list(set(df['Threat']))
        #                 )
        #             ],
        #             className='six columns',
        #             style={'margin-top': '10'}
        #         )
        #     ],
        #     className='row'
        # ),

        # Map + table + Histogram
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='map-graph',
                                  animate=True,
                                  style={'margin-top': '20'})
                    ], className = "six columns"
                ),
                html.Div(
                    [
                        dt.DataTable(
                            rows=df.to_dict('records'),
                            columns=df1.columns,
                            row_selectable=True,
                            filterable=True,
                            sortable=True,
                            selected_row_indices=[],
                            id='datatable'),
                    ],
                    style = layout_table,
                    className="six columns"
                ),
                # html.Div([
                #         dcc.Graph(
                #             id='bar-graph'
                #         )
                #     ], className= 'twelve columns'
                #     ),
                html.Div(
                    [
                        html.P('Developed by Am I a joke to you - ', style = {'display': 'inline'})
                        # html.A('amyoshino@nyu.edu', href = 'mailto:amyoshino@nyu.edu')
                    ], className = "twelve columns",
                       style = {'fontSize': 18, 'padding-top': 20}
                )
            ], className="row"
        )
    ], className='ten columns offset-by-one'))

@app.callback(
    Output('map-graph', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices')])
def map_selection(rows, selected_row_indices):
    aux = pd.DataFrame(rows)
    # print("dataframe = {}".format(aux))
    # print("row = {}".format(selected_row_indices))
    # temp_df = aux[aux.provstate == aux.loc[selected_row_indices, 4]]
    c = 0
    temp_df = pd.DataFrame()
    for i in selected_row_indices:
        df = aux[aux.provstate == aux.loc[i, "provstate"]]
        if c == 0:
            temp_df = df
            c += 1
            continue
        else:
            temp_df = pd.concat([df, temp_df], axis=0)

    # print(aux.loc[selected_row_indices, "provstate"])
    if len(selected_row_indices) == 0:
        return gen_map(aux)
    return gen_map(temp_df)

app.config['suppress_callback_exceptions']=True
@app.callback(
    Output('datatable', 'rows'),
    [Input('type', 'value'),
     Input('provstate', 'values')])
def update_selected_row_indices(type, provstate):
    map_aux = df.copy()

    # Threat filter
    map_aux = map_aux[map_aux['Threat'].isin(type)]
    # Boroughs filter
    map_aux = map_aux[map_aux["provstate"].isin(provstate)]

    rows = map_aux.to_dict('records')
    return rows

# @app.callback(
#     Output('bar-graph', 'figure'),
#     [Input('datatable', 'rows'),
#      Input('datatable', 'selected_row_indices')])
# def update_figure(rows, selected_row_indices):
#     dff = pd.DataFrame(rows)
#
#     layout = go.Layout(
#         bargap=0.05,
#         bargroupgap=0,
#         barmode='group',
#         showlegend=False,
#         dragmode="select",
#         xaxis=dict(
#             showgrid=False,
#             nticks=50,
#             fixedrange=False
#         ),
#         yaxis=dict(
#             showticklabels=True,
#             showgrid=False,
#             fixedrange=False,
#             rangemode='nonnegative',
#             zeroline=False
#         )
#     )

    # data = Data([
    #      go.Bar(
    #          x=dff.groupby('Borough', as_index = False).count()['Borough'],
    #          y=dff.groupby('Borough', as_index = False).count()['Threat']
    #      )
    #  ])

    # return go.Figure(data=data, layout=layout)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)
