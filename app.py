# ========== (c) JP Hwang 2020-03-23  ==========

import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

df = pd.read_csv('srcdata/tidy_health_data.csv', index_col=0)

data_columns = {
    'hosp_beds': 'Hospital beds per 1000 people',
    'health_exp': 'Healthcare expenditure per capita (USD)',
    'old_population': 'Population 65 or above',
    'physicians': 'Physicians per 1000 people',
    'gdp_cap': 'GDP per capita (USD)',
}
data_labels = {**data_columns, 'continent': 'Continent', 'country': 'Country'}

country_types = {'pop': 'Most populous', 'high': 'Highest values', 'low': 'Lowest values'}

cont_order = ['Africa', 'Asia', 'North America', 'South America', 'Europe', 'Oceania']
cont_colours = {cont_order[i]: px.colors.qualitative.D3[i] for i in range(len(cont_order))}

font_size = 10


def make_beds_sim(max_days):

    x_vals = list(range(max_days+1))

    max_beds = round(max(df.tot_hosp_beds), -len(str(int(max(df.tot_hosp_beds))))+1) / 1000

    df_list = list()
    for n in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        #  = 2 ^ (1/C$2) * C3 = 2 ^ (1/C$2) * 2 ^ (1/C$2)
        temp_df = pd.DataFrame([dict(x=x, y=round((2 ** (2/n)) ** x, 0), n=str(n) + ' days') for x in x_vals if ((2 ** (2/n)) ** x) < max_beds*2])
        df_list.append(temp_df)
    growth_df = pd.concat(df_list, axis=0).reset_index(drop=True)

    fig = px.scatter(growth_df, x='x', y='y', color='n', log_y=True,
                     title='Hospital beds required over time at various case growth rates<BR>(horizontal lines - capacities)',
                     range_x=[min(x_vals), max(x_vals)+1], range_y=[1, max_beds*10],
                     color_discrete_sequence=px.colors.diverging.RdYlBu,
                     labels={'n': 'Case doubling rate',
                             'x': 'Days since 1st hospitalization', 'y': 'Number of hospital beds required'},
                     template='plotly_white')
    fig.update_layout(font=dict(size=font_size, color='DarkSlateGray'), width=800, height=500)
    fig.update_traces(mode='lines+markers')
    fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    return fig


def make_scatter_comp(in_df, var1, var2, x_log, y_log, names_list):

    overlays = in_df['country'].apply(lambda x: x if x in names_list else '').values

    fig = px.scatter(in_df, x=var1, y=var2, log_x=x_log, log_y=y_log,
                     text=overlays,
                     title=data_labels[var1] + ' vs ' + data_labels[var2],
                     labels=data_labels,
                     color='continent', size="population", size_max=25,
                     color_discrete_map=cont_colours,
                     category_orders={'continent': cont_order},
                     template='plotly_white', hover_name='country')

    fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
    fig.update_layout(font=dict(size=font_size, color='DarkSlateGray'), width=800, height=500)

    return fig


def make_bar(in_df, var1, x_log=False, title='', x_range=None,
             color_var='pop'):

    if title == '':
        title = data_labels[var1]
    if x_range is None:
        x_range = [min(in_df[var1]), max(in_df[var1])]

    if color_var == 'pop':
        colors = np.log10(in_df["population"])
    elif color_var == 'cont':
        colors = 'continent'

    fig = px.bar(in_df, x=var1, y='country', title=title, log_x=x_log, range_x=x_range,
                 template='plotly_white', hover_name='country', orientation='h',
                 color=colors,
                 color_continuous_scale=px.colors.sequential.YlGnBu,
                 color_discrete_sequence=px.colors.qualitative.Safe,
                 color_discrete_map=cont_colours,
                 category_orders={'continent': cont_order},
                 labels=data_labels, range_color=[7, max(np.log10(df["population"]))],
                 )
    fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
    fig.update_layout(width=700, height=450)
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Population", tickvals=[6, 7, 8, 9], ticktext=["1M", "10M", "100M", "1B"],
        ),
        font=dict(size=font_size, color='DarkSlateGray'),
        yaxis_categoryorder='total descending'
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)


    return fig


def add_refline(fig, x0, x1, y0, y1, text):

    loght = np.log10(y0)

    fig.add_shape(
        dict(
            type="line", x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(color='Gray', width=1),
            layer='above'
        ),
    )
    fig.add_annotation(
        go.layout.Annotation(
            x=1,
            y=loght + 0.2,
            showarrow=False,
            text=text,
            font=dict(color='Gray'),
            xanchor='left',
        ),
    )

    return fig


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

navbar = dbc.NavbarSimple(
    brand="Healthcare infrastructure data dashboard",
    brand_href="#",
    color="dark",
    dark=True,
)

# TODO - make Bootstrap font size smaller, esp for pulldowns
body = dbc.Container(children=[
    dbc.Row([
        dcc.Markdown(
            """
            ### Healthcare infrastructure data dashboard            
            
            This is a demo dashboard - built on global healthcare infrastructure capacity data,
            and comparing them to potential geometric growths in hospitalizations (like we have seen with COVID-19).  
            """
        )
    ]),
    dbc.Row([
        dcc.Markdown(
            """
            -----
            ##### How many days until the number of cases exceed the total number of available hospital beds?
            -----                        
            Choose countries and a total hospital bed availability to see estimated timeframes until  
            capacities are exceeded by geometric growth in hospitalisations. 
            
            (Typically, only a small fraction of hospital beds in a country are available at any one time.)        
            """
        )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Badge("Show available beds for:", color="info", className="mr-1"),
            dbc.FormGroup([
                dcc.Dropdown(
                    id='beds_sim_ref_country',
                    options=[{'label': k, 'value': k} for k in np.sort(df.country.unique())],
                    value=['China', 'United States', 'Singapore'],
                    multi=True,
                    style={'width': '100%'}
                )],
            )], md=6
        ),
        dbc.Col([
            dbc.Badge("Choose proportion of beds available", color="info", className="mr-1"),
            dcc.Dropdown(
                id='beds_sim_avail_perc',
                options=[{'label': str(k) + '%', 'value': k} for k in [5, 10, 15, 25, 50, 100]],
                value=10,
                style={'width': '100%'}
            )], md=6
        ),
    ]),
    dcc.Loading(
        id="loading-beds_sim_graph",
        children=[
            dcc.Graph(
                'beds_sim_graph',
                config={'displayModeBar': False}
            ),
        ],
        type="circle",
    ),
    dbc.Row([  # TODO - update this section with real values & model
        dcc.Markdown(
            """
            The graph shows that slowing growth rates over time is more impactful than increasing bed capacity
            in meeting increasing demand for hospitalisations. 
            In other words, the impact of **flattening the curve** cannot be understated.       
            """
        )
    ]),
    dbc.Row([
        dcc.Markdown(
            """
            
            -----
            ##### Explore healthcare metrics' correlations           
            -----
            """
        )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Badge("Select X-axis data", color="info", className="mr-1"),
            dcc.Dropdown(
                id='scatter_xvar',
                options=[{'label': v, 'value': k} for k, v in data_columns.items()],
                value='hosp_beds',
                style={'width': '100%'}
            )], md=6
        ),
        dbc.Col([
            dbc.Badge("Select Y-axis data", color="info", className="mr-1"),
            dcc.Dropdown(
                id='scatter_yvar',
                options=[{'label': v, 'value': k} for k, v in data_columns.items()],
                value='old_population',
                style={'width': '100%'}
            )], md=6
        )
    ]),
    dbc.Row([
        dbc.Col([dbc.FormGroup(
            [
                dbc.Checkbox(
                    id="scatter_x_log_radio", className="form-check-input", checked=True,
                ),
                dbc.Label(
                    "Log x-axis scale",
                    html_for="scatter_x_log_radio",
                    className="form-check-label",
                ),
            ],
            check=True,
        )], md=6),
        dbc.Col([dbc.FormGroup(
            [
                dbc.Checkbox(
                    id="scatter_y_log_radio", className="form-check-input", checked=True,
                ),
                dbc.Label(
                    "Log y-axis scale",
                    html_for="scatter_y_log_radio",
                    className="form-check-label",
                ),
            ],
            check=True,
        )], md=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Badge("(Cosmetic) Show text for:", color="secondary", className="mr-1"),
            dbc.FormGroup([
                dcc.Dropdown(
                    id='scatter_name_overlay',
                    options=[{'label': k, 'value': k} for k in np.sort(df.country.unique())],
                    value=['China'],
                    multi=True,
                    style={'width': '100%'}
                )],
            )], md=6
        ),
    ]),
    # dcc.Loading(
    #     id="loading-scatter_one",
    #     children=[
    #         dcc.Graph(
    #             'scatter_one',
    #             config={'displayModeBar': False}
    #         ),
    #     ],
    #     type="circle",
    # ),
    dcc.Graph(
        'scatter_one',
        config={'displayModeBar': False}
    ),    
    dbc.Row([
        dcc.Markdown(
            """
            -----
            ##### Explore the data            
            -----
            """
        )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Badge("Select bar graph data", color="info", className="mr-1"),
            dbc.FormGroup([
                dcc.Dropdown(
                    id='bar_xvar',
                    options=[{'label': v, 'value': k} for k, v in data_columns.items()],
                    value='hosp_beds',
                    style={'width': '100%'},
                ),
            ]
            )], md=6
        ),
        dbc.Col([
            dbc.FormGroup([
            dbc.Badge("Show countries like these:", color="info", className="mr-1"),
                dcc.Dropdown(
                    id='bar_country_type',
                    options=[{'label': v, 'value': k} for k, v in country_types.items()],
                    value='pop',
                    style={'width': '100%'}
                )],
            )], md=6
        ),
    ]),
    dbc.Row([
        dbc.Col([dbc.FormGroup(
            [
                dbc.Checkbox(
                    id="bar_x_log_radio", className="form-check-input", checked=True,
                ),
                dbc.Label(
                    "Log x-axis scale",
                    html_for="scatter_x_log_radio",
                    className="form-check-label",
                ),
            ],
            check=True,
        )], md=6),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Badge("Choose color variable", color="info", className="mr-1"),
            dbc.FormGroup([
                dcc.Dropdown(
                    id='bar_colour_variable',
                    options=[{'label': 'Population', 'value': 'pop'}, {'label': 'Continent', 'value': 'cont'}],
                    value='pop',
                    style={'width': '100%'}
                )],
            )], md=6
        ),
    ]),
    dcc.Loading(
        id="loading-bar_one",
        children=[
            dcc.Graph(
                'bar_one',
                config={'displayModeBar': False}
            ),
        ],
        type="circle",
    ),
    dbc.Row([
        dcc.Markdown(
            """
            -----
            ##### References & notes:
            -----
                        
            ###### Data Source:
            
            [The World Bank](https://data.worldbank.org)
            
            ###### Created by:
            
            JP Hwang  (find me on [twitter](https://twitter.com/_jphwang))
            
            ###### Articles:
            
            TBC            
            
            """
        )
    ]),
], style={'marginTop': 20})

app.layout = html.Div(children=[navbar, body])


@app.callback(
    Output('beds_sim_graph', 'figure'),
    [Input('beds_sim_avail_perc', 'value'), Input('beds_sim_ref_country', 'value')]
)
def update_beds_sim(beds_avail, ref_country_list):

    max_days = 90
    fig = make_beds_sim(max_days)

    for ref_country in ref_country_list:
        ref_beds = df[df.country == ref_country].tot_hosp_beds.values[0] / 1000 * beds_avail / 100
        fig = add_refline(fig, x0=1, x1=max_days, y0=ref_beds, y1=ref_beds, text=ref_country)

    return fig


@app.callback(
    Output('scatter_one', 'figure'),
    [Input('scatter_xvar', 'value'), Input('scatter_yvar', 'value'),
     Input('scatter_x_log_radio', 'checked'), Input('scatter_y_log_radio', 'checked'),
     Input('scatter_name_overlay', 'value')
     ]
)
def update_scatter(var1, var2, x_log, y_log, names):

    if type(names) == list:
        names_list = names
    else:
        names_list = [names]

    fig = make_scatter_comp(df, var1, var2, x_log, y_log, names_list)

    return fig


@app.callback(
    Output('bar_one', 'figure'),
    [Input('bar_xvar', 'value'), Input('bar_x_log_radio', 'checked'),
     Input('bar_country_type', 'value'),
     Input('bar_colour_variable', 'value')]
)
def update_bar(var1, x_log, country_type, color_var):

    n_countries = 20  # Todo - allow choice of n_countries with dynamic graph resizing
    if country_type == 'pop':
        temp_df = df.sort_values('population', ascending=False)[:n_countries].sort_values('population', ascending=True)
    elif country_type == 'high':
        temp_df = df.sort_values(var1, ascending=False)[:n_countries].sort_values(var1, ascending=True)
    elif country_type == 'low':
        temp_df = df.sort_values(var1, ascending=True)[:n_countries]
    else:
        logger.error(f'UNEXPECTED country_type VARIABLE RECEIVED ({country_type})')
        temp_df = df.sort_values('population', ascending=False)[:n_countries]

    title = data_labels[var1] + ' (' + country_types[country_type] + ')'

    if x_log:
        x_range = [min(df[var1]), max(df[var1])]
    else:
        x_range = [0, max(df[var1])]

    fig = make_bar(temp_df, var1, x_log, title, x_range, color_var)

    return fig


if __name__ == '__main__':
     app.run_server(debug=False)