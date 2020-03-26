# ========== (c) JP Hwang 2020-03-22  ==========

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

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

df = pd.read_csv('srcdata/tidy_health_data.csv', index_col=0)

# ========== PLOT DATA ==========
data_labels = {'gdp_cap': 'GDP per capita (USD)',
               'old_population': 'Proportion of population 65 or above (%)',
               'physicians': 'Physicians per 1000 people',
               'hosp_beds': 'Hospital beds per 1000 people',
               'health_exp': 'Healthcare expenditure per capita (USD)',
               'continent': 'Continent',
               'country': 'Country'}

# ========== Healthcare expenditure ==========

# Bar chart - Healthcare expenditure per capita
largest_countries = df.sort_values('population', ascending=False).country[:20].to_list()
temp_df = df[df.country.isin(largest_countries)].sort_values('health_exp')

fig = px.bar(temp_df, x='health_exp', y='country', title='Large countries - Healthcare expenditure per capita',
             template='plotly_white', hover_name='country', orientation='h',
             color=np.log10(temp_df["population"]), color_continuous_scale=px.colors.sequential.Oranges,
             labels=data_labels, range_color=[7, max(np.log10(df["population"]))], range_x=[0, 10000]
             )
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.update_layout(width=900, height=500)
fig.update_layout(coloraxis_colorbar=dict(
    title="Population",
    tickvals=[6, 7, 8, 9],
    ticktext=["1M", "10M", "100M", "1B"],
))
fig.show()

fig = px.bar(temp_df, x='health_exp', y='country', title='Large countries - Healthcare expenditure per capita (log scale)', log_x=True,
             template='plotly_white', hover_name='country', orientation='h',
             color=np.log10(temp_df["population"]), color_continuous_scale=px.colors.sequential.Oranges,
             labels=data_labels, range_color=[7, max(np.log10(df["population"]))], range_x=[20, 10000]
             )
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.update_layout(width=900, height=500)
fig.update_layout(coloraxis_colorbar=dict(
    title="Population",
    tickvals=[6, 7, 8, 9],
    ticktext=["1M", "10M", "100M", "1B"],
))
fig.show()


top_health_exps = df.sort_values('health_exp', ascending=False).country[:20].to_list()
temp_df = df[df.country.isin(top_health_exps)].sort_values('health_exp')

fig = px.bar(temp_df, x='health_exp', y='country', title='Countries with highest healthcare expenditure per capita (log scale)', log_x=True,
             template='plotly_white', hover_name='country', orientation='h',
             color=np.log10(temp_df["population"]), color_continuous_scale=px.colors.sequential.Oranges,
             labels=data_labels, range_color=[7, max(np.log10(df["population"]))], range_x=[20, 10000]
             )
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.update_layout(width=900, height=500)
fig.update_layout(coloraxis_colorbar=dict(
    title="Population",
    tickvals=[6, 7, 8, 9],
    ticktext=["1M", "10M", "100M", "1B"],
))
fig.show()

# Scatter chart - GDP per capita vs health expenditure - ALL COUNTRIES
# titles = df['country'].apply(lambda x: x if x in largest_countries else '').values
fig = px.scatter(df, x='gdp_cap', y='health_exp', log_x=True, log_y=True,
                 title='Healthcare expenditure vs GDP - per capita',
                 labels=data_labels,
                 # text=titles,
                 color='continent',
                 size="population", size_max=30, color_discrete_sequence=px.colors.qualitative.D3,
                 template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
fig.update_layout(font=dict(size=9, color='DarkSlateGray'), width=900, height=600)
fig.show()


# ========== Hospital beds ==========

# Bar chart - hospital beds in each country
largest_countries = df.sort_values('population', ascending=False).country[:20].to_list()
top_hosp_beds = df.sort_values('hosp_beds', ascending=False).country[:10].to_list()
temp_countries = list(set(top_hosp_beds + largest_countries))

temp_df = df[df.country.isin(temp_countries)].sort_values('hosp_beds')

fig = px.bar(temp_df, x='hosp_beds', y='country', title='Hospital bed availability',
             template='plotly_white', hover_name='country', orientation='h',
             color=np.log10(temp_df["population"]), color_continuous_scale=px.colors.sequential.Oranges,
             labels=data_labels
             )
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.update_layout(width=900, height=600)
fig.update_layout(coloraxis_colorbar=dict(
    title="Population", tickvals=[6, 7, 8, 9], ticktext=["1M", "10M", "100M", "1B"],
))
fig.show()

# Scatter chart - hospital beds vs healthcare exp per capita
fig = px.scatter(temp_df, x='health_exp', y='hosp_beds', log_x=True, log_y=True, text='country',
                 title='Hospital bed availability vs Healthcare expenditure per capita (size: total population)',
                 labels=data_labels,
                 size="population", size_max=30, color_discrete_sequence=px.colors.qualitative.Safe,
                 template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
fig.update_layout(font=dict(size=9, color='DarkSlateGray'), width=900, height=600)
fig.show()

# Scatter chart - hospital beds vs size of 65+ population
fig = px.scatter(temp_df, x='old_population', y='hosp_beds', log_x=True, log_y=True, text='country',
                 title='Hospital bed availability vs Population 65 and above (size: total population)',
                 labels=data_labels,
                 size="population", size_max=30, color_discrete_sequence=px.colors.qualitative.Vivid,
                 template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
fig.update_layout(font=dict(size=9, color='DarkSlateGray'), width=900, height=600)
fig.show()

# Scatter chart - hospital beds vs size of 65+ population - ALL COUNTRIES
titles = df['country'].apply(lambda x: x if x in largest_countries else '').values
fig = px.scatter(df, x='old_population', y='hosp_beds', log_x=True, log_y=True, text=titles,
                 title='Hospital bed availability vs Population 65 and above (size: total population)',
                 labels=data_labels,
                 range_x=[2, 30], range_y=[0.2, 18],
                 size="population", size_max=30, color_discrete_sequence=px.colors.qualitative.D3,
                 template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
fig.update_layout(font=dict(size=9, color='DarkSlateGray'), width=900, height=600)
fig.show()

# Scatter chart - hospital beds vs size of 65+ population - ALL COUNTRIES & COLOR
fig = px.scatter(df, x='old_population', y='hosp_beds', log_x=True, log_y=True, text=titles,
                 title='Hospital bed availability vs Population 65 and above (size: total population)',
                 labels=data_labels,
                 range_x=[2, 30], range_y=[0.2, 18], color='continent',
                 size="population", size_max=30, color_discrete_sequence=px.colors.qualitative.D3,
                 template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
fig.update_layout(font=dict(size=9, color='DarkSlateGray'), width=900, height=600)
fig.show()

# Scatter chart - hospital beds vs size of 65+ population - ALL COUNTRIES, SUBPLOTS
fig = px.scatter(df, x='old_population', y='hosp_beds', log_x=True, log_y=True, text=titles,
                 title='Hospital bed availability vs Population 65 and above (size: total population)',
                 labels=data_labels,
                 range_x=[2, 30], range_y=[0.2, 18], color='continent', facet_row='continent',
                 size="population", size_max=30, color_discrete_sequence=px.colors.qualitative.D3,
                 template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
fig.update_layout(font=dict(size=9, color='DarkSlateGray'), width=900, height=2000)
fig.show()

# ========== Physicians ==========

# Countries with the most number of physicians per capita
fig = px.bar(df.sort_values('physicians', ascending=False)[:20][::-1],
             x='physicians', y='country', title='Countries with the most physicians per capita',
             template='plotly_white', hover_name='country', orientation='h',
             color=np.log10(largest_df["population"]), color_continuous_scale=px.colors.sequential.Oranges,
             labels=data_labels, range_x=[0, 8.5]
             )
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.update_layout(width=900, height=500)
fig.update_layout(coloraxis_colorbar=dict(
    title="Population", tickvals=[6, 7, 8, 9], ticktext=["1M", "10M", "100M", "1B"],
))
fig.show()

largest_df = df.sort_values('population', ascending=False)[:20].sort_values('physicians')

fig = px.bar(largest_df, x='physicians', y='country', title='Large countries - physicians per capita', log_x=False,
             template='plotly_white', hover_name='country', orientation='h',
             color=np.log10(largest_df["population"]), color_continuous_scale=px.colors.sequential.Oranges,
             labels=data_labels, range_x=[0, 8.5]
             )
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.update_layout(width=900, height=500)
fig.update_layout(coloraxis_colorbar=dict(
    title="Population",
    tickvals=[6, 7, 8, 9],
    ticktext=["1M", "10M", "100M", "1B"],
))
fig.show()

# ========== Physicians ==========

# Build DF based on some sort of modelling with -> x(date), y(tot), r(exponent)
x_vals = list(range(1, 101))
# growth_df = pd.DataFrame([dict(x=x, y=1.1 ** x, r=1.1) for x in x_vals])
df_list = list()
for r in [1.1, 1.11, 1.12]:
    temp_df = pd.DataFrame([dict(x=x, y=r ** x, r='Growth: ' + str(r)) for x in x_vals])
    df_list.append(temp_df)
growth_df = pd.concat(df_list, axis=0).reset_index(drop=True)

fig = px.scatter(growth_df, x='x', y='y', color='r', log_y=True)
fig.show()








# SCATTER chart - hospital beds vs physicians
fig = px.scatter(largest_df, x='hosp_beds', y='health_exp', log_x=True, log_y=True, text='country',
                 size="population", color='gdp_cap', template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
fig.update_layout(font=dict(size=9, color='DarkSlateGray'))
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.show()

# SCATTER chart - hospital beds vs physicians
fig = px.scatter(largest_df, x='gdp_cap', y='health_exp', log_x=True, log_y=True, text='country',
                 size="population", color='hosp_beds', template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
fig.update_layout(font=dict(size=9, color='DarkSlateGray'))
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.show()



fig = px.scatter(df, x='hosp_beds', y='gdp_cap', log_y=True, log_x=True,
                 size="population", color='health_exp', template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.show()





fig = px.scatter(df, x='ppl_per_hosp_bed', y='old_population', log_x=True,
                 size="population", color='physicians', template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.show()



fig = px.scatter(df, x='ppl_per_hosp_bed', y='physicians', log_x=True,
                 size="population", color='old_population', template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.show()


fig = px.scatter(df, x='ppl_per_hosp_bed', y='ppl_per_physician', log_x=True, log_y=True,
                 size="population", color='old_population', template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.show()


# With text
largest_countries = df.sort_values('population', ascending=False)['country'][:15].values
titles = df['country'].apply(lambda x: x if x in largest_countries else '').values
fig = px.scatter(df, x='ppl_per_hosp_bed', y='ppl_per_physician', log_x=True, log_y=True,
                 range_x=[50, 5000], range_y=[100, 100000], size_max=30,
                 size="population", color='old_population', color_continuous_scale=px.colors.sequential.Oranges,
                 text=titles,
                 template='plotly_white', hover_name='country')
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')), textposition='top center')
fig.update_layout(font=dict(size=9, color='DarkSlateGray'))
fig.show()


# ========== EXPLORE DATA ==========

fig = px.scatter_matrix(df, dimensions=['gdp_cap', 'hosp_beds', 'physicians', 'population', 'old_population'], hover_name='country')
fig.show()





