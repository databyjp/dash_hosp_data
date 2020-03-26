# ========== (c) JP Hwang 2020-03-21  ==========

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

# ========== LOAD DATA ==========

# Data source: https://data.worldbank.org
gdp_cap_df = pd.read_csv('srcdata/gdp_per_cap.csv', skiprows=4)
physicians_df = pd.read_csv('srcdata/physicians.csv', skiprows=4)
hosp_beds_df = pd.read_csv('srcdata/hosp_beds.csv', skiprows=4)
pop_df = pd.read_csv('srcdata/population.csv', skiprows=4)
oldpop_df = pd.read_csv('srcdata/pop_65up.csv', skiprows=4)
health_exp_df = pd.read_csv('srcdata/health_exp_per_cap.csv', skiprows=4)
comm_health_workers_df = pd.read_csv('srcdata/comm_health_workers.csv', skiprows=4)

not_countries = list(pd.read_csv('srcdata/list_not_countries.csv')['Names'])

# ========== GET LATEST DATA FOR EACH COUNTRY ==========

countries = [c for c in gdp_cap_df['Country Name'].unique() if c in hosp_beds_df['Country Name'].unique()]

yrs_range = [str(i) for i in range(2010, 2020)]

data_tuples = [
    (gdp_cap_df, 'gdp_cap'),
    (physicians_df, 'physicians'),
    (hosp_beds_df, 'hosp_beds'),
    (pop_df, 'population'),
    (oldpop_df, 'old_population'),
    (health_exp_df, 'health_exp'),
    (comm_health_workers_df, 'health_workers'),
]

tidydata_list = list()
for country in countries:
    temp_dict = {'country': country}
    # temp_dict['gdp_per_cap'] = np.nan
    # temp_dict['year'] = 'N/A'
    for year in yrs_range:
        for temp_tuple in data_tuples:
            temp_df = temp_tuple[0]
            temp_title = temp_tuple[1]
            temp_val = temp_df[temp_df['Country Name'] == country][year].values[0]
            if not np.isnan(temp_val):
                temp_dict[temp_title] = temp_val
                temp_dict[temp_title + '_year'] = year
    tidydata_list.append(temp_dict)

df = pd.DataFrame(tidydata_list)

# fig = px.scatter_matrix(df, dimensions=[i[1] for i in data_tuples], hover_name='country')
# fig.show()

filt_df = df[
    df['gdp_cap'].notna() & df['hosp_beds'].notna() & df['population'].notna()
    & df['old_population'].notna() & df['physicians'].notna()]
filt_df = filt_df[-filt_df['country'].isin(not_countries)]

filt_df = filt_df.assign(ppl_per_hosp_bed=1000/filt_df.hosp_beds)
filt_df = filt_df.assign(ppl_per_physician=1000/filt_df.physicians)
filt_df = filt_df.assign(tot_hosp_beds=filt_df.hosp_beds * filt_df.population)

# Add continent data
cont_df = pd.read_csv('srcdata/Countries-Continents.csv')
filt_df['continent'] = ''
for i, row in filt_df.iterrows():
    temp_df = cont_df[cont_df.Country == row['country']]
    if len(temp_df) != 0:
        continent = temp_df.iloc[0].Continent
    else:
        temp_df = cont_df[cont_df.Country == row['country'].split(',')[0]]
        if len(temp_df) != 0:
            continent = temp_df.iloc[0].Continent
        else:
            continent = ''
    filt_df.loc[row.name, 'continent'] = continent
filt_df = filt_df[filt_df.continent != '']

filt_df.reset_index(drop=True, inplace=True)
filt_df.to_csv('srcdata/tidy_health_data.csv')

