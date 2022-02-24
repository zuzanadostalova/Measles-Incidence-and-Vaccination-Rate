# Import needed packages
import streamlit as st
from plotly.subplots import make_subplots
from urllib.request import urlopen
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy
import geojson
import pycountry #conda install -c conda-forge pycountry

# Add title and header
#page configuration
st.set_page_config(page_title="Measles & Immunization: the Last 40 Years", # page title, displayed on the window/tab bar
        		   page_icon="⚙️", # favicon: icon that shows on the window/tab bar (tip: you can use emojis)
                   layout="wide", # use full width of the page
                   menu_items={
                       'About': "Exploration of the infection, vaccination rates of measles across the world."
                   })

# Load data and add it to cache
@st.cache
def load_dataframe(path, rows_to_skip=0):
    df = pd.read_csv(path, skiprows=rows_to_skip)
    return df

@st.cache
def load_jsonfile(path):
    with open(path) as response:
        regions = json.load(response)
    return regions

df_imm_raw = load_dataframe(path="../data/children_imm_rates_worldbank.csv", rows_to_skip=4)
df_imm = deepcopy(df_imm_raw)

#Ansam's data
per_vacc = load_dataframe(path='../data/API.csv')
income_country = pd.read_csv('../data/Metadata_Country_API.csv', na_filter=False)
incidents_100k = pd.read_csv('../data/incidents per 100k.csv')
num_cases = pd.read_csv('../data/num of measles cases.csv')
per_vacc_all = pd.read_csv('../data/percentage of children vaccinated.csv')
vacc_year_country = pd.read_csv('../data/Measles vaccination coverage.csv')
cases_year_global = pd.read_csv('../data/Measles reported cases and incidence by year.csv', index_col=0)
first_vacc = pd.read_csv('../data/received first vaccine.csv', na_filter=False)
second_vacc = pd.read_csv('../data/received second vaccine.csv', na_filter=False)
with open('../data/countries.geojson') as f:
    countries = geojson.load(f)


#data editing
per_vacc = per_vacc.dropna(axis=1, how='all')
incidents_100k = incidents_100k.rename(columns={'VALUE': 'incidents_100k'})
num_cases = num_cases.rename(columns={'VALUE': 'cases_num'})
per_vacc_all = per_vacc_all.rename(columns={'VALUE': 'vaccination_per'})
vacc_year_country['percent'] = (vacc_year_country['DOSES']/vacc_year_country['TARGET_NUMBER'])*100
vacc_year_country['percent'] = np.where(vacc_year_country['percent']>100.0, 0.0, vacc_year_country['percent'])


#mergin 3 data bases into one
#not used because of a lot of missing data
cases_and_vacc = pd.merge(pd.merge(incidents_100k, num_cases, on=['YEAR', 'COUNTRY']), per_vacc_all, on=['YEAR', 'COUNTRY'])

#droping the NaN
cases_and_vacc = cases_and_vacc.dropna()

#convert abbrevations to full names
list_alpha_3 = [i.alpha_3 for i in list(pycountry.countries)]
def country_flag(df):
    if (len(df['COUNTRY'])==3 and df['COUNTRY'] in list_alpha_3):
        return pycountry.countries.get(alpha_3=df['COUNTRY']).name
cases_and_vacc['COUNTRY']=cases_and_vacc.apply(country_flag, axis = 1)


#regions = load_jsonfile("data/raw/stzh.adm_stadtkreise_a.json")


st.markdown("""
<style>
.text-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Measles & Immunization: the Last 40 Years")
st.header("Exploring the relationship between measles incidence and vaccination levels across the world")

st.subheader("The Measles Map: Disease Incidence and Vaccination Levels")

st.text("Ansam's maps here")

st.header(" ")
st.subheader("Child Measles Vaccination Levels from 1980 to 2020")

# Enable selection of countries for plot (Widgets: selectbox)
countries = sorted(pd.unique(df_imm['Country Name']))
country = st.selectbox("Choose a Country", countries)

# Process country data
country_df = df_imm[df_imm['Country Name'] == country].copy()
country_df.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 65'], inplace=True)
country_df.reset_index(drop=True, inplace=True)
country_df = country_df.T.rename(columns={0: 'imm_rate'})
df_subset = country_df.loc['1980':'2020', 'imm_rate']

# Plot vaccination levels per country
rate_years_fig = go.Figure(data=go.Scatter(x=df_subset.index.astype('int'),
                                           y=df_subset.values,
                                           line={"color": "royalblue", "width": 2},
                                           mode="lines+markers"
                                          )
                          )
rate_years_fig.update_xaxes(range=[1980,2020])
rate_years_fig.update_yaxes(range=[0,100])

# Update the layout
rate_years_fig.update_layout(
    xaxis={"title": {"text": "Year", "font": {"size": 12}}},
    yaxis={"title": {"text": "MMR Immunization Level [%]", "font": {"size": 12}}},
    title={'text': "Percent of Children 12-23 Months of Age Immunized Against Measles in "+ country, "font": {"size": 16}, "x":0.5}
)

st.plotly_chart(rate_years_fig)

st.header(" ")
st.subheader("Vaccination Rates and National Income")
st.text("Zuzana's maps here")