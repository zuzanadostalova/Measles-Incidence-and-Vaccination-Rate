# Import needed packages
import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
#-----------------------------------------------------------------------------------------------------------------------
from plotly.subplots import make_subplots
#-----------------------------------------------------------------------------------------------------------------------
from urllib.request import urlopen
import json
from copy import deepcopy

# Load data and add it to cache
@st.cache
def load_dataframe(path, rows_to_skip):
    df = pd.read_csv(path, skiprows=rows_to_skip)
    return df

@st.cache
def load_jsonfile(path):
    with open(path) as response:
        regions = json.load(response)
    return regions
#-----------------------------------------------------------------------------------------------------------------------
df_imm_raw = load_dataframe(path="data/raw/children_imm_rates_worldbank.csv", rows_to_skip=4)
df_imm = deepcopy(df_imm_raw)
df_imm_all_raw = load_dataframe(path="data/raw/Measles vaccination coverage.csv", rows_to_skip=0)
df_imm_all = deepcopy(df_imm_all_raw)
df_incidence_raw = load_dataframe(path="data/raw/Measles reported cases and incidence by year (Incidence rate).csv", rows_to_skip=0)
df_incidence = deepcopy(df_incidence_raw)
#-----------------------------------------------------------------------------------------------------------------------
#regions = load_jsonfile("data/raw/stzh.adm_stadtkreise_a.json")

# Helper functions
#st.set_page_config(layout="wide")

st.markdown("""
<style>
.text-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

# Add title and header

st.title("Measles & Immunization: the Last 40 Years")
st.header("Exploring the relationship between measles incidence and vaccination levels across the world")

st.subheader("The Measles Map: Disease Incidence and Vaccination Levels")

st.text("Ansam's maps here")
#-----------------------------------------------------------------------------------------------------------------------
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
rate_years_fig.update_xaxes(range=[1980, 2020])
rate_years_fig.update_yaxes(range=[0, 100])

# Update the layout
rate_years_fig.update_layout(
    xaxis={"title": {"text": "Year", "font": {"size": 12}}},
    yaxis={"title": {"text": "MMR Immunization Level [%]", "font": {"size": 12}}},
    title={'text': "Percent of Children 12-23 Months of Age Immunized Against Measles in "+ country, "font": {"size": 16}, "x":0.5}
)

st.plotly_chart(rate_years_fig)

st.header(" ")
st.subheader("Overall Measles Vaccination Levels and Disease Incidence from 1980 to 2020")

# Enable selection of countries for plot (Widgets: selectbox)
countries = sorted(pd.unique(df_incidence['Country / Region']))
country = st.selectbox("Choose a Country", countries)

# process data
country_inc_df = df_incidence[df_incidence['Country / Region'] == country].copy()
country_inc_df.drop(columns=['Country / Region', 'Disease', 'Denominator'], inplace=True)
country_inc_df.reset_index(drop=True, inplace=True)
country_inc_df = country_inc_df.T.rename(columns={0: 'incidence'})
country_inc_df = country_inc_df.sort_index()
country_inc_df.incidence = country_inc_df.incidence.astype(str)
country_inc_df.incidence = country_inc_df.apply(lambda row: row.incidence.replace(',', ''), axis=1)
country_inc_df.incidence = country_inc_df.incidence.astype(float)

country_imm_1_df = df_imm_all[(df_imm_all['NAME'] == country) & (df_imm_all['ANTIGEN'] == 'MCV1') & (
            df_imm_all['COVERAGE_CATEGORY'] == 'ADMIN')].copy()
country_imm_1_df = country_imm_1_df[['YEAR', 'COVERAGE']]
country_imm_1_df.set_index('YEAR', inplace=True)
country_imm_1_df = country_imm_1_df.sort_index()

country_imm_2_df = df_imm_all[(df_imm_all['NAME'] == country) & (df_imm_all['ANTIGEN'] == 'MCV2') & (
            df_imm_all['COVERAGE_CATEGORY'] == 'ADMIN')].copy()
country_imm_2_df = country_imm_2_df[['YEAR', 'COVERAGE']]
country_imm_2_df.set_index('YEAR', inplace=True)
country_imm_2_df = country_imm_2_df.sort_index()

# Create figure with secondary y-axis
vac_inc_fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
vac_inc_fig.add_trace(
    go.Scatter(x=country_imm_1_df.index.astype('int'),
               y=country_imm_1_df.COVERAGE,
               line={"color": "royalblue", "width": 2},
               mode="lines+markers", name = "Vaccination level <br> (1st MMR dose)"),
    secondary_y=False)

vac_inc_fig.add_trace(
    go.Scatter(x=country_imm_2_df.index.astype('int'),
               y=country_imm_2_df.COVERAGE,
               line={"color": "deepskyblue", "width": 2},
               mode="lines+markers", name = "Vaccination level <br> (2nd MMR dose)"),
    secondary_y=False)

vac_inc_fig.add_trace(
    go.Scatter(x=country_inc_df.index.astype('int'),
               y=country_inc_df.incidence,
               line={"color": "darkviolet", "width": 2},
               mode="lines+markers",
               name="Measles incidence"),
    secondary_y=True)

# Add figure title
vac_inc_fig.update_layout(title={'text': "Measles Immunization Level and Disease Incidence in "+ country, "font": {"size": 16}, "x":0.5})

# Set x-axis title and range
vac_inc_fig.update_xaxes(title_text="Year", range=[1980,2020])

# Set y-axes titles and ranges
vac_inc_fig.update_yaxes(title_text="Vaccination level [%]", secondary_y=False, range=[0,100])
vac_inc_fig.update_yaxes(title_text="Incidence", secondary_y=True)

st.plotly_chart(vac_inc_fig)
#-----------------------------------------------------------------------------------------------------------------------
st.header(" ")
st.subheader("Vaccination Rates and National Income")
st.text("Zuzana's maps here")