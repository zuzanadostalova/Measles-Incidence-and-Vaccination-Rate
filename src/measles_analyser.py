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
st.set_page_config(page_title="Measles analysis", # page title, displayed on the window/tab bar
        		   page_icon="chart_with_upwards_trend", # favicon: icon that shows on the window/tab bar (tip: you can use emojis)
                   layout="wide", # use full width of the page
                   menu_items={
                       'About': "Exploration of the infection, vaccination rates of measles across the world."
                   })

st.markdown("<h1 style='text-align: center; color: red;'>Measles & Immunization: the Last 40 Years</h1>", unsafe_allow_html=True)

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

#Elena's data
df_imm_raw = load_dataframe(path="/home/ansam/Documents/github/Measles-Group-Project/measles/data/children_imm_rates_worldbank.csv", rows_to_skip=0)
df_imm = deepcopy(df_imm_raw)
#df_imm = pd.read_csv('../data/children_imm_rates_worldbank.csv', na_filter=False)
df_imm_all_raw = load_dataframe(path="/home/ansam/Documents/github/Measles-Group-Project/measles/data/Measles vaccination coverage.csv", rows_to_skip=0)
df_imm_all = deepcopy(df_imm_all_raw)
df_incidence_raw = load_dataframe(path="/home/ansam/Documents/github/Measles-Group-Project/measles/data/Measles reported cases and incidence by year (Incidence rate).csv", rows_to_skip=0)
df_incidence = deepcopy(df_incidence_raw)

#Ansam's data
per_vacc = pd.read_csv('/home/ansam/Documents/github/Measles-Group-Project/measles/data/API.csv')
income_country = pd.read_csv('/home/ansam/Documents/github/Measles-Group-Project/measles/data/API.csv', na_filter=False)
incidents_100k = pd.read_csv('/home/ansam/Documents/github/Measles-Group-Project/measles/data/incidents per 100k.csv')
num_cases = pd.read_csv('/home/ansam/Documents/github/Measles-Group-Project/measles/data/num of measles cases.csv')
per_vacc_all = pd.read_csv('/home/ansam/Documents/github/Measles-Group-Project/measles/data/percentage of children vaccinated.csv')
vacc_year_country = pd.read_csv('/home/ansam/Documents/github/Measles-Group-Project/measles/data/Measles vaccination coverage.csv')
cases_year_global = pd.read_csv('/home/ansam/Documents/github/Measles-Group-Project/measles/data/Measles reported cases and incidence by year.csv', index_col=0)
first_vacc = pd.read_csv('/home/ansam/Documents/github/Measles-Group-Project/measles/data/received first vaccine.csv', na_filter=False)
second_vacc = pd.read_csv('/home/ansam/Documents/github/Measles-Group-Project/measles/data/received second vaccine.csv', na_filter=False)
with open('/home/ansam/Documents/github/Measles-Group-Project/measles/data/countries.geojson') as f:
    countries_g = geojson.load(f)


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

st.header("Exploring the relationship between measles incidence and vaccination levels across the world")

st.subheader("The Measles Map: Disease Incidence and Vaccination Levels")

st.subheader("Number Of Cases around the world in different years")

# figures aligned in one row
col1, col2 = st.columns(2)

cases_and_vacc = cases_and_vacc.sort_values('YEAR')
cases_and_vacc['YEAR'] = cases_and_vacc['YEAR'].astype(int)
fig = px.choropleth(cases_and_vacc, locations=cases_and_vacc['COUNTRY'], locationmode='country names', color = cases_and_vacc['cases_num'],
              hover_name=cases_and_vacc['COUNTRY'], animation_frame=cases_and_vacc['YEAR'],
              color_continuous_scale=px.colors.sequential.RdBu, projection='natural earth', width=900, height=500, labels={"cases_num":"Number of cases"})
fig.update_layout(
    title_text='Number of cases per country through the years',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)
col1.plotly_chart(fig)

#another infections bar figure
cases_year_global2 =cases_year_global.T.sort_index()
fig4 = px.line(cases_year_global2, x=cases_year_global2.index, y='Measles', title="Number of global infections per year",
             labels={'index':'years', 'Measels': 'Numer of infection'})
fig4.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)')
fig4.add_annotation(x=19, y=873022,
            text="What happened?",
            showarrow=True,
            font=dict(
            family="Courier New, monospace",
            size=10,
            color="#ffffff"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8)

col2.plotly_chart(fig4)

st.header(" ")
st.subheader("Child Measles Vaccination Levels from 1980 to 2020")
colg, colt = st.columns(2)

# Enable selection of countries for plot (Widgets: selectbox)
countries = sorted(pd.unique(df_imm['Country Name']))
country = st.selectbox("Choose a Country", countries)

# Process country data
country_df = df_imm[df_imm['Country Name'] == country].copy()
country_df.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], inplace=True)
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

colg.plotly_chart(rate_years_fig)

if st.checkbox("Check the box if you are interested in the table"):
    st.subheader("This is the country's dataset:")
    colt.dataframe(data=df_subset)

#plots MCV1 and MCV1

mcv = st.selectbox('Choose type of Antigen vaccination:', ['MCV1', 'MCV2'])
col3, col4 = st.columns(2)

mcv1 = vacc_year_country.loc[vacc_year_country['ANTIGEN'] == 'MCV1']
mcv2 = vacc_year_country.loc[vacc_year_country['ANTIGEN'] == 'MCV2']
mcv1 = mcv1.sort_values('YEAR')
mcv2 = mcv2.sort_values('YEAR')

fig2 = px.choropleth(mcv1, locations=mcv1['NAME'], locationmode='country names', color = mcv1['percent'],
              hover_name=mcv1['NAME'], animation_frame=mcv1['YEAR'],
              color_continuous_scale=px.colors.sequential.RdBu, projection='natural earth', width=900, height=500, labels={"percent":"Coverage% of vaccinated"})
fig2.update_layout(
    title_text='Coverage in % of vaccinated with MCV1 per country through the years',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

fig3 = px.choropleth(mcv2, locations=mcv2['NAME'], locationmode='country names', color = mcv2['percent'],
              hover_name=mcv2['NAME'], animation_frame=mcv2['YEAR'],
              color_continuous_scale=px.colors.sequential.RdBu, projection='natural earth', width=900, height=500, labels={"percent":"Coverage % of vaccinated"})
fig3.update_layout(
    title_text='Coverage in % of vaccinated with MCV2 per country through the years',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

if mcv == "MCV1":
    col3.plotly_chart(fig2)
elif mcv == "MCV2":
    col3.plotly_chart(fig3)

anti_year = vacc_year_country.groupby(['YEAR', 'ANTIGEN']).percent.mean().reset_index()
fig5 = px.line(anti_year, x='YEAR', y='percent', color='ANTIGEN', markers=True, title='Average coverage % for different vaccine antigens',
              labels={
                     "YEAR": "Year",
                     "percent": "Coverage %",
                     "ANTIGEN": "Antigens"
                 },
              )
col4.plotly_chart(fig5)

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

st.header(" ")
st.subheader("Vaccination Rates and National Income")
st.text("Zuzana's maps here")