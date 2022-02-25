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
import pycountry #conda install -c conda-forge pycountry
from PIL import Image

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
df_imm_raw = load_dataframe(path="./measles/data/children_imm_rates_worldbank.csv", rows_to_skip=0)
df_imm = deepcopy(df_imm_raw)
#df_imm = pd.read_csv('../data/children_imm_rates_worldbank.csv', na_filter=False)
df_imm_all_raw = load_dataframe(path="./measles/data/Measles vaccination coverage.csv", rows_to_skip=0)
df_imm_all = deepcopy(df_imm_all_raw)
df_incidence_raw = load_dataframe(path="./measles/data/Measles reported cases and incidence by year (Incidence rate).csv", rows_to_skip=0)
df_incidence = deepcopy(df_incidence_raw)

#Ansam's data
per_vacc_raw = load_dataframe(path='./measles/data/API.csv', rows_to_skip=0)
per_vacc = deepcopy(per_vacc_raw)
incidents_100k_raw = load_dataframe(path='./measles/data/incidents per 100k.csv', rows_to_skip=0)
incidents_100k = deepcopy(incidents_100k_raw)
num_cases_raw = load_dataframe(path='./measles/data/num of measles cases.csv', rows_to_skip=0)
num_cases = deepcopy(num_cases_raw)
per_vacc_all_raw = load_dataframe(path='./measles/data/percentage of children vaccinated.csv', rows_to_skip=0)
per_vacc_all = deepcopy(per_vacc_all_raw)
vacc_year_country_raw = load_dataframe(path='./measles/data/Measles vaccination coverage.csv', rows_to_skip=0)
vacc_year_country = deepcopy(vacc_year_country_raw)
cases_year_global_raw = pd.read_csv('./measles/data/Measles reported cases and incidence by year.csv', index_col=0)
cases_year_global = deepcopy(cases_year_global_raw)

#Zuzana's data
df_immun_child_world_years_raw = load_dataframe(path="./measles/data/API_SH.IMM.MEAS_DS2_en_csv_v2_3692853.csv", rows_to_skip=0)
df_immun_child_world_years = deepcopy(df_immun_child_world_years_raw)
#df_immun_child_world_years.describe(include='all')
df_immun_child_world_years.dropna(axis=1, how='all', inplace=True)
df_immun_child_world_income_raw = load_dataframe(path="./measles/data/Metadata_Country_API_SH.IMM.MEAS_DS2_en_csv_v2_3692853.csv", rows_to_skip=0)
df_immun_child_world_income = deepcopy(df_immun_child_world_income_raw)
#df_immun_child_world_income.drop('Unnamed: 5', axis=1, inplace=True)

#data editing
per_vacc = per_vacc.dropna(axis=1, how='all')
incidents_100k = incidents_100k.rename(columns={'VALUE': 'incidents_100k'})
num_cases = num_cases.rename(columns={'VALUE': 'cases_num'})
per_vacc_all = per_vacc_all.rename(columns={'VALUE': 'vaccination_per'})
vacc_year_country['percent'] = (vacc_year_country['DOSES']/vacc_year_country['TARGET_NUMBER'])*100
vacc_year_country['percent'] = np.where(vacc_year_country['percent']>100.0, 0.0, vacc_year_country['percent'])
df_combined_immun_child_world_income = pd.merge(df_immun_child_world_years, df_immun_child_world_income, how="inner", on=["Country Code"])
#alL_tables = pd.merge(df_immun_child_world_years, df_immun_child_world_income, how="outer", on=["Country Code"])
#alL_tables[~alL_tables['Country Code'].isin(df_combined_immun_child_world_income["Country Code"].to_list())]
df_combined_immun_child_world_income['id'] = df_combined_immun_child_world_income.index
df_combined_immun_child_world_income.columns = [ f'value_{x}' if x.isdigit()  else x for x in df_combined_immun_child_world_income.columns]
long_format_df = pd.wide_to_long(df_combined_immun_child_world_income, stubnames = 'value_', i = 'id', j='year')
wk_df = long_format_df.reset_index().drop('id', axis=1)
final_vax_rate_income_df = wk_df.dropna(axis='rows')
check_verity_of_percentage = final_vax_rate_income_df[(final_vax_rate_income_df['value_']>100)]
final_vax_rate_income_df['IncomeGroupValue'] = final_vax_rate_income_df['IncomeGroup'].replace({"High income": 4, "Upper middle income": 3, "Lower middle income": 2, "Low income": 1})


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


mcv1 = vacc_year_country.loc[vacc_year_country['ANTIGEN'] == 'MCV1']
mcv2 = vacc_year_country.loc[vacc_year_country['ANTIGEN'] == 'MCV2']
mcv1 = mcv1.sort_values('YEAR')
mcv2 = mcv2.sort_values('YEAR')

#regions = load_jsonfile("data/raw/stzh.adm_stadtkreise_a.json")

st.header("Exploring the relationship between measles incidence and vaccination levels across the world")
st.subheader("The Measles Map: Disease Incidence and Vaccination Levels")

show_labels = st.radio(label='Choose type of Antigen vaccination:', options=['MCV1', 'MCV2'])
col1, col2= st.columns(2)

#first map
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

#second map
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

if show_labels == "MCV1":
    col2.plotly_chart(fig2)
elif show_labels == "MCV2":
    col2.plotly_chart(fig3)


st.header(" ")
st.subheader("Country Incidence of Measles vs Vaccination Rate")

# filter incidence df
year_inc_df = df_incidence.copy()
year_inc_df.drop(columns = ['Disease','Denominator'], inplace = True)
# melt incidence df
melted_inc_df = pd.melt(year_inc_df, id_vars=['Country / Region'], value_vars=[str(x) for x in list(range(1980, 2021))])
melted_inc_df.rename(columns = {'variable':'year','value':'INCIDENCE'}, inplace = True)
melted_inc_df.sort_values(by='Country / Region', inplace = True)
melted_inc_df.year = melted_inc_df.year.astype(int)
# filter immunization df (MCV1)
year_imm_1_df = df_imm_all[(df_imm_all['ANTIGEN']=='MCV1')&(df_imm_all['COVERAGE_CATEGORY']=='ADMIN')].copy()
year_imm_1_df = year_imm_1_df[['NAME','COVERAGE','YEAR']]
# merge
year_inc_imm_1_df = melted_inc_df.merge(year_imm_1_df, how='outer', left_on=['Country / Region','year'], right_on=['NAME','YEAR'])
year_inc_imm_1_df.drop(columns = ['NAME','YEAR'], inplace = True)
year_inc_imm_1_df.dropna(subset=['Country / Region'], inplace = True)
# process data
year_inc_imm_1_df.INCIDENCE = year_inc_imm_1_df.INCIDENCE.astype(str)
year_inc_imm_1_df.INCIDENCE = year_inc_imm_1_df.apply(lambda row: row.INCIDENCE.replace(',', ''), axis=1)
year_inc_imm_1_df.INCIDENCE = year_inc_imm_1_df.INCIDENCE.astype(float).sort_values()
year_inc_imm_1_df.year = year_inc_imm_1_df.year.astype(int)
year_inc_imm_1_df.year = year_inc_imm_1_df.year.sort_values()
# filter immunization df (MCV2)
year_imm_2_df = df_imm_all[(df_imm_all['ANTIGEN']=='MCV2')&(df_imm_all['COVERAGE_CATEGORY']=='ADMIN')].copy()
year_imm_2_df = year_imm_2_df[['NAME','COVERAGE','YEAR']]
# merge
year_inc_imm_2_df = melted_inc_df.merge(year_imm_2_df, how='outer', left_on=['Country / Region','year'], right_on=['NAME','YEAR'])
year_inc_imm_2_df.drop(columns = ['NAME','YEAR'], inplace = True)
year_inc_imm_2_df.dropna(subset=['Country / Region'], inplace = True)
# process data
year_inc_imm_2_df.INCIDENCE = year_inc_imm_2_df.INCIDENCE.astype(str)
year_inc_imm_2_df.INCIDENCE = year_inc_imm_2_df.apply(lambda row: row.INCIDENCE.replace(',', ''), axis=1)
year_inc_imm_2_df.INCIDENCE = year_inc_imm_2_df.INCIDENCE.astype(float).sort_values()
year_inc_imm_2_df.year = year_inc_imm_2_df.year.astype(int)
year_inc_imm_2_df.year = year_inc_imm_2_df.year.sort_values()

# Plot figure
left_column, right_column = st.columns(2)
# Enable selection of antigen type (MMR1 or MMR2) (Widgets: radio button)
plot_types = ["MMR (first dose)", "MMR (second dose)"]
plot_type = left_column.radio("Choose Vaccine Dose", plot_types)

# Enable selection of whether to show outliers
hide_outliers = right_column.checkbox("Hide outliers")
st.write("")

col3, col4 = st.columns(2)
# Make plot
year_inc_imm_1_df = year_inc_imm_1_df.sort_values('year')
if plot_type == "MMR (first dose)":
    inc_imm_scatter = px.scatter(year_inc_imm_1_df, x="COVERAGE", y="INCIDENCE", animation_frame="year",
                     hover_name="Country / Region", range_x=[0, 100])
    max_y = np.nanpercentile(year_inc_imm_1_df.INCIDENCE, 95)
else:
    inc_imm_scatter = px.scatter(year_inc_imm_2_df, x="COVERAGE", y="INCIDENCE", animation_frame="year",
                     hover_name="Country / Region", range_x=[0, 100])
    max_y = np.nanpercentile(year_inc_imm_2_df.INCIDENCE, 95)

# Enable selection of whether to show outliers
if hide_outliers:
    inc_imm_scatter.update_yaxes(range=[0, max_y])

# Update the layout
inc_imm_scatter.update_layout(
    xaxis={"title": {"text": "Immunization Level [%]", "font": {"size": 12}}},
    yaxis={"title": {"text": "Measles Incidence [per million]", "font": {"size": 12}}},
    title={'text': "Country Incidence of Measles and Vaccination Rate", "font": {"size": 16}, "x":0.5}
)

col3.plotly_chart(inc_imm_scatter)

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

col4.plotly_chart(fig4)

#adding resources about outliers
st.caption('Measles outbreak in Mongolia in 2016')
st.write("[link](https://www.who.int/mongolia/news/detail/04-05-2016-measles-outbreak-in-mongolia-faqs)")
st.caption('Measles outbreak in 1991')
st.write("[link](https://www.cdc.gov/mmwr/preview/mmwrhtml/00016101.htm)")
st.caption('Measles outbreak in 2019 in the region of Samoa and New Zealand')
st.write("[link](https://www.mfat.govt.nz/en/countries-and-regions/australia-and-pacific/niue/new-zealand-high-commission-to-niue/about-niue/)")
st.write("[link](https://en.wikipedia.org/wiki/2019_Samoa_measles_outbreak)")


st.subheader("Child Measles Vaccination Levels from 1980 to 2020")
# Enable selection of countries for plot (Widgets: selectbox)
countries = sorted(pd.unique(df_imm['Country Name']))
country = st.selectbox("Choose a Country", countries)

col6, col7, col8= st.columns(3)
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
rate_years_fig.update_layout(width=1000, height=600)
rate_years_fig.update_layout(
    xaxis={"title": {"text": "Year", "font": {"size": 12}}},
    yaxis={"title": {"text": "MMR Immunization Level [%]", "font": {"size": 12}}},
    title={'text': "Percent of Children 12-23 Months of Age Immunized Against Measles in "+ country, "font": {"size": 16}, "x":0.5}
)

col6.plotly_chart(rate_years_fig)

#add image
col7.write(" ")

col8.text(" ")
col8.text(" ")
col8.text(" ")
col8.text(" ")
col8.text(" ")
image = Image.open('/home/ansam/Documents/github/Measles-Group-Project/measles/data/5591-vaccine_vial_needle-732x549-thumbnail.jpg')
col8.image(image, use_column_width=True)

st.header(" ")
st.subheader("Overall Measles Vaccination Levels and Disease Incidence from 1980 to 2020")

# Enable selection of countries for plot (Widgets: selectbox)
countries = sorted(pd.unique(df_incidence['Country / Region']))
country = st.selectbox("Choose a Country", countries)

col8, col9= st.columns(2)

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

col8.plotly_chart(vac_inc_fig)

#MCv1 and 2 line
anti_year = vacc_year_country.groupby(['YEAR', 'ANTIGEN']).percent.mean().reset_index()
fig5 = px.line(anti_year, x='YEAR', y='percent', color='ANTIGEN', markers=True, title='Average coverage % for different vaccine antigens',
              labels={
                     "YEAR": "Year",
                     "percent": "Coverage %",
                     "ANTIGEN": "Antigens"
                 },
              )
col9.plotly_chart(fig5)


st.header(" ")
st.subheader("Measles vaccination rate based on income group")

fig_1_1 = px.scatter(final_vax_rate_income_df, y="value_", x="year", opacity=0.7, color="IncomeGroupValue", hover_name="Country Name")
fig_1_1.update_xaxes(dtick=1)
fig_1_1.update_yaxes(dtick=5)
fig_1_1.update_traces(marker={'size': 15})
fig_1_1.update_layout({
    'yaxis': {"title": "Vaccination rates (%)"},
    "xaxis": {"title": "Time evolution"},
    "title": f"Measles vaccination rates in different countries based on income group: 12-23 months old children"
    })
fig_1_1.update_coloraxes(colorbar={
    "title": "Income group from the highest income to the lowest",
    "thicknessmode": "pixels",
    "thickness": 50,
    },colorscale= "Inferno")
fig_1_1.update_layout(width=1600, height=600)

st.plotly_chart(fig_1_1)

st.header(" ")
st.subheader("Measles vaccination rates in different countries based on income group: 12-23 months old children")

final_vax_rate_income_df_sorted = final_vax_rate_income_df.sort_values(by="IncomeGroupValue", ascending=False)
fig_1_2 = px.bar(final_vax_rate_income_df_sorted, x='year', y="value_", color="IncomeGroup",
             hover_data=["Country Name"],
             title='Measles vaccination rates in different countries based on income group: 12-23 months old children')
fig_1_2.update_xaxes(tick0=0, dtick=1)
fig_1_2.update_layout({
    'yaxis': {"title": "Vaccination count"},
    "xaxis": {"title": "Time evolution"},
    "title": "Measles vaccination rates in different countries based on income group: 12-23 months old children"
    })
fig_1_2.update_layout(width=1600, height=600)

st.plotly_chart(fig_1_2)