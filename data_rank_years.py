import pandas as pd
import numpy as np

# get the data
data = pd.read_csv('data-files/merged_data_imputed.csv')
countries_by_continents = pd.read_csv('countries_by_continents.csv')

features = data.loc[:, (data.columns != 'Country Name') & (data.columns != 'Year') & (data.columns != 'Continent')].columns

# create max values for each country
data_max_grouped = data.groupby(['Country Name'])[features].max().reset_index()
data_max_grouped = pd.merge(data_max_grouped, countries_by_continents, on='Country Name')

data_max_grouped.to_csv("more_rank_data/maxes_each_country.csv", index=False)

# create min values for each country
data_min_grouped = data.groupby(['Country Name'])[features].min().reset_index()
data_min_grouped = pd.merge(data_min_grouped, countries_by_continents, on='Country Name')

data_min_grouped.to_csv("more_rank_data/mins_each_country.csv", index=False)

# create year where max occurs for each country for each feature
data_max_grouped_with_years = pd.DataFrame(columns=['Country Name'] + list(features))
for country in data['Country Name'].unique():
    country_data = data[data['Country Name'] == country]
    max_years = {'Country Name': country}

    for feature in features:
        idx_max = country_data[feature].idxmax()

        # china has no max for corruption, we set this year to na
        if (pd.isna(idx_max)):
            max_year = np.nan
        else: max_year = data.loc[idx_max]['Year']
        
        max_years[feature] = max_year

    data_max_grouped_with_years = pd.concat([data_max_grouped_with_years, pd.DataFrame([max_years])], ignore_index=True)

data_max_grouped_with_years = pd.merge(data_max_grouped_with_years, countries_by_continents, on='Country Name')
data_max_grouped_with_years.to_csv('more_rank_data/country_max_features_only_year.csv', index=False)

# create year where min occurs for each country for each feature
data_min_grouped_with_years = pd.DataFrame(columns=['Country Name'] + list(features))
for country in data['Country Name'].unique():
    country_data = data[data['Country Name'] == country]
    min_years = {'Country Name': country}

    for feature in features:
        idx_min = country_data[feature].idxmin()

        # china has no min for corruption, we set this year to na
        if (pd.isna(idx_min)):
            min_year = np.nan
        else: min_year = data.loc[idx_min]['Year']
        
        min_years[feature] = min_year

    data_min_grouped_with_years = pd.concat([data_min_grouped_with_years, pd.DataFrame([min_years])], ignore_index=True)

data_min_grouped_with_years = pd.merge(data_min_grouped_with_years, countries_by_continents, on='Country Name')
data_min_grouped_with_years.to_csv('more_rank_data/country_min_features_only_year.csv', index=False)

