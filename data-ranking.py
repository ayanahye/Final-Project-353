import pandas as pd
import numpy as np

data = pd.read_csv('merged_data_imputed.csv')

features = data.loc[:, (data.columns != 'Country Name') & (data.columns != 'Year') & (data.columns != 'Continent')].columns

# top 50 accounting for different years
top_50_rank = {}

for feature in features:
    top_50_rank[feature] = data.nlargest(50, feature)

for feature, top_values in top_50_rank.items():
    '''print(f"Top 50 highest values for {feature}:")
    print(top_values[['Country Name', 'Year', feature]])
    print('\n')'''

    filename = f'ranks/ranks_high_overall_including_years/top_50_for_{feature.replace(" ", "_").lower()}.csv'
    top_values[["Country Name", "Year", feature.title()]].to_csv(filename, index=False)


# top 50 all time
data_grouped = data.groupby('Country Name')[features].mean().reset_index()

for feature in features:
    data_grouped = data_grouped.sort_values(by=feature, ascending=False)
    top_50 = data_grouped.head(50)[['Country Name', feature]]

    filename = f'ranks/ranks_high_overall/top_50_for_{feature.replace(" ", "_").lower()}.csv'
    top_50.to_csv(filename, index=False)

# bottom 50 accounting for different years
bottom_50_rank = {}

for feature in features:
    bottom_50_rank[feature] = data.nsmallest(50, feature)

for feature, bottom_values in bottom_50_rank.items():
    '''print(f"Bottom 50 highest values for {feature}:")
    print(bottom_values[['Country Name', 'Year', feature]])
    print('\n')'''

    filename = f'ranks/ranks_low_overall_including_years/bottom_50_for_{feature.replace(" ", "_").lower()}.csv'
    bottom_values[["Country Name", "Year", feature.title()]].to_csv(filename, index=False)


# bottom 50 all time
data_grouped = data.groupby('Country Name')[features].mean().reset_index()

for feature in features:
    data_grouped = data_grouped.sort_values(by=feature, ascending=True)
    bottom_50 = data_grouped.head(50)[['Country Name', feature]]

    filename = f'ranks/ranks_low_overall/bottom_50_for_{feature.replace(" ", "_").lower()}.csv'
    bottom_50.to_csv(filename, index=False)

