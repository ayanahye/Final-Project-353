import pandas as pd
import numpy as np

data = pd.read_csv('merged_data_imputed.csv')

features = data.loc[:, (data.columns != 'Country Name') & (data.columns != 'Year')].columns

top_50_rank = {}

for feature in features:
    top_50_rank[feature] = data.nlargest(50, feature)

for feature, top_values in top_50_rank.items():
    print(f"Top 50 highest values for {feature}:")
    print(top_values[['Country Name', 'Year', feature]])
    print('\n')

    filename = f'ranks/top_50_for_{feature.replace(" ", "_").lower()}.csv'
    top_values[["Country Name", "Year", feature.title()]].to_csv(filename, index=False)