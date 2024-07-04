import pandas as pd

happiness_data = pd.read_csv("happiness.csv", na_values=' ')
life_expectancy_data = pd.read_csv("life-expectancy.csv", skiprows=3, na_values=' ')

happiness_data['year'] = happiness_data['year'].astype(int)

happiness_data = happiness_data[happiness_data["year"] <= 2022]
happiness_data.dropna(subset=['Country name', 'year', 'Healthy life expectancy at birth'], inplace=True)

grouped_happiness_data = happiness_data.groupby(["Country name", "year"]).mean()
grouped_happiness_data.to_csv("cleaned_data/grouped_happiness_data.csv")

life_expectancy_data = pd.melt(life_expectancy_data, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='year', value_name='Life Expectancy at Birth')
life_expectancy_data['year'] = life_expectancy_data['year'].astype(int)

life_expectancy_data.dropna(subset=['Country Name', 'year', 'Life Expectancy at Birth'], inplace=True)

life_expectancy_data = life_expectancy_data[life_expectancy_data["year"] >= 2005]
life_expectancy_data.to_csv("cleaned_data/life_cleaned.csv")

merged_data = pd.merge(life_expectancy_data, grouped_happiness_data, left_on=['Country Name', 'year'], right_on=['Country name', 'year'])
merged_data = merged_data.drop_duplicates()
merged_data = merged_data.sort_values(by=['Country Name', 'year']).reset_index(drop=True)

merged_data = merged_data.drop(['Country Code', 'Indicator Name', 'Indicator Code', 'Log GDP per capita', 'Healthy life expectancy at birth'], axis=1)
merged_data.columns = [column.title() for column in merged_data.columns]

print(merged_data)
merged_data.to_csv("merged_data.csv", index=False)
print("Finished data processing step!")