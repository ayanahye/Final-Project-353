import pandas as pd

happiness_data = pd.read_csv("happiness.csv")
life_expectancy_data = pd.read_csv("life-expectancy.csv", skiprows=3)

happiness_data['year'] = happiness_data['year'].astype(int)

happiness_data = happiness_data[happiness_data["year"] <= 2022]
happiness_data.dropna(subset=['Country name', 'year', 'Healthy life expectancy at birth'], inplace=True)

grouped_happiness_data = happiness_data.groupby(["Country name", "year"]).mean()
grouped_happiness_data.to_csv("cleaned_data/grouped_happiness_data.csv")

life_expectancy_data = pd.melt(life_expectancy_data, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='year', value_name='Value')
life_expectancy_data['year'] = life_expectancy_data['year'].astype(int)

life_expectancy_data.dropna(subset=['Country Name', 'year', 'Value'], inplace=True)

life_expectancy_data = life_expectancy_data[life_expectancy_data["year"] >= 2005]
life_expectancy_data.to_csv("cleaned_data/life_cleaned.csv")

merged_data = pd.merge(life_expectancy_data, grouped_happiness_data, left_on=['Country Name', 'year'], right_on=['Country name', 'year'])

print(merged_data)
merged_data.to_csv("merged_data.csv")
print("Finished data processing step!")