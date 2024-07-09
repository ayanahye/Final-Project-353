import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

merged_data = pd.read_csv("merged_data.csv", na_values='')
continent_data = pd.read_csv("countries_by_continents.csv")

all_countries = merged_data['Country Name'].unique()
all_years = np.arange(2006, 2023)

complete_data = pd.DataFrame([(country, year) for country in all_countries for year in all_years],
                             columns=['Country Name', 'Year'])


# merge data so that all countries have rows from 2006-2022
merged_data = complete_data.merge(merged_data, on=['Country Name', 'Year'], how='left')

merged_data = pd.merge(merged_data, continent_data, on='Country Name')

columns_to_impute = ['Life Expectancy At Birth', 'Life Ladder', 'Social Support', 'Freedom To Make Life Choices', 'Generosity', 'Perceptions Of Corruption', 'Positive Affect', 'Negative Affect']

grouped = merged_data.groupby('Country Name')

merged_data_imputed = merged_data.copy()

country_data_left_blank = []

# instead of predicting the level of perceived corruption nationwide, we just take the mean given that theres more than 2 values.

lows = 0
total_preds = 0

for column in columns_to_impute:
    for country, group in grouped:
        country_data = group[group[column].notna()]
        missing_data = group[group[column].isna()]

        count_points = country_data[column].count()
        # if the data for that country is not empty and if there are some missing data fields then
        if (not country_data.empty and (not missing_data.empty)):
            if column == 'Perceptions Of Corruption' and count_points > 2:
                mean_value = country_data[column].mean()
                merged_data_imputed.loc[missing_data.index, column] = round(mean_value, 3)
                # print(f'Mean for {country} on {column} with count {count_points}: {mean_value:.2f}')
            elif count_points <= 2:
                country_data_left_blank.append((country, column))
            else:
                X_train = country_data[['Year']].values.reshape(-1, 1)
                y_train = country_data[column].values

                model = RandomForestRegressor(n_estimators=100)
                model.fit(X_train, y_train)

                r2_score = model.score(X_train, y_train)
                # print(f'R^2 score for {country} on {column}: {r2_score:.2f}')
                total_preds += 1

                if r2_score > 0.75:
                    X_pred = missing_data[['Year']].values.reshape(-1, 1)
                    y_pred = model.predict(X_pred)
                    merged_data_imputed.loc[missing_data.index, column] = y_pred.round(3)
                else:
                    lows += 1
                    print(f'Low R^2 score for {country} on {column}, imputing with mean value')
                    country_data_left_blank.append((country, column))
                    
                    # impute remaining missing values with the mean of the column
                    mean_value = country_data[column].mean()
                    merged_data_imputed.loc[missing_data.index, column] = round(mean_value, 3)
        elif country_data.empty:
            country_data_left_blank.append((country, column))

print(f"Total predictions: {total_preds}, Low R^2 scores: {lows}")

print(f'This model is at least 75% accurate on the training data {1 - (lows/total_preds)} of the time')

df_country_data_left_blank = pd.DataFrame(country_data_left_blank, columns=['Country', 'Column'])
df_country_data_left_blank['Present'] = np.where(df_country_data_left_blank['Column'].notna(), 'No Data', 'Data')
df_country_data_left_blank = df_country_data_left_blank.pivot(index="Country", values="Present", columns='Column')
df_country_data_left_blank = df_country_data_left_blank.sort_values(by="Country")
df_country_data_left_blank = df_country_data_left_blank.replace(np.nan, "Data")

df_country_data_left_blank.to_csv("countries_with_missing_data.csv")

# remove countries with minimal to no data
merged_data_imputed = merged_data_imputed[merged_data_imputed['Country Name'] != 'Cuba']
merged_data_imputed = merged_data_imputed[merged_data_imputed['Country Name'] != 'Guyana']
merged_data_imputed = merged_data_imputed[merged_data_imputed['Country Name'] != 'Maldives']
merged_data_imputed = merged_data_imputed[merged_data_imputed['Country Name'] != 'Oman']
merged_data_imputed = merged_data_imputed[merged_data_imputed['Country Name'] != 'Suriname']
merged_data_imputed = merged_data_imputed[merged_data_imputed['Country Name'] != 'Belize']

merged_data_imputed.to_csv("merged_data_imputed.csv", index=False)
