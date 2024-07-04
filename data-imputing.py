import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

merged_data = pd.read_csv("merged_data.csv", na_values='')

columns_to_impute = ['Social Support', 'Freedom To Make Life Choices', 'Generosity', 'Perceptions Of Corruption', 'Positive Affect', 'Negative Affect']

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
                merged_data_imputed.loc[missing_data.index, column] = mean_value
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

                if r2_score > 0.6:
                    X_pred = missing_data[['Year']].values.reshape(-1, 1)
                    y_pred = model.predict(X_pred)
                    merged_data_imputed.loc[missing_data.index, column] = y_pred
                else:
                    lows += 1
                    print(f'Low R^2 score for {country} on {column}, keeping NaN values')
                    country_data_left_blank.append((country, column))

        elif country_data.empty:
            country_data_left_blank.append((country, column))

print(f"Total predictions: {total_preds}, Low R^2 scores: {lows}")

# 20% of predictions have 60% accuracy score or higher for linear regression on the training data
# 98% of predictions have 60% accuracy score or higher for random forests classifier on training data
print(f'This model is at least 60% accurate on the training data {1 - (lows/total_preds)} of the time')

unique_country_data_left_blank = set(country_data_left_blank)

df_country_data_left_blank = pd.DataFrame(country_data_left_blank, columns=['Country', 'Column'])
df_country_data_left_blank = df_country_data_left_blank.pivot(index="Country", values="Column", columns='Column')
df_country_data_left_blank = df_country_data_left_blank.sort_values(by="Country")

print(df_country_data_left_blank)
df_country_data_left_blank.to_csv("countries_with_missing_data.csv")

merged_data_imputed.to_csv("merged_data_imputed.csv", index=False)
