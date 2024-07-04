import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

merged_data = pd.read_csv("merged_data.csv", na_values='')

columns_to_impute = ['Social Support', 'Freedom To Make Life Choices', 'Generosity', 'Perceptions Of Corruption', 'Positive Affect', 'Negative Affect']

grouped = merged_data.groupby('Country Name')
print(grouped)

merged_data_imputed = merged_data.copy()

for column in columns_to_impute:
    for country, group in grouped:
        country_data = group[group[column].notna()]
        missing_data = group[group[column].isna()]

        if not country_data.empty and not missing_data.empty:
            X_train = country_data[['Year']].values.reshape(-1, 1)
            y_train = country_data[column].values

            model = LinearRegression()
            model.fit(X_train, y_train)

            X_pred = missing_data[['Year']].values.reshape(-1, 1)
            y_pred = model.predict(X_pred)

            merged_data_imputed.loc[missing_data.index, column] = y_pred 

merged_data_imputed.to_csv("merged_data_imputed.csv", index=False)
