import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.pipeline import make_pipeline

data = pd.read_csv("data-files/merged_data_imputed.csv")
print(data.head(20))

# Calculate Correlation Matrix
columns = ['Life Expectancy At Birth', 'Life Ladder', 'Social Support', 
                   'Freedom To Make Life Choices', 'Generosity', 'Perceptions Of Corruption', 
                   'Positive Affect', 'Negative Affect']

# somewhat of a linear relationship for Life Expectancy vs Life Ladder and social support but not for other variables

corr_matrix = data[columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.savefig("Correlation_Heatmap.png")

def createModel(column_to_predict, visualization_path):
    data_cleaned = data.dropna(subset=[column_to_predict])
    X = data_cleaned.drop([column_to_predict, 'Year', 'Country Name', 'Continent'], axis=1)
    y = data_cleaned[column_to_predict]

    # check if assumptions for multiple linear regression is satisified
    # not satisified
    for column in X.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[column], y)
        plt.xlabel(column)
        plt.ylabel(column_to_predict)
        plt.title(f'{column_to_predict} vs {column}')
        visualization_dir = f"visualizations/{visualization_path}_vs_feature"
        os.makedirs(visualization_dir, exist_ok=True)
        
        plt.savefig(f"{visualization_dir}/Life_vs_{column}")

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # dont need to scale features for random forest but potentially need to if switching the model to something else
    model = make_pipeline(
        SimpleImputer(strategy='mean'),
        MinMaxScaler(),
        RandomForestRegressor(n_estimators=100, random_state=42)
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    mse = mean_squared_error(y_valid, y_pred)

    r2 = r2_score(y_valid, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")

    return model, X

# feature_importances_
# the higher the more important the feature
# computed as the normalized total reduction of the criterion brought by that feature
# also called Gini importance
# source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


print("\nLife Expectancy Feature Ranking")
rf_model, X = createModel('Life Expectancy At Birth', 'life_expect')
rf_model = rf_model.named_steps['randomforestregressor']

importances = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
print(importances)

# life ladder importance is high, could suggest that the country's overall life ladder score is strongly associated with life expectancy

# feature importance for life expectancy

# feature ranking for life expectancy
'''
                        feature  importance
0                   Life Ladder    0.627672
5               Positive Affect    0.095920
1                Social Support    0.068563
3                    Generosity    0.067294
2  Freedom To Make Life Choices    0.048751
4     Perceptions Of Corruption    0.046255
6               Negative Affect    0.045546
'''

print("\nLife Ladder Feature Ranking")
rf_model, X = createModel('Life Ladder', 'life_ladder')
rf_model = rf_model.named_steps['randomforestregressor']

importances = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
print(importances)

# feature ranking for life ladder
'''
0      Life Expectancy At Birth    0.505781
1                Social Support    0.270890
5               Positive Affect    0.086501
4     Perceptions Of Corruption    0.039052
6               Negative Affect    0.035757
2  Freedom To Make Life Choices    0.031853
3                    Generosity    0.030166
'''

'''
for life expectancy:
    - Life Ladder and Life Expectancy are pretty correlated, (linearly related, see visualization)
        - 0.77 correlation coefficient
    - Social Support and Life Expectancy are pretty correlated
        - 0.65 correlation coefficient
'''

'''
for life ladder:
    - Life Expectancy and Life Ladder are pretty correlated, (linearly related, see visualization)
        - 0.77 correlation coefficient
    - Social Support and Life Ladder are pretty correlated
        - 0.74 correlation coefficient
    - Freedom to make life choices and Life Ladder are somewhat correlated
        - 0.57 correlation coefficient
'''


