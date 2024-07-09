import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.pipeline import make_pipeline

data = pd.read_csv("merged_data_imputed.csv")
print(data.head(20))

data_cleaned = data.dropna(subset=['Life Expectancy At Birth'])
X = data_cleaned.drop(['Life Expectancy At Birth', 'Year', 'Country Name', 'Continent'], axis=1)
y = data_cleaned['Life Expectancy At Birth']

# Calculate Correlation Matrix
columns = ['Life Expectancy At Birth', 'Life Ladder', 'Social Support', 
                   'Freedom To Make Life Choices', 'Generosity', 'Perceptions Of Corruption', 
                   'Positive Affect', 'Negative Affect']

corr_matrix = data[columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.savefig("Correlation_Heatmap.png")

# check if assumptions for multiple linear regression is satisified
    # not satisified
for column in X.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(X[column], y)
    plt.xlabel(column)
    plt.ylabel('Life Expectancy at Birth')
    plt.title(f'Life Expectancy vs {column}')
    plt.savefig(f"visualizations/life_expect_vs_feature/Life_vs_{column}")

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# somewhat of a linear relationship for Life Expectancy vs Life Ladder and social support but not for other variables

# dont need to scale features for random forest but potentially need to if switching the model to something else
rf_model = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler(),
    RandomForestRegressor(n_estimators=100, random_state=42)
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_valid)

mse = mean_squared_error(y_valid, y_pred)

r2 = r2_score(y_valid, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# feature_importances_
# the higher the more important the feature
# computed as the normalized total reduction of the criterion brought by that feature
# also called Gini importance
# source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

rf_model = rf_model.named_steps['randomforestregressor']

importances = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
importances = importances.sort_values('importance', ascending=False)
print(importances)

# life ladder importance is high, could suggest that the country's overall life ladder score is strongly associated with life expectancy





