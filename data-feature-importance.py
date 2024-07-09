import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

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


# Impute missing values in X for whatever model
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# check if assumptions for multiple linear regression is satisified
    # not satisified
for column in X.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(X[column], y)
    plt.xlabel(column)
    plt.ylabel('Life Expectancy at Birth')
    plt.title(f'Life Expectancy vs {column}')
    plt.savefig(f"visualizations/life_expect_vs_feature/Life_vs_{column}")

X_train, X_valid, y_train, y_valid = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# somewhat of a linear relationship for Life Expectancy vs Life Ladder and social support but not for other variables

