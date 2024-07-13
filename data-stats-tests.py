import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("merged_data_imputed.csv")
print(data.head(20))

neg_affect_life_exp = data[["Negative Affect", "Life Expectancy At Birth"]]
bins = [
    neg_affect_life_exp['Negative Affect'].min(),
    neg_affect_life_exp['Negative Affect'].quantile(1/3),
    neg_affect_life_exp['Negative Affect'].quantile(2/3),
    neg_affect_life_exp['Negative Affect'].max()
]

bins_life = [
    neg_affect_life_exp["Life Expectancy At Birth"].min(),
    neg_affect_life_exp["Life Expectancy At Birth"].quantile(1/3),
    neg_affect_life_exp["Life Expectancy At Birth"].quantile(2/3),
    neg_affect_life_exp["Life Expectancy At Birth"].max()
]

extents = ['Low', 'Medium', 'High']

# include min so include_lowest
neg_affect_life_exp['Negative Affect Extent'] = pd.cut(neg_affect_life_exp['Negative Affect'], bins=bins, labels=extents, include_lowest=True)
neg_affect_life_exp['Life Expectancy At Birth Extent'] = pd.cut(neg_affect_life_exp['Life Expectancy At Birth'], bins=bins_life, labels=extents, include_lowest=True)

print(neg_affect_life_exp)

# none of the extents are normal
contingency = pd.crosstab(neg_affect_life_exp['Negative Affect Extent'], neg_affect_life_exp['Life Expectancy At Birth Extent'])
print(contingency)

# p value is very small so there is an effect
chi2 = stats.chi2_contingency(contingency)
print(chi2.pvalue)

def create_bar_plots(data, column, extent, title, xlabel, ylabel, filename):
    plt.figure(figsize=(10, 6))
    low_data = data[data[extent] == 'Low'][column]
    medium_data = data[data[extent] == 'Medium'][column]
    high_data = data[data[extent] == 'High'][column]

    if (column == 'Negative Affect'):
        low_data = np.sqrt(low_data)
        medium_data = np.sqrt(medium_data)
        high_data = np.sqrt(high_data)

    print(f'P value for low_data {stats.normaltest(low_data).pvalue}')
    print(f'P value for medium_data {stats.normaltest(medium_data).pvalue}')
    print(f'P value for high_data {stats.normaltest(high_data).pvalue}')

    plt.hist(low_data, bins=10, alpha=0.7, label='Low')
    plt.hist(medium_data, bins=10, alpha=0.7, label='Medium')
    plt.hist(high_data, bins=10, alpha=0.7, label='High')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename) 


create_bar_plots(neg_affect_life_exp, 'Life Expectancy At Birth', 'Negative Affect Extent',
                         f'Life Expectancy for Negative Affect', 'Life Expectancy At Birth', 'Frequency',
                         f'life_expectancy_by_negative_affect.png')


create_bar_plots(neg_affect_life_exp, 'Negative Affect', 'Life Expectancy At Birth Extent',
                         f'Negative Affect for Life Expectancy', 'Negative Affect', 'Frequency',
                         f'negative_affect_by_life_expectancy.png')

print('\n')

# Chi-Square Test to determine:
# Do countries with high/medium/low perceived social support have different life ladder ratings on average?

# Only conducting the test for the year 2022 since that is the latest data we have, but for any year, the p-value is significant and always < 0.05 as expected

data = pd.read_csv('merged_data_imputed.csv')
data = data[data['Year'] == 2006]
data = data[['Country Name', 'Life Ladder', 'Social Support']]

# Categorize Social Support to either low, medium, or high
data['Category Social Support'] = data['Social Support'].apply(lambda x: 'Low SS' if x < 0.35 else ('Med SS' if x < 0.75 else 'High SS'))

data['Category Life Ladder'] = data['Life Ladder'].apply(lambda x: 'Low LL' if x <= 3 else ('Med LL' if x <= 5 else 'High LL'))

contingency = pd.crosstab(data['Category Life Ladder'], data['Category Social Support'])
print(f'Contingency Table: {contingency}')

chi2 = stats.chi2_contingency(contingency)
print(f'Chi Square P-value: {chi2.pvalue}\n')
print(f'Chi Square Expected Frequency: {chi2.expected_freq}\n')

# Since p-value = 9.0337804366599e-21 < 0.05, there's some common cause that affects both Life Ladder score (happiness level) and Social Support score. Or, the categories Life Ladder and Social Support are dependent on each other (one effects the other). Rejecting H0!
# Shows that countries with different levels of perceived social support (high, medium, low) have different life ladder ratings on average. However, the chi square test does not answer the direction of this association as well as which category impacts which.

# T test to determine:
# H0: The mean Life Expectancy At Birth is the same across countries with low, medium, and high Perceptions of Corruption.
# H1: The mean Life Expectancy At Birth differs significantly across countries with different levels of Perceptions of Corruption.

# Again, only testing for the most recent year: 2022

data = pd.read_csv('merged_data_imputed.csv')
data = data[data['Year'] == 2022]
data = data[['Country Name', 'Life Expectancy At Birth', 'Perceptions Of Corruption']]

low_corruption = data[data['Perceptions Of Corruption'] < 0.3].copy()
med_corruption = data[data['Perceptions Of Corruption'] < 0.65].copy()
high_corruption = data[data['Perceptions Of Corruption'] >= 0.65].copy()