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

