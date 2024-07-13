import pandas as pd
import numpy as np
from scipy import stats

# Chi-Square Test to determine:
# Do countries with high/medium/low perceived social support have different life ladder ratings on average?

# Only conduction the test for the year 2022 since that is the latest data we have, but for any year, the p-value is significant and always < 0.05 as expected

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

