# Data moderation analysis for life expectancy and life ladder

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
data = pd.read_csv("data-files/merged_data_imputed.csv")
'''
Moderation analysis is used to examine if the effect of an independent variable on the dependent variable is the same across different levels of another independent variable (moderator). In other words, it is used to examine whether the moderator will change the strength of the relationship between the independent and dependent variables.

Source: https://statsnotebook.io/blog/analysis/moderation_interaction_regression/
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


# RQ1: 
# Does life ladder (indp variable) affect life expectancy (dep variable) more for higher or lower social support rates

X = data[['Life Ladder', 'Social Support']]
X["Life_Ladder_Mul_Social_Suport"] = data['Life Ladder'] * data['Social Support']
y = data['Life Expectancy At Birth']

# add the intercept to the independent variable
X = sm.add_constant(X)

# ordinary least squares to estimate unknown params in the linear regression model
# linear regression since we wanna model the linear relationship between the indp , dep variable and the moderator
# 
model_rq1 = sm.OLS(y, X)
results_rq1 = model_rq1.fit()

print(results_rq1.summary())

# Residuals should follow normal dist

plt.hist(residuals, bins=20, density=True, alpha=0.6, color='g')
plt.title('Histogram of Residuals (RQ1)')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.show()

# p < 0.05 then our indp variable has some effect on the dept
# H0 means basically there is not effect

'''
                              OLS Regression Results
====================================================================================
Dep. Variable:     Life Expectancy At Birth   R-squared:                       0.606
Model:                                  OLS   Adj. R-squared:                  0.606
Method:                       Least Squares   F-statistic:                     1203.
Date:                      Fri, 12 Jul 2024   Prob (F-statistic):               0.00
Time:                              15:29:28   Log-Likelihood:                -7349.8
No. Observations:                      2346   AIC:                         1.471e+04
Df Residuals:                          2342   BIC:                         1.473e+04
Df Model:                                 3
Covariance Type:                  nonrobust
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
const                            52.4502      2.961     17.713      0.000      46.644      58.257
Life Ladder                       0.7855      0.693      1.134      0.257      -0.573       2.144
Social Support                   -7.6756      3.560     -2.156      0.031     -14.656      -0.695
Life_Ladder_Mul_Social_Suport     4.5695      0.773      5.908      0.000       3.053       6.086
==============================================================================
Omnibus:                      104.768   Durbin-Watson:                   0.252
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              152.459
Skew:                          -0.411   Prob(JB):                     7.84e-34
Kurtosis:                       3.939   Cond. No.                         301.
==============================================================================

These Results suggest:
    - Social Support has an impact on life expectancy 
    - Social support and life ladder together have a significant impact on life expectancy 
    - Life ladder alone does not have much impact (fail to reject)

In summary, our results suggests that while the direct effect of Life Ladder alone on Life Expectancy is not statistically significant, the interaction between Life Ladder and Social Support significantly influences Life Expectancy
'''

# RQ2:
# Does Social support (indp variable) affect life ladder (dep variable) more when there is more freedom to make life choices?

X = data[['Social Support', 'Freedom To Make Life Choices']]
X["Social_Support_Mul_Freedom"] = data['Social Support'] * data['Freedom To Make Life Choices']
y = data["Life Ladder"]

X = sm.add_constant(X)

model_rq2 = sm.OLS(y, X)
results_rq2 = model_rq2.fit()
print(results_rq2.summary())

# Residuals should follow normal dist

plt.hist(residuals, bins=20, density=True, alpha=0.6, color='g')
plt.title('Histogram of Residuals (RQ2)')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.show()

'''
                           OLS Regression Results
==============================================================================
Dep. Variable:            Life Ladder   R-squared:                       0.670
Model:                            OLS   Adj. R-squared:                  0.669
Method:                 Least Squares   F-statistic:                     1581.
Date:                Fri, 12 Jul 2024   Prob (F-statistic):               0.00
Time:                        15:41:47   Log-Likelihood:                -2408.4
No. Observations:                2346   AIC:                             4825.
Df Residuals:                    2342   BIC:                             4848.
Df Model:                           3
Covariance Type:            nonrobust
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
const                            5.4869      0.381     14.383      0.000       4.739       6.235
Social Support                  -2.6923      0.492     -5.469      0.000      -3.658      -1.727
Freedom To Make Life Choices    -6.5773      0.539    -12.195      0.000      -7.635      -5.520
Social_Support_Mul_Freedom      11.5288      0.672     17.143      0.000      10.210      12.848
==============================================================================
Omnibus:                       31.039   Durbin-Watson:                   0.433
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.106
Skew:                          -0.252   Prob(JB):                     6.47e-08
Kurtosis:                       3.291   Cond. No.                         122.
==============================================================================

Social Support and Freedom to make life choices together tend to lead to higher levels of reported life ladder

all variables are statistically significant.
'''

# Helping clarify OLS Summary Table with Source: https://www.youtube.com/watch?v=U7D1h5bbpcs
