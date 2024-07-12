import pandas as pd
from scipy import stats
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("merged_data_imputed.csv")
# differences in life expectancy across continents (17 observations in each group)


def plot_boxplot(data, dependent_variable):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Continent", y=dependent_variable, data=data)
    plt.title(f'Boxplot of {dependent_variable} by Continent')
    plt.xticks(rotation=50)
    plt.savefig(f'visualizations/regional/boxplots/boxploy_regional_{dependent_variable}_stat_test.png')


def perform_regional_test(data, dependent_variable):
    data_grouped = data.groupby(["Continent", "Year"]).mean(numeric_only=True).reset_index()

    data_relevant = data_grouped[["Continent", "Year", dependent_variable]]

    print(f"Results for {dependent_variable}:\n")
    #print(data_grouped.to_string())
    #print(data_relevant.to_string())

    for continent in data_relevant["Continent"].unique():
        p = stats.normaltest(data_relevant[data_relevant["Continent"] == continent][dependent_variable]).pvalue
        print(f'p value for {continent} is: {p}')

    grouped_data_values = [data_relevant[data_relevant["Continent"] == continent][dependent_variable]
                        for continent in data_relevant["Continent"].unique()]

    levene_p = stats.levene(*grouped_data_values).pvalue
    print(f"Levene's test p-value: {levene_p}")

    # so groups are normal but variances are not equal, proceed with Welch's ANOVA test
    # seems like a good fit according to: https://scales.arabpsychology.com/stats/how-to-perform-welchs-anova-in-python-step-by-step/

    # https://pingouin-stats.org/build/html/generated/pingouin.welch_anova.html
    aov = pg.welch_anova(dv=dependent_variable, between="Continent", data=data_relevant)
    print(aov)

    # p value is very small: 5.150693e-38 so there is significant difference in life expectancy across continents
    # we reject H0 that all continent life expctenacy means are equal
    print(f"P value for Welch's ANOVA {aov['p-unc']}")

    # post hoc can do Games-Howell
    # source: https://statisticsbyjim.com/anova/welchs-anova-compared-to-classic-one-way-anova/

    # most comparisons have p < 0.05 so there is statistically significant differences. 
    # more analysis needed
    games_howell = pg.pairwise_gameshowell(dv=dependent_variable, between="Continent", data=data_relevant)
    print(games_howell)
    print("\n" + "="*50 + "\n")



perform_regional_test(data, "Life Expectancy At Birth")

'''
                A              B    mean(A)    mean(B)       diff        se          T         df          pval     hedges
0          Africa           Asia  60.840578  72.969444 -12.128866  0.522691 -23.204659  25.806769  4.884981e-15  -7.771122
1          Africa         Europe  60.840578  77.968077 -17.127498  0.496000 -34.531227  22.402267  0.000000e+00 -11.564331
2          Africa  North America  60.840578  73.527529 -12.686950  0.487248 -26.037976  21.184541  2.220446e-16  -8.719985
3          Africa        Oceania  60.840578  81.817978 -20.977400  0.483628 -43.375070  20.671649  3.774758e-15 -14.526089
4          Africa  South America  60.840578  74.464209 -13.623631  0.503566 -27.054326  23.421441  1.887379e-15  -9.060355
5            Asia         Europe  72.969444  77.968077  -4.998633  0.334929 -14.924447  30.224380  2.675637e-14  -4.998121
6            Asia  North America  72.969444  73.527529  -0.558084  0.321826  -1.734119  28.585589  5.217359e-01  -0.580748
7            Asia        Oceania  72.969444  81.817978  -8.848534  0.316318 -27.973501  27.721260  4.662937e-15  -9.368182
8            Asia  South America  72.969444  74.464209  -1.494765  0.346034  -4.319702  31.162691  1.884032e-03  -1.446646
9          Europe  North America  77.968077  73.527529   4.440548  0.276373  16.067206  31.598891  2.109424e-15   5.380825
10         Europe        Oceania  77.968077  81.817978  -3.849902  0.269940 -14.262046  31.138518  4.629630e-14  -4.776286
11         Europe  South America  77.968077  74.464209   3.503868  0.304221  11.517527  31.787776  1.033429e-11   3.857161
12  North America        Oceania  73.527529  81.817978  -8.290450  0.253500 -32.703994  31.904575  0.000000e+00 -10.952400
13  North America  South America  73.527529  74.464209  -0.936681  0.289732  -3.232926  30.855402  3.158396e-02  -1.082690
14        Oceania  South America  81.817978  74.464209   7.353769  0.283602  25.929921  30.191956  6.661338e-16   8.683798
'''

# note for Oceania the normaltest does not pass (p=0.02) however we proceed anyway
    # check graph 
perform_regional_test(data, "Life Ladder")

'''
                A              B   mean(A)   mean(B)      diff        se          T         df          pval     hedges
0          Africa           Asia  4.360442  5.390902 -1.030460  0.025521 -40.377679  26.455928  0.000000e+00 -13.522278
1          Africa         Europe  4.360442  6.112498 -1.752056  0.047512 -36.876147  23.957323  0.000000e+00 -12.349633
2          Africa  North America  4.360442  6.110986 -1.750545  0.035196 -49.736670  30.344673  3.108624e-15 -16.656556
3          Africa        Oceania  4.360442  7.269824 -2.909382  0.032498 -89.524394  31.676761  5.773160e-15 -29.981261
4          Africa  South America  4.360442  6.035752 -1.675310  0.060323 -27.772362  20.695356  0.000000e+00  -9.300822
5            Asia         Europe  5.390902  6.112498 -0.721596  0.044263 -16.302381  19.138848  1.547129e-11  -5.459584
6            Asia  North America  5.390902  6.110986 -0.720084  0.030670 -23.478724  23.020338  4.440892e-15  -7.862904
7            Asia        Oceania  5.390902  7.269824 -1.878922  0.027532 -68.246201  24.897752  5.662137e-15 -22.855303
8            Asia  South America  5.390902  6.035752 -0.644850  0.057799 -11.156794  17.780220  2.499418e-08  -3.736353
9          Europe  North America  6.112498  6.110986  0.001511  0.050465   0.029950  27.586803  1.000000e+00   0.010030
10         Europe        Oceania  6.112498  7.269824 -1.157326  0.048622 -23.802668  25.433286  0.000000e+00  -7.971391
11         Europe  South America  6.112498  6.035752  0.076746  0.070333   1.091177  29.684881  8.809975e-01   0.365430
12  North America        Oceania  6.110986  7.269824 -1.158837  0.036681 -31.592682  31.420992  2.220446e-15 -10.580227
13  North America  South America  6.110986  6.035752  0.075235  0.062675   1.200391  23.301407  8.322142e-01   0.402005
14        Oceania  South America  7.269824  6.035752  1.234072  0.061201  20.164303  21.687959  1.909584e-14   6.752922
'''

columns = data.columns[data.columns.isin(['Year', 'Continent', 'Country Name']) == False]

# group by continent so we can access each group and plot it aganst the feature
grouped_data = data.groupby('Continent')

for column in columns:
    plot_boxplot(data, column)
    plt.figure(figsize=(10, 6))
    for continent, group in grouped_data:
        plt.bar(group['Continent'], group[column], alpha=0.8, label=continent)

    plt.xlabel('Continent')
    plt.ylabel(column)
    plt.title(f'Bar Plot of {column} by Continent')
    plt.legend()
    plt.savefig(f'visualizations/regional/barplots/{column}.png')
