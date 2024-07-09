import pandas as pd
from scipy import stats
import pingouin as pg

data = pd.read_csv("merged_data_imputed.csv")
# differences in life expectancy across continents (17 observations in each group)

data_grouped = data.groupby(["Continent", "Year"]).mean(numeric_only=True).reset_index()

life_expect_continents = data_grouped[["Continent", "Year", "Life Expectancy At Birth"]]

#print(data_grouped.to_string())
print(life_expect_continents.to_string())

for continent in life_expect_continents["Continent"].unique():
    p = stats.normaltest(life_expect_continents[life_expect_continents["Continent"] == continent]["Life Expectancy At Birth"]).pvalue
    print(f'p value for {continent} is: {p}')

grouped_data_values = [life_expect_continents[life_expect_continents["Continent"] == continent]["Life Expectancy At Birth"]
                       for continent in life_expect_continents["Continent"].unique()]


africa = life_expect_continents[life_expect_continents["Continent"] == "Africa"]["Life Expectancy At Birth"]
asia = life_expect_continents[life_expect_continents["Continent"] == "Asia"]["Life Expectancy At Birth"]
europe = life_expect_continents[life_expect_continents["Continent"] == "Europe"]["Life Expectancy At Birth"]
n_america = life_expect_continents[life_expect_continents["Continent"] == "North America"]["Life Expectancy At Birth"]
oceania = life_expect_continents[life_expect_continents["Continent"] == "Oceania"]["Life Expectancy At Birth"]
s_america = life_expect_continents[life_expect_continents["Continent"] == "South America"]["Life Expectancy At Birth"]

p = stats.levene(africa, asia, europe, n_america, oceania, s_america).pvalue

print(f"levene test p {p}")

# so groups are normal but variances are not equal, proceed with Welch's ANOVA test
# seems like a good fit according to: https://scales.arabpsychology.com/stats/how-to-perform-welchs-anova-in-python-step-by-step/

# https://pingouin-stats.org/build/html/generated/pingouin.welch_anova.html
aov = pg.welch_anova(dv="Life Expectancy At Birth", between="Continent", data=life_expect_continents)
print(aov)

# p value is very small: 5.150693e-38 so there is significant difference in life expectancy across continents
# we reject H0 that all continent life expctenacy means are equal
print(f"P value for Welch's ANOVA {aov['p-unc']}")

# post hoc can do Games-Howell
# source: https://statisticsbyjim.com/anova/welchs-anova-compared-to-classic-one-way-anova/

# most comparisons have p < 0.05 so there is statistically significant differences. 
# more analysis needed
games_howell = pg.pairwise_gameshowell(dv="Life Expectancy At Birth", between="Continent", data=life_expect_continents)
print(games_howell)

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
