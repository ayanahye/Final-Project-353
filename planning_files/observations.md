## Data Imputation 
After imputing and predicting values using linear regression, I noticed some parts of the column for corruption of government are still blank
- Notably, there is not data on how corrupt people think the government is for these countries:
    - China
    - Cuba
    - Maldives
    - Oman
    - Turkmenistan

## More Rank Data
There is a future problem that the behavior of Series.idxmin or Series.idxmax with all NA values will be deprecated and in future versions will raise a ValueError.
- Because some countries are missing all their data in one feature the whole feature data is null which is a problem. One solution is to pass .idxmax(skipna=True) however this will not allow us to put a NaN value for that year which is needed.

## Changes in continents data:
- Burkina -> Burkina Faso
- Added Eswatini 
- Changed Burma -> Myanmar
- Added North Macedonia
- Change column name country to country name

## Stats Test results:
* Test: The median Life Expectancy At Birth differs significantly across countries with different levels of Perceptions of Corruption. (Kruskal Wallis test)
    * Result: P-value is really small (9.217059352733059e-06) so we can successfully reject H0 and conclude that there is a significant difference in the median life expectancy at birth across the different levels of corruption.

* Test: Do countries with high/medium/low perceived social support have different life ladder ratings on average? (Chi Square Test)
    * Result: Since p-value = 9.0337804366599e-21 < 0.05, there's some common cause that affects both Life Ladder score (happiness level) and Social Support score. Shows that countries with different levels of perceived social support (high, medium, low) have different life ladder ratings on average. However, can't determine the direction of the association.

* Test: Does the categorical variable continent affect the categorical response variable of whether or not a country is above or below the median global life expectancy? (72.81 years)
    * Result: P-value: 2.3988294461371302e-12 < 0.05 => reject H0. Conclude that the categorical variable continent affects the categorical response variable of whether or not a country is above or below the median global life expectancy.

* Test: Does the categorical variable continent affect the categorical response variable of whether or not a country is above or below the median global happiness index (5.54 points)?
    * Result: P-value: 5.630043483949672e-13 < 0.05 => reject H0. Conclude that the categorical variable continent affects the categorical response variable of whether or not a country is above or below the median happiness index.