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
* The median Life Expectancy At Birth differs significantly across countries with different levels of Perceptions of Corruption. (Kruskal Wallis test)
    * Results: P-value is really small (9.217059352733059e-06) so we can successfully reject H0 and conclude that there is a significant difference in the median life expectancy at birth across the different levels of corruption.

* Do countries with high/medium/low perceived social support have different life ladder ratings on average? (Chi Square Test)
    * Since p-value = 9.0337804366599e-21 < 0.05, there's some common cause that affects both Life Ladder score (happiness level) and Social Support score. Shows that countries with different levels of perceived social support (high, medium, low) have different life ladder ratings on average. However, can't determine the direction of the association.