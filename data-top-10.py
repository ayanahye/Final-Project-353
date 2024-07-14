import pandas as pd
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from os.path import join, isfile

data = pd.read_csv("data-files/merged_data_imputed.csv")

def top_ranking_countries(year):

    # create a dataframe of the top 20 countries ranked with all features considered in a given year based on a score

    data_year = data[data['Year'] == year]

    weights = {
        'Life Expectancy At Birth': 0.25,
        'Life Ladder': 0.25,
        'Social Support': 0.1,
        'Freedom To Make Life Choices': 0.1,
        'Generosity': 0.1,
        'Perceptions Of Corruption': 0.05,
        'Positive Affect': 0.1,
        'Negative Affect': 0.05
    }

    data_year['Score'] = (
        data_year['Life Expectancy At Birth'] * weights['Life Expectancy At Birth'] +
        data_year['Life Ladder'] * weights['Life Ladder'] +
        data_year['Social Support'] * weights['Social Support'] + 
        data_year['Freedom To Make Life Choices'] * weights['Freedom To Make Life Choices'] + 
        data_year['Generosity'] * weights['Generosity'] + 
        data_year['Perceptions Of Corruption'] * weights['Perceptions Of Corruption'] + 
        data_year['Positive Affect'] * weights['Positive Affect'] + 
        data_year['Negative Affect'] * weights['Negative Affect']
    )

    score_min = data_year['Score'].min()
    score_max = data_year['Score'].max()
    # normalize score
    data_year['Score'] = (data_year['Score'] - score_min) / (score_max - score_min)

    ranked_by_score = data_year.sort_values(by='Score', ascending=False)
    ranked_by_score.head(20).to_csv(f'ranks/top-rank-countries/top-ranked-countries-in-{year}.csv', index=False)
    
for year in range(2006, 2023):
    top_ranking_countries(year)

folder = 'ranks/top-rank-countries'
path = folder
for i, filename in enumerate(listdir(path)):
    full_filename = join(path, filename)
    data = pd.read_csv(full_filename)
    year = data.loc[1, 'Year']
    plt.figure(figsize=(10, 6))
    plt.title(f'Top 20 Countries Based on Score in {year}')
    plt.xlabel('Countries')
    plt.ylabel(f'Score')
    plt.xticks(rotation=90)
    plt.bar(data['Country Name'], data['Score'])
    plt.tight_layout()
    plt.savefig(f'visualizations/top-ranked-countries/top-ranked-countries-in{year}.png')
    plt.close()
