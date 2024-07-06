import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('merged_data_imputed.csv', parse_dates=['Year'])

# Plot a feature for a country
def plot_country(country_name, feature_name):
    country_data = data[data['Country Name'] == country_name]
    plt.title(f'{feature_name} for {country_name}')
    plt.xlabel('Year')
    plt.ylabel(feature_name)
    plt.plot(country_data['Year'], country_data[feature_name], 'b.')

countries = data['Country Name'].unique()

data_columns = np.array(['Life Expectancy At Birth', 'Life Ladder',
       'Social Support', 'Freedom To Make Life Choices', 'Generosity',
       'Perceptions Of Corruption', 'Positive Affect', 'Negative Affect'])

# Plot all features of a country
def plot_country_features(country_name):
    n = data_columns.size
    plt.figure(figsize=(10, 45))
    for i, column in enumerate(data_columns, 1):
        plt.subplot(n, 1, i)
        plot_country(country_name, column)
    plt.savefig(f'visualizations/{country_name}_data.png')

# Plot 2 features against one another for a country
def plot_2_features(country_name, feature1_name, feature2_name):
    country_data = data[data['Country Name'] == country_name]
    plt.title(f'{feature1_name} against {feature2_name} for {country_name}')
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.plot(country_data[feature1_name], country_data[feature2_name], 'b.')
    plt.savefig(f'visualizations/{country_name}_2_features.png')

# Plot 2 countries side by side with a feature
def plot_2_countries(country_1, country_2, feature_name):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plot_country(country_1, feature_name)
    plt.title(country_1)

    plt.subplot(1, 2, 2)
    plot_country(country_2, feature_name)
    plt.title(country_2)

    plt.tight_layout()
    plt.savefig(f'visualizations/{country_1}-vs-{country_2}.png')

# Visualize the top 50 countries based on each feature
folder = 'ranks_high_overall'
from os import listdir
from os.path import join, isfile

path = folder
for i, filename in enumerate(listdir(path)):
    full_filename = join(path, filename) # get file name joint with the path
    data = pd.read_csv(full_filename) # get data for that file
    feature = data.columns[1]
    plt.figure(figsize=(10, 6)) # plot a bar graph
    plt.title(f'Top 50 countries based on {feature}')
    plt.xlabel('Country')
    plt.ylabel(f'{feature.title()} Score')
    plt.xticks(rotation=90)
    plt.bar(data['Country Name'], data[feature])
    plt.tight_layout()
    plt.savefig(f'high_ranks_visualizations/top_50_{feature.replace(" ", "_").lower()}_data.png')

# Visualize the bottom 50 countries based on each feature
folder = 'ranks_low_overall'
path = folder
for i, filename in enumerate(listdir(path)):
    full_filename = join(path, filename)
    data = pd.read_csv(full_filename)
    feature = data.columns[1]
    plt.figure(figsize=(10, 6))
    plt.title(f'Bottom 50 countries based on {feature}')
    plt.xlabel('Country')
    plt.ylabel(f'{feature.title()} Score')
    plt.xticks(rotation=90)
    plt.bar(data['Country Name'], data[feature])
    plt.tight_layout()
    plt.savefig(f'low_ranks_visualizations/bottom_50_{feature.replace(" ", "_").lower()}_data.png')