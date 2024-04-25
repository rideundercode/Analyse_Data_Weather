import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import kendalltau
import numpy as np
import statsmodels.api as sm
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose

# Replace 'your_api_key' with your actual API key
headers = {
    'x-rapidapi-key': "6ceb7b2424msh1982d3d9a2696a6p11cb49jsnce30609a6a6a",
    'x-rapidapi-host': "meteostat.p.rapidapi.com"
}
params = {
    'station': '07510',
    'start': '1990-01-01',
    'end': '2024-04-01'
}

# Perform the request to get the data
response = requests.get('https://meteostat.p.rapidapi.com/stations/monthly', headers=headers, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data['data'])
    df.to_csv('bordeaux_weather_data.csv', index=False)
else:
    print(f"Failed to fetch data: {response.status_code}")
    exit()

# Cleaning and preparing the data
df = pd.read_csv('bordeaux_weather_data.csv')
df['date'] = pd.to_datetime(df['date'])
mean_values = df.select_dtypes(include=[float]).mean()
df.fillna(mean_values, inplace=True)
df.drop(columns=['snow', 'tsun'], inplace=True)
df.drop_duplicates(inplace=True)
df.rename(columns={'tavg': 'temp_moy', 'prcp': 'precipitations'}, inplace=True)

# Converting 'date' to ordinal for linear regression
df['ordinal_date'] = df['date'].map(dt.datetime.toordinal)

# Linear regression
X = sm.add_constant(df['ordinal_date'])
y = df['temp_moy']
model = sm.OLS(y, X, missing='drop').fit()  # 'missing=drop' to ignore NaN
df['trend'] = model.predict(X)

# Calculate the moving average
df['temp_moy_moving_avg'] = df['temp_moy'].rolling(window=12, min_periods=1).mean()

# Plotting the temperature trend with moving average and linear trend
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['temp_moy'], label='Temperature Average', alpha=0.5)
plt.plot(df['date'], df['trend'], label='Linear Trend', color='green', linewidth=2)
plt.plot(df['date'], df['temp_moy_moving_avg'], label='12-month Moving Average', color='red', linewidth=2)
plt.title('Average Temperature in Bordeaux', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig('temperature_trend.png')
plt.show()


# Régression linéaire pour les précipitations
X_precip = sm.add_constant(df['ordinal_date'])
y_precip = df['precipitations']
model_precip = sm.OLS(y_precip, X_precip, missing='drop').fit()
df['trend_precip'] = model_precip.predict(X_precip)



# Affichage de la tendance des précipitations
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['precipitations'], label='Précipitations', alpha=0.5)
plt.plot(df['date'], df['trend_precip'], label='Tendance Linéaire des Précipitations', color='blue', linewidth=2)
plt.title('Tendance Linéaire des Précipitations à Bordeaux', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Précipitations (mm)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()


# Décomposition saisonnière des précipitations
decompose_result_precip = seasonal_decompose(df['precipitations'].dropna(), model='additive', period=12)
decompose_result_precip.plot()
plt.show()


# Histogramme des précipitations annuelles avec indication des années extrêmes
plt.figure(figsize=(14, 7))
annual_precipitation.plot(kind='bar', color='skyblue', alpha=0.7)
high_precip_years.plot(kind='bar', color='red', alpha=0.7)
low_precip_years.plot(kind='bar', color='green', alpha=0.7)
plt.title('Précipitations Annuelles à Bordeaux avec Années Extrêmes', fontsize=16)
plt.xlabel('Année', fontsize=14)
plt.ylabel('Précipitations (mm)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.show()


# Corrélation mensuelle entre température et précipitations
monthly_corr = df.groupby('month')[['temp_moy', 'precipitations']].corr().iloc[0::2, -1]
print("Corrélation mensuelle entre température et précipitations:\n", monthly_corr)


# Boxplot des températures par mois avec l'ordre correct des mois
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['month'] = df['date'].dt.strftime('%B')  # Convertit en noms de mois
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)  # Convertit en catégorie ordonnée

# First, define the 'annual_precipitation' right after the data preparation section
df['year'] = df['date'].dt.year
annual_precipitation = df.groupby('year')['precipitations'].sum()

# Then, define the 'high_precip_years' and 'low_precip_years'
high_precip_years = annual_precipitation[annual_precipitation > annual_precipitation.quantile(0.9)]
low_precip_years = annual_precipitation[annual_precipitation < annual_precipitation.quantile(0.1)]

# Now, the plotting should not give you the NameError
plt.figure(figsize=(14, 7))
annual_precipitation.plot(kind='bar', color='skyblue', alpha=0.7)
high_precip_years.plot(kind='bar', color='red', alpha=0.7)
low_precip_years.plot(kind='bar', color='green', alpha=0.7)
plt.title('Annual Precipitation in Bordeaux with Extreme Years Highlighted', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Precipitation (mm)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.show()

# Test de Mann-Kendall pour détecter les tendances
df['ordinal_date'] = df['date'].apply(lambda x: x.toordinal())  # Convertit la date en entier ordinal
tau, p_value = kendalltau(df['ordinal_date'], df['temp_moy'])
print(f'Mann-Kendall tau: {tau}, p-value: {p_value}')

# Interprétation du test
if p_value < 0.05:
    print("Tendance significative détectée dans la température moyenne")
else:
    print("Aucune tendance significative détectée dans la température moyenne")


# Analyse statistique
# Assurez-vous de n'inclure que les colonnes numériques pour la corrélation
numerical_df = df.select_dtypes(include=[np.number])  # Inclure seulement les données numériques

# Calcul de la matrice de corrélation sur le DataFrame nettoyé
correlation_matrix = numerical_df.corr()

# Affichage de la matrice de corrélation avec une heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation des données météorologiques', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.show()

# Test de Mann-Kendall pour les tendances dans les précipitations
tau_precip, p_value_precip = kendalltau(df['ordinal_date'], df['precipitations'])
print(f"Mann-Kendall tau pour les précipitations: {tau_precip}, p-value: {p_value_precip}")

# Interprétation du test pour les précipitations
if p_value_precip < 0.05:
    print("Tendance significative détectée dans les précipitations")
else:
    print("Aucune tendance significative détectée dans les précipitations")
