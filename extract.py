import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import kendalltau
import numpy as np


# Remplacez 'your_api_key' par votre clé API réelle
headers = {
    'x-rapidapi-key': "6ceb7b2424msh1982d3d9a2696a6p11cb49jsnce30609a6a6a",
    'x-rapidapi-host': "meteostat.p.rapidapi.com"
}
params = {
    'station': '07510',
    'start': '1990-01-01',
    'end': '2024-04-01'
}

# Effectuez la requête pour obtenir les données
response = requests.get('https://meteostat.p.rapidapi.com/stations/monthly', headers=headers, params=params)

# Vérifiez si la requête a réussi
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data['data'])
    df.to_csv('bordeaux_weather_data.csv', index=False)
else:
    print(f"Failed to fetch data: {response.status_code}")
    exit()

# Nettoyage et préparation des données
df = pd.read_csv('bordeaux_weather_data.csv')
df['date'] = pd.to_datetime(df['date'])
mean_values = df.select_dtypes(include=[float]).mean()
df.fillna(mean_values, inplace=True)
df.drop(columns=['snow', 'tsun'], inplace=True)
df.drop_duplicates(inplace=True)
df.rename(columns={'tavg': 'temp_moy', 'prcp': 'precipitations'}, inplace=True)
df.to_csv('bordeaux_weather_data_cleaned.csv', index=False)

# Visualisation des données
df = pd.read_csv('bordeaux_weather_data_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])

# Application d'une moyenne mobile pour lisser la série temporelle
df['temp_moy_moving_avg'] = df['temp_moy'].rolling(window=12).mean()

# Graphique linéaire des températures moyennes avec lissage
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['temp_moy'], label='Température Moyenne', alpha=0.5)
plt.plot(df['date'], df['temp_moy_moving_avg'], label='Moyenne mobile 12 mois', color='red', linewidth=2)
plt.title('Température moyenne à Bordeaux', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Température (°C)', fontsize=14)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()  # Ajuste la mise en page pour éviter les chevauchements
plt.grid(True)
plt.savefig('temperature_trend.png')  # Enregistre la figure en image PNG
plt.show()

# Histogramme des précipitations annuelles
df['year'] = df['date'].dt.year
annual_precipitation = df.groupby('year')['precipitations'].sum()
plt.figure(figsize=(14, 7))
annual_precipitation.plot(kind='bar', color='skyblue')
plt.title('Précipitations annuelles à Bordeaux', fontsize=16)
plt.xlabel('Année', fontsize=14)
plt.ylabel('Précipitation (mm)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.savefig('annual_precipitation.png')
plt.show()

# Boxplot des températures par mois avec l'ordre correct des mois
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['month'] = df['date'].dt.strftime('%B')  # Convertit en noms de mois
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)  # Convertit en catégorie ordonnée

plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='month', y='temp_moy', order=month_order)
plt.title('Distribution de la température moyenne par mois', fontsize=16)
plt.xlabel('Mois', fontsize=14)
plt.ylabel('Température (°C)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.savefig('monthly_temp_distribution.png')
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

# Test de Mann-Kendall pour détecter les tendances
tau, p_value = kendalltau(df['date'].apply(lambda x: x.toordinal()), df['temp_moy'])
print(f'Mann-Kendall tau: {tau}, p-value: {p_value}')

# Interprétation du test
if p_value < 0.05:
    print("Tendance significative détectée dans la température moyenne")
else:
    print("Aucune tendance significative détectée dans la température moyenne")
