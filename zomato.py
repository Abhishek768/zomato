import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

restaurants_data = pd.read_csv('zomato.csv', encoding='latin-1')
print(restaurants_data.head())
print(restaurants_data.info())

plt.figure(figsize=(10,7))

snp.scatterplot(x='Votes', y='Aggregate rating', data=restaurant_data, hue='Rating color')
snp.scatterplot(x='Average Cost for two', y='Price range', data=restaurant_data, hue='Aggregate rating')
snp.pairplot(data=restaurant_data)
snp.scatterplot(x='Latitude', y='Longitude', data=restaurant_data, hue='Price range')
plt.tight_layout()

logmodel = RandomForestClassifier(n_estimators=100)
log = LogisticRegression()
scalar = StandardScaler()

scalar.fit(restaurant_data)
scalar.transform()

# X = restaurant_data.drop('Aggregate rating', axis=1)
# y = restaurant_data['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

log.fit(X_train, y_train)

plt.show()
