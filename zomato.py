import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snp

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import utils, preprocessing
from sklearn.metrics import classification_report, confusion_matrix

# data preprocessing
restaurant_data = pd.read_csv('zomato.csv', encoding='latin-1')
filter_data = restaurant_data.drop(['Restaurant Name','City','Address','Locality','Locality Verbose','Cuisines','Currency','Has Table booking','Is delivering now','Switch to order menu','Rating color','Rating text','Has Online delivery'], axis=1)
restaurant_data.head(2)
filter_data.head(2)
filter_data.describe()

# EDA
	fig,axes = plt.subplots(nrows=1, ncols=3,figsize=(10,9))
snp.scatterplot(x='Votes', y='Aggregate rating', data=restaurant_data, hue='Rating text', ax=axes[0])
snp.scatterplot(x='Average Cost for two', y='Price range', data=restaurant_data, hue='Aggregate rating', ax=axes[1])
snp.scatterplot(x='Latitude', y='Longitude', data=restaurant_data, hue='Price range',ax=axes[2])
snp.pairplot(data=restaurant_data)
plt.tight_layout()

# ** Apply Logistic Regression Algorithm **
log = LogisticRegression()
scalar = StandardScaler()

filter_data['Agg rating'] = filter_data[(filter_data['Aggregate rating'] == 0) | (filter_data['Aggregate rating'] == 4)]['Aggregate rating']
filter_data.fillna(value = filter_data['Aggregate rating'].mean(), axis=1, inplace=True)
filter_data.drop('Restaurant ID',axis=1, inplace=True)
filter_data.drop('Aggregate rating',axis=1, inplace=True)

X = filter_data.drop(['Agg rating'], axis=1)
y = filter_data['Agg rating']
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, encoded, test_size=0.30, random_state=101)

log.fit(X_train, y_train)

prediction = log.predict(X_test)

print(classification_report(y_test, prediction))
print('\n')
print(confusion_matrix(y_test, prediction))

plt.show()
