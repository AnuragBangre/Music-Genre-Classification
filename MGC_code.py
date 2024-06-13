import pandas as pd
import numpy as np
# Importing dataset
df = pd.read_csv('F:/SEM 6/TH Artificial Intelligence/Project/MGC_data.csv')
df.head()
# Preprocesing
print(df['class_name'].unique())
df['class_name'] = df['class_name'].astype('category')
df['class_label'] = df['class_name'].cat.codes
lookup_genre_name = dict(zip(df.class_label.unique(), df.class_name.unique())) 
lookup_genre_name
df['class_name'].unique()
cols = list(df.columns)
cols.remove('label')
cols.remove('class_label')
cols.remove('class_name')
#print(df[cols])
# Data splitting for Training and Testing
%matplotlib notebook
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X = df.iloc[:,1:28]
y = df['class_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
# Min-Max Normalisation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Model fitting using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)
# Fetching File Metadata for Prediction
from Metadata import getmetadata
a = getmetadata("F:/SEM 6/TH Artificial Intelligence/Project/sample.wav")
# Prediction
d1 =np.array(a)
data1 = scaler.transform([d1])
genre_prediction = knn.predict(data1)
print(lookup_genre_name[genre_prediction[0]])
