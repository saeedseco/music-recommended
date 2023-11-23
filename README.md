# music-recommended
This is a simple machine learning project to provide suggested music genre based on your age.
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music = pd.read_csv('D:\programing\qest.csv')
X = music.drop(columns = ['genre'])
Y = music['genre']
model = DecisionTreeClassifier()
model.fit(X, Y)
predections = model.predict([ [45, 1], [23,0] ])
predections 
