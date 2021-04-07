import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

fifaDS =  pd.read_csv('FIFA_Statistics.csv', na_values="not available")
playoffDS =  pd.read_csv('FIFA_Statistics.csv', na_values="not available")

fifaDS.drop(fifaDS.tail(16).index, inplace = True)
y = (fifaDS['Win'] == "Yes")
inputs = fifaDS.drop('Win', axis = 'columns')
inputs = inputs.drop('Man of the Match', axis = 'columns')

numeric = [i for i in inputs.columns if inputs[i].dtype in [np.int64]]
# X = inputs.iloc[:, 1:]
X = inputs[numeric]


# le_team = LabelEncoder()

# X['team_new'] = le_team.fit_transform(X['Team'])


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


RandomForest = RandomForestClassifier(criterion="entropy")
RandomForest.fit(X_train, y_train)
featureImportance2 = RandomForest.feature_importances_

print("\nRandom Forest: ")
for i in range(len(featureImportance2)):
    print("Feature: " + str(X.columns[i]) + " --- Importance: " + str(featureImportance2[i]))

y_pred2 = RandomForest.predict(X_test)
randomForestScore = accuracy_score(y_test, y_pred2)
print("Accuracy Score: " + str(randomForestScore))
