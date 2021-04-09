import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


fifaDS =  pd.read_csv('FIFA18_Statistics.csv', na_values="not available")
playoffDS =  pd.read_csv('FIFA18_Statistics.csv', na_values="not available")

fifaDS.drop(fifaDS.tail(16).index, inplace = True)
y = (fifaDS['Win'] == "Yes")
inputs = fifaDS.drop('Win', axis = 'columns')
inputs = inputs.drop(['Date', 'Man of the Match', 'Round', 'PSO', 'Goals in PSO'], axis = 'columns')

numeric = [i for i in inputs.columns if inputs[i].dtype in [np.int64]]
# X = inputs.iloc[:, 1:]
X = inputs[numeric]

playoffDS.drop(playoffDS.head(112).index, inplace = True)
y_test = (fifaDS['Win'] == "Yes")
# print(playoffDS)
test_inputs = playoffDS.drop('Win', axis = 'columns')
test_inputs = test_inputs.drop(['Date', 'Man of the Match', 'Round', 'PSO', 'Goals in PSO'], axis = 'columns')

numeric_y = [i for i in test_inputs.columns if test_inputs[i].dtype in [np.int64]]
# X = inputs.iloc[:, 1:]
x_test = test_inputs[numeric_y]

# le_team = LabelEncoder()

# X['team_new'] = le_team.fit_transform(X['Team'])

corr = X.corr()
plt.figure(figsize=(18, 15))
sns.heatmap(corr, linewidths=0.01, square=True, annot=True, linecolor='Black')
plt.title('Feature correlation heatmap')

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

y_pred = RandomForest.predict(X_test)
randomForestScore = accuracy_score(y_test, y_pred)
print("Accuracy Score: " + str(randomForestScore))

