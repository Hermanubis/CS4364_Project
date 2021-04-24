import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def group_stage_proc(group_data, fifaGroupDS):      
    group_data = group_data.groupby('Team').sum()
    
    group_data = pd.merge(group_data, fifaGroupDS, on='Team')
    
    # Dropping unnecessary columns    
    group_data = group_data.drop(['Goals in PSO', 'Own goals', 'Group', 'GroupWin', 
                                  'GroupDraw', 'GroupLost', 'GroupScore'], axis = 'columns')
    
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    
    y = group_data['Advance']
    
    numeric = [i for i in group_data.columns if group_data[i].dtype in [np.int64]]
    X = group_data[numeric]
    
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X)
    y_train = y
    # X_test = sc.transform(X_test)
    
    RandomForest = RandomForestClassifier(criterion="entropy")
    RandomForest.fit(X_train, y_train)
    featureImportance2 = RandomForest.feature_importances_
    fea_importance_ranked = []
    
    print("\nRandom Forest: ")
    for i in range(len(featureImportance2)):
        fea_importance_ranked.append((X.columns[i], featureImportance2[i]))
    
    fea_importance_ranked.sort(key=lambda x:x[1], reverse=True)
    for tup in fea_importance_ranked:
        print("Feature: " + str(tup[0]) + " --- Importance: " + str(tup[1]))
    
    return fea_importance_ranked
    
    # y_pred = RandomForest.predict(X_test)
    # randomForestScore = accuracy_score(y_test, y_pred)
    # print("Accuracy Score: " + str(randomForestScore))
    
    # model = LogisticRegression(max_iter = 50000)
    # model.fit(X_train, y_train)
    # trainScore = model.score(X_train, y_train)
    # testScore = model.score(X_test, y_test)
    
    # # y_pred = model.predict(test_inputs)
    # # playoffScore = accuracy_score(y_test, y_pred)
    
    # print("\nLogistic Regression: ")
    # print("Train Score: " + str(trainScore))
    # print("Test Score: " + ": " + str(testScore))
    
    
def Rof16_stage_proc(Rof16_data, group_res):
    Rof16_data = Rof16_data.groupby('Team').sum()
    
    Rof16_res = {}
    
    for i, row in Rof16_data.iterrows():
        temp_res = 0
        for tup in group_res:
            feature_name = tup[0]
            feature_impo = tup[1]
            
            temp_res += row[feature_name] * feature_impo
        Rof16_res[row.name] = temp_res
              
    pprint.pprint(Rof16_res)
        
    return Rof16_res

def Rof16_Improve(Rof16_res, Rof16_data):  
    Rof16_improved = {}
    
    for i in range(0, len(Rof16_data), 2):
        row = Rof16_data.iloc[i]
        team = row['Team']
        opponent = row['Opponent']
        result = row['Win']
        
        team_score = Rof16_res[team]
        opponent_score = Rof16_res[opponent]
        dif = abs(team_score - opponent_score)
        
        # Team defeat opponent
        if(result == 'Yes'):
            if(team_score > opponent_score):
                Rof16_improved[team] = team_score + dif * 0.2
                Rof16_improved[opponent] = opponent_score - dif * 0.2
            else:
                Rof16_improved[team] = team_score + dif * 0.8
                Rof16_improved[opponent] = opponent_score - dif * 0.8
        else:
            if(team_score > opponent_score):
                Rof16_improved[team] = team_score - dif * 0.8
                Rof16_improved[opponent] = opponent_score + dif * 0.8
            else:
                Rof16_improved[team] = team_score - dif * 0.2
                Rof16_improved[opponent] = opponent_score + dif * 0.2
    
    pprint.pprint(Rof16_improved)
    
    return Rof16_improved
    
def QuarF_Improve(Rof16_improved, QuarF_data):  
    QuarF_improved = {}
    
    for i in range(0, len(QuarF_data), 2):
        row = QuarF_data.iloc[i]
        team = row['Team']
        opponent = row['Opponent']
        result = row['Win']
        
        team_score = Rof16_improved[team]
        opponent_score = Rof16_improved[opponent]
        dif = abs(team_score - opponent_score)
        
        # Team defeat opponent
        if(result == 'Yes'):
            if(team_score > opponent_score):
                QuarF_improved[team] = team_score + dif * 0.4
                QuarF_improved[opponent] = opponent_score - dif * 0.4
            else:
                QuarF_improved[team] = team_score + dif * 0.6
                QuarF_improved[opponent] = opponent_score - dif * 0.6
        else:
            if(team_score > opponent_score):
                QuarF_improved[team] = team_score - dif * 0.6
                QuarF_improved[opponent] = opponent_score + dif * 0.6
            else:
                QuarF_improved[team] = team_score - dif * 0.4
                QuarF_improved[opponent] = opponent_score + dif * 0.4
    
    pprint.pprint(QuarF_improved)
    

fifaDS =  pd.read_csv('FIFA18_Statistics.csv', na_values="not available")
playoffDS =  pd.read_csv('FIFA18_Statistics.csv', na_values="not available")
fifaGroupDS =  pd.read_csv('FIFA18_Group_Statistics.csv', na_values="not available")

# fifaDS.drop(fifaDS.tail(16).index, inplace = True)
inputs = fifaDS

# inputs = fifaDS.drop('Win', axis = 'columns')

# Divide data into group, round of 16, quarter, semi and finals 
group_data = inputs[inputs['Round'] == 'Group Stage']
Rof16_data = inputs[inputs['Round'] == 'Round of 16']
QuarF_data = inputs[inputs['Round'] == 'Quarter Finals']
SemiF_data = inputs[inputs['Round'] == 'Semi- Finals']
Final_data = inputs[inputs['Round'] == 'Final']

# fifaDS.drop(fifaDS.tail(16).index, inplace = True)
# y = (fifaDS['Win'])
# inputs = fifaDS.drop('Win', axis = 'columns')
# inputs = inputs.drop(['Date', 'Man of the Match', 'Round', 'PSO', 'Goals in PSO'], axis = 'columns')

# numeric = [i for i in inputs.columns if inputs[i].dtype in [np.int64]]
# X = inputs.iloc[:, 1:]
# X = inputs[numeric]

# playoffDS.drop(playoffDS.head(112).index, inplace = True)
# y_test = (playoffDS['Win'])
# # print(playoffDS)
# test_inputs = playoffDS.drop('Win', axis = 'columns')
# test_inputs = test_inputs.drop(['Date', 'Man of the Match', 'Round', 'PSO', 'Goals in PSO'], axis = 'columns')


# numeric_y = [i for i in test_inputs.columns if test_inputs[i].dtype in [np.int64]]
# X = inputs.iloc[:, 1:]
# x_test = test_inputs[numeric_y]

# le_team = LabelEncoder()

# X['team_new'] = le_team.fit_transform(X['Team'])

# corr = X.corr()
# plt.figure(figsize=(18, 15))
# sns.heatmap(corr, linewidths=0.01, square=True, annot=True, linecolor='Black')
# plt.title('Feature correlation heatmap')

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

group_res = group_stage_proc(group_data, fifaGroupDS)

print('\nRof16 Result\n')

Rof16_res = Rof16_stage_proc(Rof16_data, group_res)

print('\nRof16 Improved\n')

Rof16_improved = Rof16_Improve(Rof16_res, Rof16_data)

print('\nQuarF Improved\n')

QuarF_improved = QuarF_Improve(Rof16_improved, QuarF_data)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# model = LogisticRegression(max_iter = 50000)
# model.fit(X_train, y_train)
# trainScore = model.score(X_train, y_train)
# testScore = model.score(X_test, y_test)

# y_pred = model.predict(test_inputs)
# playoffScore = accuracy_score(y_test, y_pred)

# print("\nLogistic Regression: ")
# print("Train Score: " + str(trainScore))
# print("Test Score: " + ": " + str(testScore))
# print("Prediction accuracy score: " + str(playoffScore))
