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

Rof16_win_weight = 0.9
Rof16_lose_weight = 0.1
QuarF_win_weight = 0.6
QuarF_lose_weight = 0.4
# QuarF_win_weight = 0.4
# QuarF_lose_weight = 0.6

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
    
    #Obtain feature Importances using random forest
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
    
    
    
def Rof16_stage_proc(Rof16_data, group_res):
    Rof16_data = Rof16_data.groupby('Team').sum()
    
    Rof16_res = {}
    
    for i, row in Rof16_data.iterrows():
        temp_res = 0
        #Using feature importance as weight to compute team rating
        for tup in group_res:
            feature_name = tup[0]
            feature_impo = tup[1]
            
            temp_res += row[feature_name] * feature_impo
        Rof16_res[row.name] = temp_res
    
    #Ratings after group stage         
    pprint.pprint(Rof16_res)
        
    return Rof16_res

def Rof16_Improve(Rof16_res, Rof16_data):  
    Rof16_improved = {}
    
    for i in range(0, len(Rof16_data), 2):
        row = Rof16_data.iloc[i]
        team = row['Team']
        opponent = row['Opponent']
        result = row['Win']
        
        #update team ratings based on round of 16 results
        #Teams' rating changes depends on the opponent's rating
        team_score = Rof16_res[team]
        opponent_score = Rof16_res[opponent]
        # dif = abs(team_score - opponent_score)
        dif = (team_score + opponent_score) / 2
        
        # Team defeat opponent
        if(result == 'Yes'):
            if(team_score > opponent_score):
                Rof16_improved[team] = team_score + dif * Rof16_lose_weight
                Rof16_improved[opponent] = opponent_score - dif * Rof16_lose_weight
            else:
                Rof16_improved[team] = team_score + dif * Rof16_win_weight
                Rof16_improved[opponent] = opponent_score - dif * Rof16_win_weight
        else:
            if(team_score > opponent_score):
                Rof16_improved[team] = team_score - dif * Rof16_win_weight
                Rof16_improved[opponent] = opponent_score + dif * Rof16_win_weight
            else:
                Rof16_improved[team] = team_score - dif * Rof16_lose_weight
                Rof16_improved[opponent] = opponent_score + dif * Rof16_lose_weight
    
    pprint.pprint(Rof16_improved)
    
    return Rof16_improved
    
def QuarF_Improve(Rof16_improved, QuarF_data):  
    QuarF_improved = {}
    
    for i in range(0, len(QuarF_data), 2):
        row = QuarF_data.iloc[i]
        team = row['Team']
        opponent = row['Opponent']
        result = row['Win']
        
        #update team ratings based on Quarter Finals results
        #Teams' rating changes depends on the opponent's rating
        team_score = Rof16_improved[team]
        opponent_score = Rof16_improved[opponent]
        # dif = abs(team_score - opponent_score)
        dif = (team_score + opponent_score) / 2
        
        # Team defeat opponent
        if(result == 'Yes'):
            if(team_score > opponent_score):
                QuarF_improved[team] = team_score + dif * QuarF_lose_weight
                QuarF_improved[opponent] = opponent_score - dif * QuarF_lose_weight
            else:
                QuarF_improved[team] = team_score + dif * QuarF_win_weight
                QuarF_improved[opponent] = opponent_score - dif * QuarF_win_weight
        else:
            if(team_score > opponent_score):
                QuarF_improved[team] = team_score - dif * QuarF_win_weight
                QuarF_improved[opponent] = opponent_score + dif * QuarF_win_weight
            else:
                QuarF_improved[team] = team_score - dif * QuarF_lose_weight
                QuarF_improved[opponent] = opponent_score + dif * QuarF_lose_weight
    
    pprint.pprint(QuarF_improved)
    return QuarF_improved
    

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


# corr = X.corr()
# plt.figure(figsize=(18, 15))
# sns.heatmap(corr, linewidths=0.01, square=True, annot=True, linecolor='Black')
# plt.title('Feature correlation heatmap')

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

group_res = group_stage_proc(group_data, fifaGroupDS)

print('\nTeam Ratings After Group Stage:')

Rof16_res = Rof16_stage_proc(Rof16_data, group_res)

print('\nTeam Ratings After Round of 16:')

Rof16_improved = Rof16_Improve(Rof16_res, Rof16_data)

print('\nTeam Ratings After Quarter Finals:')

QuarF_improved = QuarF_Improve(Rof16_improved, QuarF_data)

sorted_values = sorted(QuarF_improved.values(),reverse = True) 
sorted_team = {}

for i in sorted_values:
    for j in QuarF_improved.keys():
        if QuarF_improved[j] == i:
            sorted_team[j] = QuarF_improved[j]
            break

print("\n Predicted Final Rannking: ")
rank = 1        
for ranking in sorted_team:
    print(" Top " + str(rank) + ": " + ranking)
    rank += 1
    if rank > 4:
        break


