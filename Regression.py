
#############################################################
##################     model training      ##################          
#############################################################
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from deepforest import CascadeForestRegressor  
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import calculateR2   
from sklearn.preprocessing import StandardScaler
import joblib
import torch
# put utils.py in the home page of your python file


##########################  
###### Me-O CN Demo ###### 

data = pd.read_excel("...your path.../Dummy.xlsx",sheet_name=0)
x = data[['Mineral','SSA','Fe.O_at','Cellb','Metal',
          'O_Num','Radius','Electronegativity','pHe',
          'C_mineral','C_OM','C_metal','Time',
          'Temperature','IonicStrength']]
y = data.iloc[:,32] # choose the label that you want to predict

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)   
sc = StandardScaler()
sc.fit(x_train)  
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
x_train.shape; y_train.shape; x_test.shape; y_test.shape 

### SVM ### 
import optuna
from optuna.samplers import TPESampler
from sklearn import svm

def SVR_objective(trial):
  gamma = trial.suggest_float('gamma', 1e-5, 1e0,log=True)
  C = trial.suggest_float("C", 1e-6, 1e1,log=True)
  svr = svm.SVR(kernel="rbf",gamma = gamma, C = C)
 
  kf = KFold(n_splits=5,shuffle=True)
  r2 = cross_val_score(svr, x_train, y_train, scoring='r2', cv=kf)
  return r2.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(SVR_objective, n_trials=50) 
study.best_params  
study.best_value    


### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = KFold(n_splits=5,shuffle=True)
R2list_train = [] 
R2list_test = [] 
rmselist_train = [] 
rmselist_test = [] 
maelist_train = []
maelist_test = []
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    svr = svm.SVR(kernel="rbf",gamma = 0.1188, C = 8.938)
    svr.fit(x_pretrain.values, y_pretrain.values) 

    p_train = svr.predict(x_pretrain)
    p_test = svr.predict(x_valid) 
    r2_trian = calculateR2(y_pretrain, p_train)
    r2_test = calculateR2(y_valid, p_test) 
    rmse_train = np.sqrt(mean_squared_error(y_pretrain, p_train))
    rmse_test = np.sqrt(mean_squared_error(y_valid, p_test)) 
    mae_train = mean_absolute_error(y_pretrain, p_train)
    mae_test = mean_absolute_error(y_valid, p_test)
    
    R2list_train.append(r2_trian) 
    R2list_test.append(r2_test)  
    rmselist_train.append(rmse_train)
    rmselist_test.append(rmse_test) 
    maelist_train.append(mae_train)
    maelist_test.append(mae_test)
    
R2df=pd.DataFrame({'tarin':R2list_train,'valid':R2list_test})   
#np.mean(R2df.iloc[:,0]);np.mean(R2df.iloc[:,1])
rmsedf=pd.DataFrame({'tarin':rmselist_train,'valid':rmselist_test})    
#np.mean(rmsedf.iloc[:,0]);np.mean(rmsedf.iloc[:,1])         
maedf=pd.DataFrame({'tarin':maelist_train,'valid':maelist_test})    
#np.mean(maedf.iloc[:,0]);np.mean(maedf.iloc[:,1])   


### testing
svr = svm.SVR(kernel="rbf",gamma = 0.1188, C = 8.938)
svr.fit(x_train, y_train)
p_test = svr.predict(x_test)   
R2_test = calculateR2(y_test, p_test);print(R2_test)
rmse = np.sqrt(mean_squared_error(y_test, p_test));print(rmse)  
mae = mean_absolute_error(y_test, p_test);print(mae)  


### GBDT ###
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingRegressor

def GBDT_objective(trial):
  max_depth = trial.suggest_int('max_depth', 2, 10)
  n_estimators = trial.suggest_int("n_estimators", 50, 300, 50)
  gbr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
 
  kf = KFold(n_splits=5,shuffle=True)
  r2 = cross_val_score(gbr, x_train, y_train, scoring='r2', cv=kf)
  return r2.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(GBDT_objective, n_trials=50)  
study.best_params  
study.best_value    

gbr = GradientBoostingRegressor(n_estimators=150, max_depth=4)
r2 = cross_val_score(gbr, x_train, y_train, scoring='r2', cv=kf)
r2.mean()
 
### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = KFold(n_splits=5,shuffle=True)
R2list_train = [] 
R2list_test = [] 
rmselist_train = [] 
rmselist_test = [] 
maelist_train = []
maelist_test = []
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    gbr = GradientBoostingRegressor(n_estimators=150, max_depth=4)
    gbr.fit(x_pretrain.values, y_pretrain.values) 

    p_train = gbr.predict(x_pretrain)
    p_test = gbr.predict(x_valid)   
    r2_trian = calculateR2(y_pretrain, p_train)
    r2_test = calculateR2(y_valid, p_test) 
    rmse_train = np.sqrt(mean_squared_error(y_pretrain, p_train))
    rmse_test = np.sqrt(mean_squared_error(y_valid, p_test)) 
    mae_train = mean_absolute_error(y_pretrain, p_train)
    mae_test = mean_absolute_error(y_valid, p_test)
    
    R2list_train.append(r2_trian) 
    R2list_test.append(r2_test)  
    rmselist_train.append(rmse_train)
    rmselist_test.append(rmse_test) 
    maelist_train.append(mae_train)
    maelist_test.append(mae_test)
    
R2df=pd.DataFrame({'tarin':R2list_train,'valid':R2list_test})   
#np.mean(R2df.iloc[:,0]);np.mean(R2df.iloc[:,1])
rmsedf=pd.DataFrame({'tarin':rmselist_train,'valid':rmselist_test})    
#np.mean(rmsedf.iloc[:,0]);np.mean(rmsedf.iloc[:,1])         
maedf=pd.DataFrame({'tarin':maelist_train,'valid':maelist_test})    
#np.mean(maedf.iloc[:,0]);np.mean(maedf.iloc[:,1])   


### testing
gbr = GradientBoostingRegressor(n_estimators=150, max_depth=4)
gbr.fit(x_train, y_train)
p_test = gbr.predict(x_test)   
R2_test = calculateR2(y_test, p_test);print(R2_test)
rmse = np.sqrt(mean_squared_error(y_test, p_test));print(rmse)  
mae = mean_absolute_error(y_test, p_test);print(mae)  


### rf ###
import optuna
from optuna.samplers import TPESampler

def rf_objective(trial):
  max_depth = trial.suggest_int('max_depth', 5, 30)
  n_estimators = trial.suggest_int("n_estimators", 50, 400, 50)
  rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=23)

  kf = KFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(rf, x_train, y_train, scoring='r2', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(rf_objective, n_trials=50) 
study.best_params  
study.best_value    

rf = RandomForestRegressor(n_estimators=400, max_depth=11,random_state=23)
r2 = cross_val_score(rf, x_train, y_train, scoring='r2', cv=kf)
r2.mean()

### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = KFold(n_splits=5,shuffle=True)
R2list_train = [] 
R2list_test = [] 
rmselist_train = [] 
rmselist_test = [] 
maelist_train = []
maelist_test = []
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    rf = RandomForestRegressor(n_estimators=400, max_depth=11,random_state=23)
    rf.fit(x_pretrain.values, y_pretrain.values) 

    p_train = rf.predict(x_pretrain)
    p_test = rf.predict(x_valid)  
    r2_trian = calculateR2(y_pretrain, p_train)
    r2_test = calculateR2(y_valid, p_test) 
    rmse_train = np.sqrt(mean_squared_error(y_pretrain, p_train))
    rmse_test = np.sqrt(mean_squared_error(y_valid, p_test)) 
    mae_train = mean_absolute_error(y_pretrain, p_train)
    mae_test = mean_absolute_error(y_valid, p_test)
    
    R2list_train.append(r2_trian) 
    R2list_test.append(r2_test)  
    rmselist_train.append(rmse_train)
    rmselist_test.append(rmse_test) 
    maelist_train.append(mae_train)
    maelist_test.append(mae_test)
    
R2df=pd.DataFrame({'tarin':R2list_train,'valid':R2list_test})   
#np.mean(R2df.iloc[:,0]);np.mean(R2df.iloc[:,1])
rmsedf=pd.DataFrame({'tarin':rmselist_train,'valid':rmselist_test})    
#np.mean(rmsedf.iloc[:,0]);np.mean(rmsedf.iloc[:,1])         
maedf=pd.DataFrame({'tarin':maelist_train,'valid':maelist_test})    
#np.mean(maedf.iloc[:,0]);np.mean(maedf.iloc[:,1])   


### tesing
rf = RandomForestRegressor(n_estimators=400, max_depth=11,random_state=23)
rf.fit(x_train, y_train)
p_test = rf.predict(x_test)   
R2_test = calculateR2(y_test, p_test);print(R2_test)
rmse = np.sqrt(mean_squared_error(y_test, p_test));print(rmse)  
mae = mean_absolute_error(y_test, p_test);print(mae)  


### gcf ###
import optuna
from optuna.samplers import TPESampler

def gcf_objective(trial):
  max_layers = trial.suggest_int('max_layers', 2, 5)
  n_trees = trial.suggest_int('n_trees', 100, 500, 50)
  n_estimators = n_estimators('n_estimators', 1, 4)
  gcf = CascadeForestRegressor(max_layers=max_layers, n_estimators =n_estimators,n_trees=n_trees)
  
  kf = KFold(n_splits=5,shuffle=True)
  r2 = cross_val_score(gcf, x_train, y_train, scoring='r2', cv=kf)
  return r2.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(gcf_objective, n_trials=50)
study.best_params  
study.best_value    

gcf = CascadeForestRegressor(max_layers=2,n_estimators=1,n_trees=150,random_state=23)
r2 = cross_val_score(gcf, x_train, y_train, scoring='r2', cv=kf)
r2.mean()

### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = KFold(n_splits=5,shuffle=True)
R2list_train = [] 
R2list_test = [] 
rmselist_train = [] 
rmselist_test = [] 
maelist_train = []
maelist_test = []
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    gcf = CascadeForestRegressor(max_layers=2,n_estimators=1,n_trees=150,random_state=23)
    gcf.fit(x_pretrain.values, y_pretrain.values) 

    p_train = gcf.predict(x_pretrain)
    p_test = gcf.predict(x_valid)   
    r2_trian = calculateR2(y_pretrain, p_train.flatten())
    r2_test = calculateR2(y_valid, p_test.flatten()) 
    rmse_train = np.sqrt(mean_squared_error(y_pretrain, p_train))
    rmse_test = np.sqrt(mean_squared_error(y_valid, p_test)) 
    mae_train = mean_absolute_error(y_pretrain, p_train)
    mae_test = mean_absolute_error(y_valid, p_test)
    
    R2list_train.append(r2_trian) 
    R2list_test.append(r2_test)  
    rmselist_train.append(rmse_train)
    rmselist_test.append(rmse_test) 
    maelist_train.append(mae_train)
    maelist_test.append(mae_test)
    
R2df=pd.DataFrame({'tarin':R2list_train,'valid':R2list_test})   
#np.mean(R2df.iloc[:,0]);np.mean(R2df.iloc[:,1])
rmsedf=pd.DataFrame({'tarin':rmselist_train,'valid':rmselist_test})    
#np.mean(rmsedf.iloc[:,0]);np.mean(rmsedf.iloc[:,1])         
maedf=pd.DataFrame({'tarin':maelist_train,'valid':maelist_test})    
#np.mean(maedf.iloc[:,0]);np.mean(maedf.iloc[:,1])   


### tesing 
gcf = CascadeForestRegressor(max_layers=2,n_estimators=1,n_trees=150,random_state=23)
gcf.fit(x_train, y_train)  
p_test = gcf.predict(x_test)   
R2_test = calculateR2(y_test, p_test.flatten());print(R2_test)
rmse = np.sqrt(mean_squared_error(y_test, p_test));print(rmse)  
mae = mean_absolute_error(y_test, p_test);print(mae)  

plt.clf();plt.close()
plt.scatter(y_test, gcf.predict(x_test).flatten())
plt.show()


        
