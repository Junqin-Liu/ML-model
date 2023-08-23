
#############################################################
##################     mdoel training      ##################          
#############################################################
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from deepforest import CascadeForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score
from imblearn.over_sampling import BorderlineSMOTE 
from sklearn.preprocessing import StandardScaler
from collections import Counter
import joblib


####################################
### Me-O/Me-Fe coordination Demo ###
### oversampleing 
data = pd.read_excel("...your path.../Dummy.xlsx",sheet_name=0)
data.loc[data['Bond']=='Me-O and Me-Fe','Bond']= 1
data.loc[data['Bond']=='Me-O','Bond']= 0
data['Bond'] = data['Bond'].astype(int)

x = data[['Mineral','SSA','Fe.O_at','Cellb','Metal',
          'O_Num','Radius','Electronegativity','pHe',
          'C_mineral','C_OM','C_metal','Time',
          'Temperature','IonicStrength']]
y = data.iloc[:,31] 
print('Original dataset shape %s' % Counter(y))
os = BorderlineSMOTE(kind="borderline-1")
x_res, y_res = os.fit_resample(x, y)
print('Resampled dataset shape %s' % Counter(y_res))

 resdf = pd.concat([x_res,y_res], axis=1)
 writer= pd.ExcelWriter('...your path.../Me-Oresample.xlsx') 
 resdf.to_excel(writer,sheet_name='sheet')
 writer.save()
##!! after the Me-Oresample.xlsx is written, you should leave the final 15 features and substituted metal features by the preoperties of nearest metal

### 
data = pd.read_excel("...your path.../Me-Oresample.xlsx",sheet_name=0)
data = data.iloc[:,1:17]
print('Resampled dataset shape %s' % Counter(data['Bond']))
x = data[['Mineral','SSA','Fe.O_at','Cellb','Metal',
          'O_Num','Radius','Electronegativity','pHe',
          'C_mineral','C_OM','C_metal','Time',
          'Temperature','IonicStrength']]
y = data.iloc[:,15] 
print('Resampled dataset shape %s' % Counter(y))

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

def SVC_objective(trial):
  gamma = trial.suggest_float('gamma', 1e-5, 1e0,log=True)
  C = trial.suggest_float("C", 1e-6, 1e1,log=True)
  svc = svm.SVC(kernel="rbf",gamma = gamma, C = C)
 
  kf = StratifiedKFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(svc, x_train, y_train, scoring='accuracy', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(SVC_objective, n_trials=50) 
study.best_params  
study.best_value    

svc = svm.SVC(kernel="rbf",gamma = 0.68324, C = 9.03808)
accuracy = cross_val_score(svc, x_train, y_train, scoring='accuracy', cv=kf)
accuracy.mean()
 

### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = StratifiedKFold(n_splits=5,shuffle=True)
svc = svm.SVC(kernel="rbf",gamma = 0.6832, C = 9.0381)
cmlist_train = [] 
cmlist_test = [] 
auclist_train = [] 
auclist_test = [] 
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    svc.fit(x_pretrain, y_pretrain) 

    cm_train = confusion_matrix(y_pretrain, svc.predict(x_pretrain))
    cm_valid = confusion_matrix(y_valid, svc.predict(x_valid))
    auc_train = roc_auc_score(y_pretrain,svc.decision_function(x_pretrain))
    auc_valid = roc_auc_score(y_valid,svc.decision_function(x_valid))
    
    cmlist_train.append(cm_train)
    cmlist_test.append(cm_valid)
    auclist_train.append(auc_train)
    auclist_test.append(auc_valid)
    

cmlist_train[0];cmlist_train[1];cmlist_train[2];cmlist_train[3];cmlist_train[4]
cmlist_test[0];cmlist_test[1];cmlist_test[2];cmlist_test[3];cmlist_test[4]
aucdf=pd.DataFrame({'tarin':auclist_train,'valid':auclist_test})   

### testing
svc = svm.SVC(kernel="rbf",gamma = 0.6832, C = 9.038)
svc.fit(x_train, y_train)  
#cm_train = confusion_matrix(y_train,svc.predict(x_train))
cm = confusion_matrix(y_test,svc.predict(x_test))
auc = roc_auc_score(y_test,svc.decision_function(x_test))
accuracy = accuracy_score(y_test, svc.predict(x_test))
 

### GBDT ###
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier


def GBDT_objective(trial):
  max_depth = trial.suggest_int('max_depth', 2, 10)
  n_estimators = trial.suggest_int("n_estimators", 50, 300, 50)
  gbc = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
 
  kf = StratifiedKFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(gbc, x_train, y_train, scoring='accuracy', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(GBDT_objective, n_trials=50)  
study.best_params  
study.best_value    

gbc = GradientBoostingClassifier(n_estimators=150, max_depth=5)
accuracy = cross_val_score(gbc, x_train, y_train, scoring='accuracy', cv=kf)
accuracy.mean()

### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = StratifiedKFold(n_splits=5,shuffle=True)
gbc = GradientBoostingClassifier(n_estimators=150, max_depth=5)
cmlist_train = [] 
cmlist_test = [] 
auclist_train = [] 
auclist_test = [] 
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    gbc.fit(x_pretrain, y_pretrain) 

    cm_train = confusion_matrix(y_pretrain, gbc.predict(x_pretrain))
    cm_valid = confusion_matrix(y_valid, gbc.predict(x_valid))
    auc_train = roc_auc_score(y_pretrain,gbc.predict_proba(x_pretrain)[:,1])
    auc_valid = roc_auc_score(y_valid,gbc.predict_proba(x_valid)[:,1])
    
    cmlist_train.append(cm_train)
    cmlist_test.append(cm_valid)
    auclist_train.append(auc_train)
    auclist_test.append(auc_valid)
    

cmlist_train[0];cmlist_train[1];cmlist_train[2];cmlist_train[3];cmlist_train[4]
cmlist_test[0];cmlist_test[1];cmlist_test[2];cmlist_test[3];cmlist_test[4]
aucdf=pd.DataFrame({'tarin':auclist_train,'valid':auclist_test})   


### tesing
gbc = GradientBoostingClassifier(n_estimators=150, max_depth=5)
gbc.fit(x_train, y_train)  
#cm_train = confusion_matrix(y_train,svc.predict(x_train))
cm = confusion_matrix(y_test,gbc.predict(x_test))
auc = roc_auc_score(y_test,gbc.predict_proba(x_test)[:,1])


### rf ###
import optuna
from optuna.samplers import TPESampler

def rf_objective(trial):
  max_depth = trial.suggest_int('max_depth', 5, 30)
  n_estimators = trial.suggest_int("n_estimators", 50, 400, 50)
  rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

  kf = StratifiedKFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(rf, x_train, y_train, scoring='accuracy', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(rf_objective, n_trials=50)  
study.best_params  
study.best_value    

rf = RandomForestClassifier(n_estimators=100, max_depth=25,random_state=0)
accuracy = cross_val_score(rf, x_train, y_train, scoring='accuracy', cv=kf)
accuracy.mean()

### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = StratifiedKFold(n_splits=5,shuffle=True)
rf = RandomForestClassifier(n_estimators=100, max_depth=25,random_state=0)
cmlist_train = [] 
cmlist_test = [] 
auclist_train = [] 
auclist_test = [] 
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    rf.fit(x_pretrain, y_pretrain) 

    cm_train = confusion_matrix(y_pretrain, rf.predict(x_pretrain))
    cm_valid = confusion_matrix(y_valid, rf.predict(x_valid))
    auc_train = roc_auc_score(y_pretrain,rf.predict_proba(x_pretrain)[:,1])
    auc_valid = roc_auc_score(y_valid,rf.predict_proba(x_valid)[:,1])
    
    cmlist_train.append(cm_train)
    cmlist_test.append(cm_valid)
    auclist_train.append(auc_train)
    auclist_test.append(auc_valid)
    

cmlist_train[0];cmlist_train[1];cmlist_train[2];cmlist_train[3];cmlist_train[4]
cmlist_test[0];cmlist_test[1];cmlist_test[2];cmlist_test[3];cmlist_test[4]
aucdf=pd.DataFrame({'tarin':auclist_train,'valid':auclist_test})   


### testing 
rf = RandomForestClassifier(n_estimators=100, max_depth=25,random_state=0)
rf.fit(x_train, y_train)  
#cm_train = confusion_matrix(y_train,svc.predict(x_train))
cm = confusion_matrix(y_test,rf.predict(x_test))
auc = roc_auc_score(y_test,rf.predict_proba(x_test)[:,1])

### gcf  ###
import optuna
from optuna.samplers import TPESampler

def gcf_objective(trial):
  max_layers = trial.suggest_int('max_layers', 2, 5)
  n_trees = trial.suggest_int('n_trees', 100, 500, 50)
  min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20,5)
  gcf = CascadeForestClassifier(max_layers=max_layers, min_samples_leaf =min_samples_leaf,n_trees=n_trees)
  
  kf = StratifiedKFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(gcf, x_train, y_train, scoring='accuracy', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(gcf_objective, n_trials=50)  
study.best_params  
study.best_value    

gcf = CascadeForestClassifier(max_layers=2, min_samples_leaf =1,n_trees=400,random_state=0)
accuracy = cross_val_score(gcf, x_train, y_train, scoring='accuracy', cv=kf)
accuracy.mean()

### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = StratifiedKFold(n_splits=5,shuffle=True)
cmlist_train = [] 
cmlist_test = [] 
auclist_train = [] 
auclist_test = [] 
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    gcf = CascadeForestClassifier(max_layers=2, min_samples_leaf =1,n_trees=400,random_state=0)
    gcf.fit(x_pretrain.values, y_pretrain.values) 

    cm_train = confusion_matrix(y_pretrain, gcf.predict(x_pretrain))
    cm_valid = confusion_matrix(y_valid, gcf.predict(x_valid))
    auc_train = roc_auc_score(y_pretrain,gcf.predict_proba(x_pretrain)[:,1])
    auc_valid = roc_auc_score(y_valid,gcf.predict_proba(x_valid)[:,1])
    
    cmlist_train.append(cm_train)
    cmlist_test.append(cm_valid)
    auclist_train.append(auc_train)
    auclist_test.append(auc_valid)
    

cmlist_train[0];cmlist_train[1];cmlist_train[2];cmlist_train[3];cmlist_train[4]
cmlist_test[0];cmlist_test[1];cmlist_test[2];cmlist_test[3];cmlist_test[4]
aucdf=pd.DataFrame({'tarin':auclist_train,'valid':auclist_test})   


### tesing
gcf = CascadeForestClassifier(max_layers=2, min_samples_leaf =1,n_trees=400,random_state=0)
gcf.fit(x_train, y_train)   
#cm_train = confusion_matrix(y_train,gcf.predict(x_train))
cm = confusion_matrix(y_test,gcf.predict(x_test))
auc = roc_auc_score(y_test,gcf.predict_proba(x_test)[:,1])

plt.close();plt.clf()
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

plt.close();plt.clf(); 
auc = roc_auc_score(y_test,gcf.predict_proba(x_test)[:,1])
fpr,tpr, thresholds = roc_curve(y_test,gcf.predict_proba(x_test)[:,1])
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,color='darkorange',label='Me-O/Me-Fe Bond (AUC = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.title('Receiver Operating Characteristic Curve',fontsize=15)
plt.legend(loc="lower right",prop = {'size':12})
plt.show()



#############################
#####  Me-Fe shell Demo #####
data = pd.read_excel("...your path.../Dummy",sheet_name=1)
data.loc[data['FeShell']=='S','FeShell']= 1
data.loc[data['FeShell']=='M','FeShell']= 0
data['FeShell'] = data['FeShell'].astype(int)

x = data[['Mineral','SSA','Fe.O_at','Cellb','Metal',
          'O_Num','Radius','Electronegativity','pHe',
          'C_mineral','C_OM','C_metal','Time',
          'Temperature','IonicStrength']]
y = data.iloc[:,31] 
print('Original dataset shape %s' % Counter(y))


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

def SVC_objective(trial):
  gamma = trial.suggest_float('gamma', 1e-5, 1e0,log=True)
  C = trial.suggest_float("C", 1e-6, 1e1,log=True)
  svc = svm.SVC(kernel="rbf",gamma = gamma, C = C)
 
  kf = StratifiedKFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(svc, x_train, y_train, scoring='accuracy', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(SVC_objective, n_trials=50) 
study.best_params  
study.best_value    

svc = svm.SVC(kernel="rbf",gamma = 0.12511, C = 2.30738)
accuracy = cross_val_score(svc, x_train, y_train, scoring='accuracy', cv=kf)
accuracy.mean()

### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = StratifiedKFold(n_splits=5,shuffle=True)
svc = svm.SVC(kernel="rbf",gamma = 0.1251, C = 2.3074)
cmlist_train = [] 
cmlist_test = [] 
auclist_train = [] 
auclist_test = [] 
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    svc.fit(x_pretrain, y_pretrain) 

    cm_train = confusion_matrix(y_pretrain, svc.predict(x_pretrain))
    cm_valid = confusion_matrix(y_valid, svc.predict(x_valid))
    auc_train = roc_auc_score(y_pretrain,svc.decision_function(x_pretrain))
    auc_valid = roc_auc_score(y_valid,svc.decision_function(x_valid))
    
    cmlist_train.append(cm_train)
    cmlist_test.append(cm_valid)
    auclist_train.append(auc_train)
    auclist_test.append(auc_valid)
    

cmlist_train[0];cmlist_train[1];cmlist_train[2];cmlist_train[3];cmlist_train[4]
cmlist_test[0];cmlist_test[1];cmlist_test[2];cmlist_test[3];cmlist_test[4]
aucdf=pd.DataFrame({'tarin':auclist_train,'valid':auclist_test})   

### testing
svc = svm.SVC(kernel="rbf",gamma = 0.1251, C = 2.3074)
svc.fit(x_train, y_train)  
#cm_train = confusion_matrix(y_train,svc.predict(x_train))
cm = confusion_matrix(y_test,svc.predict(x_test))
auc = roc_auc_score(y_test,svc.decision_function(x_test))


### GBDT ###
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier


def GBDT_objective(trial):
  max_depth = trial.suggest_int('max_depth', 2, 10)
  n_estimators = trial.suggest_int("n_estimators", 50, 300, 50)
  gbc = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
 
  kf = StratifiedKFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(gbc, x_train, y_train, scoring='accuracy', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(GBDT_objective, n_trials=50)  
study.best_value    

gbc = GradientBoostingClassifier(n_estimators=200, max_depth=7)
accuracy = cross_val_score(gbc, x_train, y_train, scoring='accuracy', cv=kf)
accuracy.mean()

### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = StratifiedKFold(n_splits=5,shuffle=True)
cmlist_train = [] 
cmlist_test = [] 
auclist_train = [] 
auclist_test = [] 
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    gbc = GradientBoostingClassifier(n_estimators=200, max_depth=7)
    gbc.fit(x_pretrain, y_pretrain) 

    cm_train = confusion_matrix(y_pretrain, gbc.predict(x_pretrain))
    cm_valid = confusion_matrix(y_valid, gbc.predict(x_valid))
    auc_train = roc_auc_score(y_pretrain,gbc.predict_proba(x_pretrain)[:,1])
    auc_valid = roc_auc_score(y_valid,gbc.predict_proba(x_valid)[:,1])
    
    cmlist_train.append(cm_train)
    cmlist_test.append(cm_valid)
    auclist_train.append(auc_train)
    auclist_test.append(auc_valid)
    

cmlist_train[0];cmlist_train[1];cmlist_train[2];cmlist_train[3];cmlist_train[4]
cmlist_test[0];cmlist_test[1];cmlist_test[2];cmlist_test[3];cmlist_test[4]
aucdf=pd.DataFrame({'tarin':auclist_train,'valid':auclist_test})   


### testing
gbc = GradientBoostingClassifier(n_estimators=200, max_depth=7)
gbc.fit(x_train, y_train)  
#cm_train = confusion_matrix(y_train,svc.predict(x_train))
cm = confusion_matrix(y_test,gbc.predict(x_test))
auc = roc_auc_score(y_test,gbc.predict_proba(x_test)[:,1])
 
   
### rf ###
import optuna
from optuna.samplers import TPESampler

def rf_objective(trial):
  max_depth = trial.suggest_int('max_depth', 5, 30)
  n_estimators = trial.suggest_int("n_estimators", 50, 400, 50)
  rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

  kf = StratifiedKFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(rf, x_train, y_train, scoring='accuracy', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(rf_objective, n_trials=50) 
study.best_params  
study.best_value    

rf = RandomForestClassifier(n_estimators=50, max_depth=8,random_state=0)
accuracy = cross_val_score(rf, x_train, y_train, scoring='accuracy', cv=kf)
accuracy.mean()

### training and validation 
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = StratifiedKFold(n_splits=5,shuffle=True)
cmlist_train = [] 
cmlist_test = [] 
auclist_train = [] 
auclist_test = [] 
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    rf = RandomForestClassifier(n_estimators=50, max_depth=8,random_state=0)
    rf.fit(x_pretrain, y_pretrain) 

    cm_train = confusion_matrix(y_pretrain, rf.predict(x_pretrain))
    cm_valid = confusion_matrix(y_valid, rf.predict(x_valid))
    auc_train = roc_auc_score(y_pretrain,rf.predict_proba(x_pretrain)[:,1])
    auc_valid = roc_auc_score(y_valid,rf.predict_proba(x_valid)[:,1])
    
    cmlist_train.append(cm_train)
    cmlist_test.append(cm_valid)
    auclist_train.append(auc_train)
    auclist_test.append(auc_valid)
    

cmlist_train[0];cmlist_train[1];cmlist_train[2];cmlist_train[3];cmlist_train[4]
cmlist_test[0];cmlist_test[1];cmlist_test[2];cmlist_test[3];cmlist_test[4]
aucdf=pd.DataFrame({'tarin':auclist_train,'valid':auclist_test})   


### testing
rf = RandomForestClassifier(n_estimators=50, max_depth=8,random_state=0)
rf.fit(x_train, y_train)  
#cm_train = confusion_matrix(y_train,svc.predict(x_train))
cm = confusion_matrix(y_test,rf.predict(x_test))
auc = roc_auc_score(y_test,rf.predict_proba(x_test)[:,1])
  

### gcf ###
import optuna
from optuna.samplers import TPESampler

def gcf_objective(trial):
  max_layers = trial.suggest_int('max_layers', 2, 5)
  n_trees = trial.suggest_int('n_trees', 100, 500, 50)
  min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20,5)
  gcf = CascadeForestClassifier(max_layers=max_layers, min_samples_leaf =min_samples_leaf,n_trees=n_trees)
  
  kf = StratifiedKFold(n_splits=5,shuffle=True)
  accuracy = cross_val_score(gcf, x_train, y_train, scoring='accuracy', cv=kf)
  return accuracy.mean()

study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(gcf_objective, n_trials=50) 
study.best_params  
study.best_value    

gcf = CascadeForestClassifier(max_layers=2, min_samples_leaf =1,n_trees=100,random_state=0)
accuracy = cross_val_score(gcf, x_train, y_train, scoring='accuracy', cv=kf)
accuracy.mean()

### training and validation
x_train =  pd.DataFrame(x_train)
x_test =  pd.DataFrame(x_test)

kf = StratifiedKFold(n_splits=5,shuffle=True)
cmlist_train = [] 
cmlist_test = [] 
auclist_train = [] 
auclist_test = [] 
  for train_index, test_index in kf.split(x_train,y_train):
    global x_pretrain,y_pretrain,x_valid,y_valid
    x_pretrain= x_train.iloc[train_index] 
    y_pretrain= y_train.iloc[train_index]
    x_valid=x_train.iloc[test_index]
    y_valid=y_train.iloc[test_index] 
    gcf = CascadeForestClassifier(max_layers=2, min_samples_leaf =1,n_trees=100,random_state=0)
    gcf.fit(x_pretrain.values, y_pretrain.values) 

    cm_train = confusion_matrix(y_pretrain, gcf.predict(x_pretrain))
    cm_valid = confusion_matrix(y_valid, gcf.predict(x_valid))
    auc_train = roc_auc_score(y_pretrain,gcf.predict_proba(x_pretrain)[:,1])
    auc_valid = roc_auc_score(y_valid,gcf.predict_proba(x_valid)[:,1])
    
    cmlist_train.append(cm_train)
    cmlist_test.append(cm_valid)
    auclist_train.append(auc_train)
    auclist_test.append(auc_valid)
    

cmlist_train[0];cmlist_train[1];cmlist_train[2];cmlist_train[3];cmlist_train[4]
cmlist_test[0];cmlist_test[1];cmlist_test[2];cmlist_test[3];cmlist_test[4]
aucdf=pd.DataFrame({'tarin':auclist_train,'valid':auclist_test})   


### testing
gcf = CascadeForestClassifier(max_layers=2, min_samples_leaf =1,n_trees=100,random_state=0)
gcf.fit(x_train, y_train)   
#cm_train = confusion_matrix(y_train,gcf.predict(x_train))
cm = confusion_matrix(y_test,gcf.predict(x_test))
auc = roc_auc_score(y_test,gcf.predict_proba(x_test)[:,1])

plt.close();plt.clf()
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
plt.savefig('...your path....png',dpi=600)

plt.close();plt.clf(); 
auc = roc_auc_score(y_test,gcf.predict_proba(x_test)[:,1])
fpr,tpr, thresholds = roc_curve(y_test,gcf.predict_proba(x_test)[:,1])
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,color='darkorange',label='Me-O/Me-Fe Bond (AUC = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.title('Receiver Operating Characteristic Curve',fontsize=15)
plt.legend(loc="lower right",prop = {'size':12})
#plt.savefig('...your path....png',dpi=800)
plt.show()



