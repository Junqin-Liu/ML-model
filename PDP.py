
#############################################################
##############     model interpretation      ################          
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
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance
import joblib
import copy
import torch
torch.cuda.is_available()
######## Coordination type Demo ########

def pdp1d(model, feature):
    global pdplist,va
    if feature == 2 :
        va = partial_dependence(rf, x_train, feature, response_method='auto', percentiles=(0, 0.99), grid_resolution=20, method='auto')[1]
    elif feature == 10 :
        va = partial_dependence(rf, x_train, feature, response_method='auto', percentiles=(0, 0.963), grid_resolution=20, method='auto')[1]
    elif feature == 11:
        va = partial_dependence(rf, x_train, feature, response_method='auto', percentiles=(0, 0.97), grid_resolution=20, method='auto')[1]
    else: 
        va = partial_dependence(rf, x_train, feature, response_method='auto', percentiles=(0, 1), grid_resolution=20, method='auto')[1]
    k = len(va[0])  
    x_train1 = copy.deepcopy(x_train)
    for i in range(k):
      x_train1[feature]=va[0][i]
      pdp = model.predict_proba(x_train1)[:,1].mean()
      pdplist.append(pdp)
    return pdplist,va


gcf = joblib.load("...your path.../Coordination type.pkl")
sc = joblib.load("...your path/Coordination type Sc")
x_train = pd.DataFrame(x_train)

'''
columns =('Mineral','SSA','Fe.O_at','Cellb','Metal',
          'O_Num','Radius','Electronegativity','pHe',
          'C_mineral','C_OM','C_metal','Time',
          'Temperature','IonicStrength')
'''

pdplist = []
va = []
writer= pd.ExcelWriter('...your path.../Coordination type.xlsx') 
for i in range(15):
  pdplist = []
  va = []
  pdp1d(gcf, i)
  pdpdf=pd.DataFrame({columns[i]:(va[0]*sc.scale_[i]+sc.mean_[i]),'Me-Fe Coordination probability':pdplist})
  pdpdf.to_excel(writer,sheet_name=columns[i])
  writer.save()



def pdp2d(model, feature1, feature2):
    global pdplist2d,XX,YY
    if feature1 == 2:
        XX = partial_dependence(rf, x_train, feature1, response_method='auto', percentiles=(0, 0.99), grid_resolution=20, method='auto')[1]
    elif feature1 == 10 :
        XX = partial_dependence(rf, x_train, feature1, response_method='auto', percentiles=(0, 0.963), grid_resolution=20, method='auto')[1]
    elif feature1 == 11:
        XX = partial_dependence(rf, x_train, feature1, response_method='auto', percentiles=(0, 0.97), grid_resolution=20, method='auto')[1]
    else: 
        XX = partial_dependence(rf, x_train, feature1, response_method='auto', percentiles=(0, 1), grid_resolution=20, method='auto')[1]
    if feature2 == 2:
        YY = partial_dependence(rf, x_train, feature2, response_method='auto', percentiles=(0, 0.99), grid_resolution=20, method='auto')[1]
    elif feature2 == 10:
        YY = partial_dependence(rf, x_train, feature2, response_method='auto', percentiles=(0, 0.963), grid_resolution=20, method='auto')[1]
    elif feature2 == 11:
        YY = partial_dependence(rf, x_train, feature2, response_method='auto', percentiles=(0, 0.97), grid_resolution=20, method='auto')[1]
    else: 
        YY = partial_dependence(rf, x_train, feature2, response_method='auto', percentiles=(0, 1), grid_resolution=20, method='auto')[1]
    k = len(XX[0])  
    l = len(YY[0])
    x_train1 = copy.deepcopy(x_train)
    for i in range(k):
        pdplist = []
        x_train1[feature1]=XX[0][i]
        for j in range(l):
            x_train1[feature2]=YY[0][j]
            pdp = model.predict_proba(x_train1)[:,1].mean()
            pdplist.append(pdp)
        pdplist2d.append(pdplist)
    return pdplist2d,XX,YY

pdp, axes = partial_dependence(rf, x_train, [(1, 5)],  percentiles=(0, 1), grid_resolution=20)

pdplist2d = []
XX = []
YY = []
pdp2d(gcf, 2, 5)

XX1 = XX[0]*sc.scale_[2]+sc.mean_[2]
YY1 = YY[0]*sc.scale_[5]+sc.mean_[5]
#XX1, YY1 = np.meshgrid(axes[0], axes[1])
XX2, YY2 = np.meshgrid(XX1, YY1)
Z2 = np.asarray(pdplist2d).T

plt.close();plt.clf()
plt.figure(figsize=(5.5,5))
ctf= plt.contourf(XX2,YY2,Z2,10) 
ct = plt.contour(XX2,YY2,Z2,10,colors='white',linewidths=np.arange(0,1,0.5))    
#plt.clabel(ct, inline=True, fontsize=13)  
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel(columns[2],fontsize=16)
plt.ylabel(columns[5],fontsize=16)
cb = plt.colorbar(ctf,pad=0.02,ticks=[0.46,0.49,0.52,0.55,0.58,0.61,0.64],shrink=0.8)   
cb.ax.tick_params(labelsize=12)
plt.savefig('...Your path.../SSA vs Fe.O 2.png',dpi=100)
plt.show()


