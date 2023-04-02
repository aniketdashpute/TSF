#%%
import numpy as np
import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%

# Create material list

MatDict = {}
matList = open("../data/DataSetListNonMetals.txt").read().splitlines()
nMats = len(matList)
for iter in range(nMats):
    strObjName = matList[iter]
    
    if (strObjName != ''):
        strMaterial = strObjName.split("_")
        material = strMaterial[0]
        #print(material)

        MatDict[material] = 1 # just add an entry

Materials = list(MatDict.keys())
print(Materials)

#%%

# Read data

#Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('trainingData.csv', header = None)
[nM, nF] = dataset.shape
print('Shape of the dataset: ' + str(dataset.shape))
#Creating the dependent variable class
factor = pd.factorize(dataset[nF-1])

# print("label nums: ", factor[0])
# print("label names: ", factor[1])

numSamples = 4
numFeatures = nF-1
numMaterials = nM//numSamples

X_train_full = dataset.iloc[:,0:numFeatures].values
y_train_full = factor[0]
# y_train_full = dataset.iloc[:,numFeatures].values

# Feature Scaling
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)

X_train = np.zeros([(numSamples-1)*numMaterials, numFeatures])
y_train = np.zeros((numSamples-1)*numMaterials)
X_test = np.zeros([numMaterials,numFeatures])
y_test = np.zeros(numMaterials)

y_test_full = []
y_pred_full = []

for i in range(numSamples):
    iter = 0
    iter_test = 0
    for j in range(numSamples*numMaterials):
        if (j%numSamples != i):
            X_train[iter,:] = X_train_full[j,:]
            y_train[iter] = y_train_full[j]
            iter += 1
        else:
            X_test[iter_test,:] = X_train_full[j,:]
            y_test[iter_test] = y_train_full[j]
            iter_test += 1
    # print(y_train)
    # print(y_test)

    # Using an MLP to classify the data
    classifier = MLPClassifier(hidden_layer_sizes=(90,), random_state=1, max_iter=2000, alpha=0.0001)

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    # y_pred_train = classifier.predict(X_train)
    y_pred = classifier.predict(X_test)

    mat_test = [ Materials[int(i)] for i in y_test]
    mat_pred = [ Materials[int(i)] for i in y_pred]
    # print('================')
    # print(mat_test)
    # print('-----')
    # print(mat_pred)
    # print('================')
    
    # print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))
    # print(y_test==y_pred)

    y_test_full.append(y_test)
    y_pred_full.append(y_pred)

y_test_full = np.array(y_test_full).flatten()
y_pred_full = np.array(y_pred_full).flatten()

# Confusion matrix
print(confusion_matrix(y_test_full, y_pred_full))

# Print Accuracy
acc = 100*sum(y_test_full==y_pred_full)/sum(y_test_full==y_test_full)
print("Accuracy is: ", acc, "%")
print("NumFeatures is: ", numFeatures)
# %%
