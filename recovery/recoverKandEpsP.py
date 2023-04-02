#%%
# Import required libraries and packages
from FDModel import run_optimization, loadProcessedData
from processData import processData
from utils import config_parser

%load_ext autoreload
%autoreload 2

import torch
if (torch.cuda.is_available()):
    print(torch.cuda.get_device_name(0))
else:
    print("Cuda not available, using CPU")

#%%
# Option 1: Execute for a specific material dataset
# Choose the dataset to run by
# choosing the directory name of the material
strObjName = "Rosewood_1"
parser = config_parser("../data/" + strObjName + "/params.txt")
params = parser.parse_args(args=[])
print(params)

processData(strObjName, params)
run_optimization(strObjName, params.learning_rate, params.total_iterations)
K, Eps, numLayersK, numLayersEps, features = loadProcessedData(strObjName)
#%%
# Uncomment below to use
'''
# Option 2: Execute for the complete dataset of materials
matList = open("../data/DataSetList.txt").read().splitlines()
nMats = len(matList)
for iter in range(nMats):
    strObjName = matList[iter]
    print(strObjName)
    if (strObjName != ''):
        parser = config_parser("../data/" + strObjName + "/params.txt")
        params = parser.parse_args(args=[])
        print(params)
        processData(strObjName, params)
        run_optimization(strObjName, params.learning_rate, params.total_iterations)
'''
# %%
