#%%
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../recovery")
from FDModel import loadProcessedData

%load_ext autoreload
%autoreload 2

# %%

sFilenameTrain = "./trainingData.csv"
f = open(sFilenameTrain, 'w')
f.truncate(0)

matList = open("../data/DataSetListNonMetals.txt").read().splitlines()
nMats = len(matList)
for iter in range(nMats):
    strObjName = matList[iter]

    if (strObjName != ''):
        strMaterial = strObjName.split("_")
        material = strMaterial[0]
        # print(material)

        # get the features from saved data
        _, _, _, _, features = loadProcessedData(strObjName)
        features = features.tolist()
        features += [material]

        with open(sFilenameTrain, "ab") as f:
            np.savetxt(f, [features], delimiter=",", fmt="% s")

#%%

## PLOT
# Plot (K, EpsP) clusters
# Plotting code - to keep sanity in the code,
# feature collection is done above separately
# Uncomment below to plot
'''
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Lucida Grande']
from matplotlib.ticker import MultipleLocator

plotMarkers = ["x",">","<","+","o","p","s","D","^","P","v","1","2","4","8","*"]
plotColors = ["red", "green", "blue", "navy", "violet", "brown", "orange", "pink", "black", "cyan", "lavender", "darkorchid", "deepskyblue", "salmon", "rosybrown", "gold"]

# Figure subplots
fig, ax = plt.subplots(figsize=(15, 13.5))
# Title
ax.set_title("(b) Diffusivity vs Source term at center", fontsize=40, pad=30)

# Can add all the required data to plot in the list in the text file
matList = open("../data/DataSetListNonMetals.txt").read().splitlines()
nMats = len(matList)
matCount = -1
for iter in range(nMats):
    strObjName = matList[iter]
  
    if (strObjName != ''):    
        material = strObjName.split("_")

        # get the features from saved data
        _, _, _, _, features = loadProcessedData(strObjName)
        features = features.tolist()
        features += [material]

        # plot the center value to create the scatter plot
        if (material[1]=='1'):
            matCount = matCount + 1
            plt.scatter(features[1], features[0], label=material[0], 
                        s=600,c=plotColors[matCount],
                        marker=plotMarkers[matCount], linewidths=4)
        else:
            plt.scatter(features[1], features[0], s=600,
                        c=plotColors[matCount],
                        marker=plotMarkers[matCount], linewidths=4)

# Labels
ax.set_xlabel(r"$\epsilon' \times fInp$ (Absorption times source)", fontsize=40, labelpad=30)
ax.set_ylabel("K (Diffusivity)", fontsize=40, labelpad=25)
# Legend
ax.legend(bbox_to_anchor=(0.91, 0.98), bbox_transform=fig.transFigure,
           ncol = 1, fontsize=40)
# Major ticks
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.tick_params(which='major', direction='in', length=15, width=2, colors='black',
               grid_color='green', grid_alpha=0.5, labelsize=40, top=True, right=True)
# Minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(0.02))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))
ax.tick_params(which='minor', direction='in', length=10, color='black', top=True, right=True)
plt.savefig("Fig_ScatterPlot.pdf", bbox_inches="tight")
'''