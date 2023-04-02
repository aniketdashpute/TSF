# TSF

```
pip install -r requirements.txt
```

## Data

Data link: https://drive.google.com/drive/folders/11Tu2-hhs92YUfApI6zG6UzioaYAq0t2a?usp=sharing

Download the data from ```the above link``` and place it in the ```data``` folder such that each of the extracted material folder has data as its parent directory, example: ```./data/Rosewood_1```

Recovery code is in ```recovery``` folder, and ```classification``` has the code for classification

## Recovering K and EpsP parameters

Inside the ```recovery``` directory, run the python file ```recoverKandEpsP.py``` (or use it as a jupyter notebook using VS Code feature, which uses ```#%%``` to convert the cells into executable jupyter code blocks)

## Classification

Inside the ```classification``` directory, run the python file ```generateTrainingDataset.py``` (or use it as a jupyter notebook using VS Code feature) to generate training set ```trainingData.csv```, and then run ```classify.py``` to use the classification algorithms.

## Issues

Please contact the authors or raise an issue on github if something seems broken or does not work, thanks!

aniket dot d at rice dot edu

vishwanath dot saragadam at rice dot edu
