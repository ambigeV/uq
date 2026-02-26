All files contain a Pandas DataFrame with SMILES, mol, MorganFP and Outcome columns. You can use SMILES directly for featurization of choice, or use existing MorganFP (radius 2, fpSize 1024). The target variable is in the Outcome column.

Training set: HEK293_train_BM.pkl
Validation set: HEK293_test_BM.pkl
Test set: tox21_all.pkl

The train/ val split is done using scaffold splitting. The original set has 1:5 ratio of actives to inactives. The test set has about ~2500 molecules, no duplicates from training/ validation sets.