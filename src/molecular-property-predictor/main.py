import glob
import os
import time

import kagglehub
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split

from MolecularSolubilityPredictor import MolecularSolubilityPredictor

"""
    Downloads the Delaney solubility dataset from Kaggle and loads it into a DataFrame.
    Returns a pandas DataFrame containing the raw dataset.
"""
def loadDataset():

    dirPath = kagglehub.dataset_download("yeonseokcho/delaney")

    print("Path to dataset files:", dirPath)

    csvFiles = glob.glob(os.path.join(dirPath, "*.csv"))
    if not csvFiles:
        raise FileNotFoundError(f"No CSV files found in {dirPath}")

    rawData = pd.read_csv(csvFiles[0])

    print("Data loading completed \n" + "-" * 40)
    print(f"Number of row: {rawData.shape[0]}")
    print(f"Number of column: {rawData.shape[1]}")

    print(rawData.head())

    return rawData

def main():

    # 1. Load Dataset
    rawData = loadDataset()

    # 2. Run predictor pipeline
    predictor = MolecularSolubilityPredictor(data = rawData)
    predictor.run()

    


if __name__ == "__main__":
    
    main()

