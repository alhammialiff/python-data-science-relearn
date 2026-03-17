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


class MolecularSolubilityPredictor:

    # Constructor
    def __init__(self, data):
        
        self.data = data

        # These get populated as the pipeline runs
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.compound_id_train = None
        self.compound_id_test = None

        self.best_model = None

        self.y_pred = None


    '''
    Performs structural analysis and investigating missing values of dataset

        Args:
            rawData (DataFrame): The raw dataset containing SMILES strings and solubility values.

        Returns:
            Refined Dataframe
    '''
    def performStucturalAnalysisAndMissingValues(self, rawData):

        # 1. Check the structural information of the dataset
        print("Dataset info")

        rawData.info()

        print("\n" + "=" * 60 + "\n")

        # 2. Check for missing values in each column . Output each index (feature) and the sum of its missing values
        print("Missing Values per Column:")
        missingValue = rawData.isnull().sum()

        # [Note] To refactor this to synthesize missing data if missingValue is more than 5% of dataset, or otherwise delete the rows
        if missingValue.sum() > 0:

            print(missingValue[missingValue > 0])

            # Placeholder to synthesize missing values
            ###
        
        else:
            print("There are no missing vaues in the dataset")
        
        # 4. Get statistical summary of numerical columns
        print("Statistical Summary of Numerical Columns")

        # 5. Tranpose it to make the output easier to read
        print(rawData.describe().T)

        return rawData



    '''
    Investigate distribution of categorical variables (are the features evenly distributed or skewed?)
    '''
    def investigateCategoricalVariableDistribution(self, refinedDf):

        # Select categorical columns
        categoricalColumns = refinedDf.select_dtypes(include=["object"]).columns

        print("Categorical Columns:")
        print(categoricalColumns.tolist())
        print("\n" + "=" * 60 + "\n")

        # Display the top 5 most frequent values for some key categorical  columns
        keyCategoricalColumns = ["Compound ID", "SMILES"]

        for col in keyCategoricalColumns:

            print(f"--- Top 5 Categories in '{col} ---")

            print(refinedDf[col].value_counts().head(5))

            print("\n")


    
    """
        Converts a SMILES string into a Morgan fingerprint (bit vector).
        
        SMILES (Simplified Molecular Input Line Entry System) is a text notation
        representing the structure of a molecule.

        Morgan fingerprints encode the local chemical environment of each atom
        as a fixed-length binary vector, useful for ML models.

        Args:
            smiles (str): A SMILES string representing a molecule.

        Returns:
            list: A list of 1024 bits (0s and 1s) representing the molecule's fingerprint.
    """
    def smilesToFingerprintPlusPhysicochemFeatures(self, smiles):

        # Convert the SMILES string into an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)

        # Create a Morgan fingerprint generator with:
        # - radius = 2 (captures atom neighborhoods up to 2 bonds away)
        # - fpSize = 1024 (output vector length)
        generator = GetMorganGenerator(radius = 3, fpSize = 2048)
        
        # Generate the fingerprint and convert to a plain Python list
        fingerprint = list(generator.GetFingerprintAsNumPy(mol))

        # [Learning - Molecular Fingerprint]
        # In our context, a Molecular Fingerprint is a fixed-length binary vector (1024 bits) 
        # that encodes the structural features of a molecule.
        # Each bit represents whether a particular substructure or chemical pattern exists (1) or doesn't exist (0) in the molecule.
        # For example, given the SMILES CCO (ethanol):
        # print("Fingerprint (numpy)")
        # print(f"{fingerprint[:20]} \n")
        
        # [Feature Engineering] Append physicochemical descriptors to fingerprint to give it more features
        descriptors = [
            Descriptors.MolWt(mol),           # Molecular weight
            Descriptors.MolLogP(mol),         # Lipophilicity (known to correlate with solubility)
            Descriptors.NumHDonors(mol),      # Hydrogen bond donors
            Descriptors.NumHAcceptors(mol),   # Hydrogen bond acceptors
            Descriptors.TPSA(mol),            # Topological polar surface area
            Descriptors.NumRotatableBonds(mol) # Molecular flexibility
        ]

        # Combine fingerprint + descriptors into one feature vector
        return fingerprint + descriptors



    def performDuplicateAnalysis(self, refinedData):

        duplicates = refinedData[refinedData["Compound ID"].duplicated(keep=False)]

        print("\n" + "=" * 60)
        print("\nFind Duplicates...")
        print(duplicates[["Compound ID", "SMILES", "measured log(solubility:mol/L)"]])
        print("\n" + "=" * 60 + "\n")

        if(len(duplicates) > 1):

            # Average the solubility values of duplicated rows
            fixedDupe = duplicates.groupby(
                ["Compound ID", "SMILES"], as_index=False
            )["measured log(solubility:mol/L)"].mean()

            # Step 1: Remove All duplicate rows from the original DF
            cleanedData = refinedData[~refinedData["Compound ID"].duplicated(keep=False)]

            # Step 2: Append the averaged (fixed) row back in 
            cleanedData = pd.concat([cleanedData, fixedDupe], ignore_index = True)

            # Debug
            print(f"Fixed {len(duplicates)} duplicate rows → merged into {len(fixedDupe)} rows")
            print("Cleaned dataset shape:", cleanedData.shape)

            return cleanedData
        

        return refinedData



    """
        Prepares features (X) and labels (y) from the raw data,
        then splits them into training and testing sets.

        Args:
            refinedData (DataFrame): The refined dataset containing SMILES strings and solubility values.

        Returns:
            tuple: X_train, X_test, y_train, y_test splits.
    """
    def performTrainTestSplit(self, refinedData):

        # Input: Molecular fingerprint
        X = refinedData["SMILES"].apply(self.smilesToFingerprintPlusPhysicochemFeatures)
        X = list(X)

        # Target: Solubility
        y = refinedData["measured log(solubility:mol/L)"]

        X_train, X_test, y_train, y_test, compound_id_train, compound_id_test = train_test_split(X, y, refinedData['Compound ID'], test_size=0.2, random_state=42)

        return X, y, X_train, X_test, y_train, y_test, compound_id_train, compound_id_test

    """
        Trains a Random Forest Regressor on the training data and evaluates it
        on the test data.

        Random Forest is an ensemble of decision trees that averages their
        predictions to improve accuracy and reduce overfitting.

        Args:
            X_train: Training feature vectors.
            X_test:  Testing feature vectors.
            y_train: Training labels (solubility values).
            y_test:  Testing labels (solubility values).

        Returns:
            bestModel: The best Random Forest Regressor model found in Grid Search Cross Validation Training.
            y_pred: Target Predictions
    """
    def modelTrainingGscv(self, X_train, X_test, y_train, y_test):

        # 1. Define the parameter grid - all combinations will be tested
        paramGrid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 10, 15, 20, 25, None],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_features': ['sqrt', 'log2']
        }

        # 2. Instantiate model
        model = RandomForestRegressor(
            random_state=42
        )

        # 3. Use GridSearchCV to try every combination using 5-fold-CV
        #    n_jobs=-1 uses all CPU cores to speed up the search
        gridSearch = GridSearchCV(
            estimator=model,
            param_grid=paramGrid,
            scoring='r2',
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        startTime = time.time()

        print(f"="*60 + "\n")
        print(f"\nBegin GSCV Training...\n")

        # 4. Fit using Grid Search (Long process)
        gridSearch.fit(X_train, y_train) 
        # model.fit(X_train, y_train)

        endTime = time.time() - startTime

        print(f"\nTraining duration: {endTime}s")

        # 5. Retrieve the best model and parameters found
        bestModel = gridSearch.best_estimator_
        print("\nBest Parameters Found:")
        print(gridSearch.best_params_)


        print("\nBest Model Test Score: ", bestModel.score(X_test, y_test))
        # print("Model Score: ", model.score(X_test, y_test))

        # y_pred = model.predict(X_test)
        y_pred = bestModel.predict(X_test)

        return bestModel, y_pred



    """
        Trains a Random Forest Regressor on the training data and evaluates it
        on the test data.

        Random Forest is an ensemble of decision trees that averages their
        predictions to improve accuracy and reduce overfitting.

        Args:
            X_train: Training feature vectors.
            X_test:  Testing feature vectors.
            y_train: Training labels (solubility values).
            y_test:  Testing labels (solubility values).

        Returns:
            bestModel: The best Random Forest Regressor model found in Randomized Search Cross Validation Training.
            y_pred: Target Predictions
    """
    def modelTrainingRscv(self, X_train, X_test, y_train, y_test):

        # 1. Define the parameter grid - all combinations will be tested
        paramGrid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 10, 15, 20, 25, None],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_features': ['sqrt', 'log2']
        }

        # 2. Instantiate model
        model = RandomForestRegressor(
            random_state=42
        )

        # 3. Use RandomizerSearchCV to try every combination using 5-fold-CV
        #    n_jobs=-1 uses all CPU cores to speed up the search
        # Only tries 20 random combinations instead of all 96
        randomSearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=paramGrid,
            n_iter=20,          # Number of random combinations to try
            scoring='r2',
            cv=5,
            n_jobs=-1,
            random_state=42
        )

        startTime = time.time()


        print(f"="*60 + "\n")
        print(f"\nBegin RSCV Training...\n")


        # 4. Fit using Grid Search (Long process)
        randomSearch.fit(X_train, y_train) 
        # model.fit(X_train, y_train)

        endTime = time.time() - startTime

        print(f"\nTraining duration: {endTime}s")

        # 5. Retrieve the best model and parameters found
        bestModel = randomSearch.best_estimator_
        print("\nBest Parameters Found:")
        print(randomSearch.best_params_)


        print("\nBest Model Test Score: ", bestModel.score(X_test, y_test))
        print(f"="*60 + "\n")

        # print("Model Score: ", model.score(X_test, y_test))

        # y_pred = model.predict(X_test)
        y_pred = bestModel.predict(X_test)

        return bestModel, y_pred



    '''
        Evaluate model performance more accurately by performing K-Fold Cross Valudation on X and y

        Args:
            X: Features
            y: Target
            y_test: Test labels
            y_pred: Predicted labels
            trainModel: Trained Model    
    '''
    def evaluateModelPerformance(self, y_test, y_pred):

        # print("\n" + "=" * 60 + "\n")
        # print("Running 5-Fold Cross-Validation...\n")
        

        # Obtain Cross-Validation Scores
        # cvScores = cross_val_score(trainedModel, X, y, scoring='r2', n_jobs=-1)

        # print(f"Cross-Validation R² Scores : {np.round(cvScores, 4)}")
        # print(f"Average CV R² Score: {cvScores.mean():.4f}")
        # print("-" * 60)

        # Set residual (y_test - y_pred)
        residuals = y_test - y_pred

        # Plot Residual and Predictions
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color='teal')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

        plt.title('Residual Analysis: Predicted Molecular Solubility vs Errors', fontsize=14)
        plt.xlabel('Predicted Solubility',fontsize = 12)
        plt.ylabel('Residuals / Error (mol/L)', fontsize = 12)

        plt.tight_layout()
        plt.show()



    '''
        Print model performance by listing target vs predicted

        Args:
            y_test: Test labels
            y_pred: Predicted labels 
    '''
    def printTargetVsPredictedResult(self, y_test, y_pred, compound_id_test):

        # Create a comparison DataFrame
        comparison_df = pd.DataFrame({
            'Compound ID': compound_id_test.values,
            'Actual Solubility':    y_test,
            'Predicted Solubility': y_pred,
            'Residual (Error)':     y_test - y_pred,
            'Abs Error':            np.abs(y_test - y_pred)
        })

        # Rounding for readability
        comparison_df = comparison_df.round(4)

        # Sort by absolute error to see worst predictions first
        comparison_df_sorted = comparison_df.sort_values('Abs Error', ascending=False)

        print("=" * 65)
        print(f"{'Compound':>15} {'Actual':>15} {'Predicted':>15} {'Residual':>15} {'Abs Error':>12}")
        print("=" * 65)
        print(comparison_df_sorted.to_string(index=True))
        print("=" * 65)
        print(f"Mean Absolute Error : {comparison_df['Abs Error'].mean():.4f}")
        print(f"Max Error           : {comparison_df['Abs Error'].max():.4f}")
        print(f"Min Error           : {comparison_df['Abs Error'].min():.4f}")


    '''
        Run the full pipeline in sequence
    '''
    def run(self):

        # 2. Perform Structural Analysis and Handling Missing Values
        self.data = self.performStucturalAnalysisAndMissingValues(self.data)

        # 3. Find rows where Compound ID is duplicated
        self.data = self.performDuplicateAnalysis(self.data)

        # 4. Investigate distribution of catergorical variables
        self.investigateCategoricalVariableDistribution(self.data)

        # 5. Perform train test split
        self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test, self.compound_id_train, self.compound_id_test = self.performTrainTestSplit(self.data)
        
        # 6.1 Model Training (With RandomizerSearchCV - faster)
        self.bestTrainedRscvModel, self.y_pred = self.modelTrainingRscv(self.X_train, self.X_test, self.y_train, self.y_test)
        
        # 6.2 Model Training (With GridSearchCV - slower)
        # bestTrainedGscvModel, y_pred = modelTrainingGscv(X_train, X_test, y_train, y_test)

        # 7. Evaluate Performance of two models
        self.evaluateModelPerformance(self.y_test, self.y_pred)
        # evaluateModelPerformance(X, y, y_test, y_pred, bestTrainedGscvModel)

        self.printTargetVsPredictedResult(self.y_test, self.y_pred, self.compound_id_test)