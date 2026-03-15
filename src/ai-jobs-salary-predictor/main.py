'''
Machine Learning Practice
Materials: https://www.kaggle.com/code/realtalhacelik/ai-salary-predictor-0-93-r
Credits: Talha Sage

'''


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

import time


'''
Load Dataset

Return:
    Dataframe of raw dataset
'''
def loadDataset():

    # List down potential paths that contains global_ai_jobs.csv
    candidatePaths = [
        Path(__file__).resolve().parent / "data" / "global_ai_jobs.csv", # src/ai-jobs-salary-predictor/data
        Path(__file__).resolve().parents[2] / "data" / "global_ai_jobs.csv" # python_refresher/data
    ]
    
    # Return the filepath that exists
    filePath = next((p for p in candidatePaths if p.exists()), None)

    if filePath is None:
        raise FileNotFoundError(
            "global_ai_jobs.csv not found. Checked:\n- " + "\n- ".join(str(p) for p in candidatePaths)
        )

    # Read CSV
    rawDf = pd.read_csv(filePath)

    print("Data loading completed \n" + "-" * 40)
    print(f"Number of rows: {rawDf.shape[0]}") 
    print(f"Number of columns: {rawDf.shape[1]}")
    print("-" * 40)

    print(rawDf.head())

    return rawDf


'''
Performs structural analysis and investigating missing values of dataset

Return:
    Refined Dataframe
'''
def performStructuralAnalysisAndMissingValues(rawDf):

    # 1. Check the structural information of the dataset
    print("Dataset info")
    rawDf.info()
    print("\n" + "="*60 + "\n")

    # 2. Check for missing values in each column. Output each index (features) and the sum of its missing values.
    print("Missing Values per Column:")
    missingValues = rawDf.isnull().sum()

    # 3. Display columns that only have missing values if any
    if missingValues.sum() > 0:

        print(missingValues[missingValues > 0])

        # Perform data cleaning here if there are missing values
        ###

    else:

        print("There are no missing values in the dataset")

    print("\n" + "="*60 + "\n")

    # 4. Get statistical summary of numerical columns
    print("Statistical Summary of Numerical Columns:")

    # 5. Transpose it to make the output easier to read
    print(rawDf.describe().T)

    return rawDf


'''
Investigate distribution of categorical variables (are the features evenly distributed or skewed?)
'''
def investigateCategoricalVariableDistribution(refinedDf1):

    # Select categorical columns
    categoricalColumns = refinedDf1.select_dtypes(include=["object"]).columns

    print("Categorical Columns:")
    print(categoricalColumns.tolist())
    print("\n" + "="*60 + "\n")

    # Display the top 5 most frequent values for some key categorical columns
    keyCategoricalColumns = ["country", "job_role", "ai_specialization", "experience_level", "education_required", "work_mode"]

    for col in keyCategoricalColumns:
        
        print(f"--- Top 5 Categories in '{col}' ---")

        # value_counts() calculates frequencies, head(5) gets the top 5
        print(refinedDf1[col].value_counts().head(5))

        print("\n")

'''
Investigate the overall distribution using Histogram, and how salary varies with experience level

'''
def visualisingTargetVariable(refinedDf1):

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16,6))

    # 1. Histogram: Distribution of Salary
    sns.histplot(refinedDf1["salary_usd"], bins=100, kde=True, color="skyblue", ax=axes[0])
    axes[0].set_title("Distribution of Salary in USD", fontsize=14)

    # 2. Boxplot: Salary vs Experience Level
    # We define the order to show progression logically
    exp_order = ["Entry","Mid","Senior","Lead"]
    sns.boxplot(x="experience_level",y="salary_usd", data=refinedDf1, order=exp_order, palette="Set2", ax=axes[1])
    axes[1].set_title("Salary Distribution by Experience Level", fontsize=14)
    axes[1].set_xlabel("Experience Level", fontsize=12)
    axes[1].set_ylabel("Salary (USD)", fontsize=12)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()


'''
Investigate how our numerical features relate to each other, especially to our target variable salary_usd

'''
def performCorrelationMatrixAndHeatmap(refinedDf1):

    # Prepare columns for analysis
    columnsForCorrelationAnalysis = [
        "salary_usd", "bonus_usd", "experience_years", "weekly_hours",
        "skill_demand_score", "company_rating", "employee_satisfaction",
        "work_life_balance_score", "career_growth_score"
    ]

    # Calculate the correlation matrix
    correlationMatrix = refinedDf1[columnsForCorrelationAnalysis].corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10,8))

    # Draw the heatmap using seaborn
    # annot=True show the correlation values, cmap chooses the color scheme
    sns.heatmap(correlationMatrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=.5)

    # Add a title
    plt.title("Correlation Heatmap of Key Variables")

    # Show the plot
    plt.show()


'''
While YoE pays well, what about location and job title?
'''
def investigateSalaryBasedOnCountryAndRole(refinedDf1):

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14,12))

    # 1. Average Salary by Job Role
    # Group by job_role, calculation mean of salary_usd, and sort values
    salaryByRole = refinedDf1.groupby('job_role')["salary_usd"].mean().sort_values(ascending=False)
    sns.barplot(x=salaryByRole.values, y=salaryByRole.index, palette='viridis', ax=axes[0])
    axes[0].set_title("Average Salary by Job Role", fontsize=14)
    axes[0].set_xlabel("Average Salary (USD)", fontsize=12)
    axes[0].set_ylabel("Job Role", fontsize=12)

    # 2. Average Salary by Country
    salaryByCountry = refinedDf1.groupby("country")["salary_usd"].mean().sort_values(ascending=False)
    sns.barplot(x=salaryByCountry.values, y=salaryByCountry.index, palette="magma", ax=axes[1])
    axes[1].set_title("Average Salary by Country", fontsize=14)
    axes[1].set_title("Average Salary (USD)", fontsize=12)
    axes[1].set_ylabel("Country", fontsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.show()


'''
Perform pre-processing by encoding features (to ordinal/nominal)
Return:
    Resultant Model for Model Fit
'''
def dataPreprocessing(refinedDf1):

    # 1. Drop useless columns
    dfModel = refinedDf1.drop(columns=['id'])

    # 2. Ordinal encoding for Experience Level
    expMapping = {
        "Entry": 1,
        "Mid": 2,
        "Senior": 3,
        "Lead": 4
    }
    dfModel["experience_level"] = dfModel["experience_level"].map(expMapping)

    # 3. One-hot Encoding for nominal categorical variables
    # drop_first=True prevents the dummy variable trap
    categoricalToEncode = [
        "country", "job_role", "ai_specialization",
        "education_required", "industry", "company_size", "work_mode"
    ]

    dfModel = pd.get_dummies(dfModel, columns=categoricalToEncode, drop_first=True)

    print(f"Old dataset shape: {refinedDf1.shape}")
    print(f"New dataset shape: {dfModel.shape}")
    print("-" * 60)

    print(dfModel.head())

    return dfModel

'''
Perform train-test-split and fit model
'''
def splitDatasetAndFitModel(dfModel):

    # 1. Define Features (X) and Target (y)
    # X contains all columns except "salary_usd" 
    X = dfModel.drop(columns=["salary_usd"])

    # y is out target variable
    y = dfModel["salary_usd"]

    # 2. Train-Test Split
    # 80% fot training, %20 for testing
    Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size= 0.2, random_state= 42)
    
    print(f"Training data shape: {Xtrain.shape}")
    print(f"Testing data shape: {Xtest.shape}")
    print("-" * 50)

    # 3. Initialize and Train the model
    # n_jobs = -1 uses all CPU cores for faster training
    print("Training Random Forest Model with prepared data")

    rfModel = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    startTime = time.time()
    rfModel.fit(Xtrain, yTrain)
    endTime = time.time() - startTime

    print(f"Training time taken: {endTime:.2f} sec")

    # 4. Make Predictions on the test set
    yPred = rfModel.predict(Xtest)

    # 5. Evaluate the model
    mae = mean_absolute_error(yTest, yPred)
    rmse = np.sqrt(mean_squared_error(yTest, yPred))
    r2 = r2_score(yTest, yPred)

    print("\n" + "="*50)
    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")
    print(f"R-Squared: {r2:.4f}")
    print("="*50)

    return X, y, Xtest, yTest, yPred, mae, rmse, r2, rfModel

'''
Evaluate Model Performance more accurately by going through K-Fold CV
'''
def evaluateModelPerformance(X, y, yTest, yPred, rfModel):

    print("Running 5-Fold Cross-Validation...")

    # cv=5 means 5 different splits
    cvScores = cross_val_score(rfModel, X, y, cv=5, scoring='r2', n_jobs=-1)

    print(f"Cross-Validation R² Scores : {np.round(cvScores, 4)}")
    print(f"Average CV R² Score: {cvScores.mean():.4f}")
    print("-" * 60)

    # Set residual
    residuals = yTest - yPred

    # Plot Residual and Predictions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=yPred, y=residuals, alpha=0.5, color='teal')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

    plt.title('Residual Analysis: Predicted Salaries vs Errors', fontsize=14)
    plt.xlabel('Predicted Salary (USD)', fontsize=12)
    plt.ylabel('Residuals / Errors (USD)', fontsize=12)

    plt.tight_layout()
    plt.show()

def main():

    # Data Loading
    rawDf = loadDataset()

    # EDA - 1. Structural Analysis and Missing Values
    refinedDf1 = performStructuralAnalysisAndMissingValues(rawDf=rawDf) 

    # EDA - 2. Categorical Variable Distribution
    investigateCategoricalVariableDistribution(refinedDf1)

    # EDA - 3. Visualizing Target Variable
    visualisingTargetVariable(refinedDf1)

    # EDA - 4. Correlation Matrix and Heatmap
    performCorrelationMatrixAndHeatmap(refinedDf1)

    # EDA - 5. Salary Deep Dive - Country and Role
    investigateSalaryBasedOnCountryAndRole(refinedDf1)

    # DPP - 1. Preprocessing and Feature Encoding
    dfModel = dataPreprocessing(refinedDf1)

    # Split and Fit
    X, y, Xtest, yTest, yPred, mae, rmse, r2, rfModel = splitDatasetAndFitModel(dfModel)

    # Post-fitting model performance evaluation
    evaluateModelPerformance(X, y, yTest, yPred, rfModel)



if __name__ == "__main__":
    main()