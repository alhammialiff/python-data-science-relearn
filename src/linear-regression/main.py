import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Random forest imports
from sklearn.ensemble import RandomForestRegressor

# Load your data
PLAYER_VALUATION_DATASET_PATH = './data/player_valuations.csv'
PLAYER_DATASET_PATH = './data/players.csv'
PLAYER_APPEARANCE_DATASET_PATH = './data/appearances.csv'

""" Step 1: Data Preparation """
def prepare_data():
    
    playerDf = pd.read_csv(PLAYER_DATASET_PATH)
    playerValuationDf = pd.read_csv(PLAYER_VALUATION_DATASET_PATH)
    playerAppearanceDf = pd.read_csv(PLAYER_APPEARANCE_DATASET_PATH)

    # Aggregate goals by player
    player_goals = playerAppearanceDf.groupby("player_id").agg({
        "goals": "sum",
        "assists": "sum",
        "minutes_played": "sum"
    }).reset_index()    

    player_goals.columns = ["player_id", "total_goals", "total_assists", "total_minutes"]
    
    # INNER JOIN to ensure players that exists in both datasets are kept only
    df = pd.merge(playerDf, playerValuationDf, on="player_id", how="inner")

    # LEFT JOIN ensures all players are kept (even those without appearance data)
    df = pd.merge(df, player_goals, on="player_id", how="left")

    # Note: Look at the columns before feature engineering is performed
    print(print(f"\ndf before feature engineering:"))
    print(df.describe)

    # Fill NaN with 0 (players with no appearance records)
    df["total_goals"] = df["total_goals"].fillna(0)
    df["total_assists"] = df["total_assists"].fillna(0)
    df["total_minutes"] = df["total_minutes"].fillna(0)
    
    # Create better rate-based features
    # Smoothing technique: total_minutes is added with 1 so that we don't get division-by-zero error
    df["goals_per_90"] = (df["total_goals"] / (df["total_minutes"] + 1)) * 90
    df["assists_per_90"] = (df["total_assists"] / (df["total_minutes"] + 1)) * 90
    df["goal_contributions_per_90"] = df["goals_per_90"] + df["assists_per_90"]
    
    # Career productivity indicators (better than is_elite binary)
    df["career_games"] = (df["total_minutes"] / 90).astype(int)
    df["has_experience"] = (df["career_games"] > 50).astype(int)

    # Convert date and extract year
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    # df["age"] = 2026 - df["date_of_birth"].dt.year

    df = df.dropna(subset=["date","date_of_birth"])

    df["age_at_valuation"] = ((df["date"] - df["date_of_birth"]).dt.days / 365.25).astype(int)
    
    # Age-based features (peak performance years)
    # By squaring (polynomial feature), age can be used to capture non-linear rs with market value
    df["age_squared"] = df["age_at_valuation"] ** 2
    df["is_prime_age"] = ((df["age_at_valuation"] >= 24) & (df["age_at_valuation"] <= 29)).astype(int)



    """ Feature Engineering """
    features = [
        "age_at_valuation",
        "age_squared",
        "is_prime_age",
        "height_in_cm", 
        "goals_per_90",
        "assists_per_90",
        "goal_contributions_per_90",
        "career_games",
        "has_experience",
        "market_value_in_eur_y"
    ]

    """ One-hot encode positions and add them into features """
    # df = df[["name","year","age","height_in_cm","market_value_in_eur_y"]].dropna()
    if "position" in df.columns:

        # Create encoded columns for target column "position"
        df = pd.get_dummies(df, columns=["position"], prefix="pos")

        # Add encoded columns into the array of features
        features.extend([col for col in df.columns if col.startswith("pos_")])

    # Keep full dataframe with all columns before filtering
    full_df = df.copy()
    
    # Select only features for training
    # Why player_id or name aren't included in the training features
    #   player_id is an arbitrary identifier with no predictive value. Including it would cause severe problems:
    #       1. No meaningful information - It's just a random number/code assigned to each player
    #       2. Causes overfitting - The model would memorize specific IDs instead of learning patterns
    #       3. Can't generalize - Model couldn't predict values for new players with different IDs
    #       4. Breaks the purpose - You want to predict based on characteristics (age, performance, position), not identity
    df = df[features].dropna()

    # Check if valuations are varied
    print(f"\nMarket Value Stats:")
    print(df["market_value_in_eur_y"].describe())   

    print("\nFeature Correlations with Market Value:")


    # Select only numeric columns and exclude non-numeric columns like strings ("name", "positions")
    numeric_df = df.select_dtypes(include=[np.number]) 
    print(numeric_df.corr()["market_value_in_eur_y"].sort_values(ascending=False))

    # Filter outliers
    numeric_df = df[df["market_value_in_eur_y"] < 150_000_000]
    
    
    print("\n === Pre-processed Data ===")
    print(numeric_df)

    return numeric_df, full_df


""" Step 2a: Train Linear Regression Model """
def train_model(df):


    # Features (X) and Target (y)
    # X = df[["year", "age", "height_in_cm"]]
    X = df.drop("market_value_in_eur_y", axis=1)
    
    y = np.log1p(df["market_value_in_eur_y"])
    
    # Applying log transformation to handle right-skewed target variable
    # Log transformation prevents disproportionate influence from high-magnitude values, 
    # ensuring the model learns fairly from the entire dataset.
    np.log1p(df["market_value_in_eur_y"]) 

    # Split data: 80% and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred

""" Step 2b: Train Random Forest Model """
def train_random_forest(df):
    """
    Note:
    Train Random Forest - handles non-linear relationships better than Linear Regression
    No need for log transformation as it naturally handles outliers
    """

    # Features (X) and Target (y)
    X = df.drop("market_value_in_eur_y", axis=1)
    y = df["market_value_in_eur_y"] # Note: No log transformation needed

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.2, random_state=42
    )

    # Create and train Random Forest
    # n_estimators: number of tree (more = better but slower)
    # max_depth: prevents overfitting
    # random_state: for reproducibilty
    rf_model = RandomForestRegressor(
        n_estimators=100, # Build 100 decision trees
        max_depth=20, # Limit tree depth to prevent overfitting
        min_samples_split=5, # Minimum samples to split a node
        random_state=42, 
        n_jobs=-1 # Use all CPU cores for faster training
    )

    # Train Random Forest Model
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Return Model, X_train, X_test, y_train, y_test, y_pred
    return rf_model, X_train, X_test, y_train, y_test, y_pred


""" Step 3a: Evaluate Model Performance """
def evaluate_model(y_test, y_pred, model, X_train):


    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=" * 50)
    print("MODEL EVALUATE METRICS")
    print("=" * 50)

    print(f"R2 Score: {r2:.4f} (Higher is better, max 1.0)")
    print(f"RMSE: {rmse:,.2f} (Lower is better)")
    print(f"MAE: {mae:,.2f} (Average error)")

    for feature, coef in zip(X_train.columns, model.coef_):
        print(f" {feature}: {coef:,.2f}")
    
    print(f"   Intercept: {model.intercept_:,.2f}")
    print("=" * 50)

    print(f"\nPrediction Range: €{y_pred.min():,.0f} to €{y_pred.max():,.0f}")
    print(f"Actual Range: €{y_test.min():,.0f} to €{y_test.max():,.0f}")

""" Step 3b: Evaluate Random Forest Performance """
def evaluate_random_forest(y_test, y_pred, rf_model, X_train):
    """
    Evaluate Random Forest with metrics and feature importance
    """

    # Metrics (same as Linear Regression for comparison)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("RANDOM FOREST EVALUATION METRICS")
    print("=" * 50)

    print(f"R2 Score: {r2: .4f} (Higher is better, max 1.0)")
    print(f"RMSE: €{rmse:,.2f} (Lower is better)")
    print(f"MAE: €{mae:,.2f} (Average error)")

    # Random Forest does not have coef like Linear Regression
    # Instead, it has feature importance (how much each feature contributes)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)



    # Iterate first 10 of index and rows using iterrows()
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")



    print("=" * 50)
    
    print(f"\nPrediction Range: €{y_pred.min():,.0f} to €{y_pred.max():,.0f}")
    print(f"Actual Range: €{y_test.min():,.0f} to €{y_test.max():,.0f}")


"""Step 4: Visualize Results"""
def visualize_predictions(y_test, y_pred):


    plt.figure(figsize=(12,5))

    # Subplot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([y_test.min(),y_test.max()],
             [y_test.min(),y_test.max()],
             'r--', lw=2, label="Perfect Prediction")
    plt.xlabel('Actual Market Value (EUR)')
    plt.ylabel('Predicted Market Value (EUR)')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)


    # Subplot 2: Residuals
    plt.subplot(1, 2, 2)
    residual = y_test - y_pred
    plt.scatter(y_pred, residual, alpha=0.5, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Market Value (EUR)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


""" Step 4b: Compare Both Models Visually """
def compare_models_visualization(y_test_lr, y_pred_lr, y_test_rf, y_pred_rf):
    """
    Side-by-side comparison of Linear Regression vs Random Forest predictions
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear Regression Plot
    axes[0].scatter(y_test_lr, y_pred_lr, alpha=0.5, color='blue', edgecolors='black', linewidth=0.5)
    axes[0].plot([y_test_lr.min(), y_test_lr.max()], 
                 [y_test_lr.min(), y_test_lr.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Market Value (€)', fontsize=12)
    axes[0].set_ylabel('Predicted Market Value (€)', fontsize=12)
    axes[0].set_title(f'Linear Regression\nR² = {r2_score(y_test_lr, y_pred_lr):.4f}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Random Forest Plot
    axes[1].scatter(y_test_rf, y_pred_rf, alpha=0.5, color='green', edgecolors='black', linewidth=0.5)
    axes[1].plot([y_test_rf.min(), y_test_rf.max()], 
                 [y_test_rf.min(), y_test_rf.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Market Value (€)', fontsize=12)
    axes[1].set_ylabel('Predicted Market Value (€)', fontsize=12)
    axes[1].set_title(f'Random Forest\nR² = {r2_score(y_test_rf, y_pred_rf):.4f}', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Comparison plot saved as 'model_comparison.png'")


""" Step 4c: Error Distribution Comparison """
def compare_error_distribution(y_test_lr, y_pred_lr, y_test_rf, y_pred_rf):
    """
    Compare the distribution of prediction errors between models
    Shows which model makes more consistent predictions
    """
    
    # Calculate errors
    lr_errors = y_test_lr - y_pred_lr
    rf_errors = y_test_rf - y_pred_rf
    
    # 1 row, 2 columns, fig size of 16 inches by 6 inches
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear Regression Error Distribution
    axes[0].hist(lr_errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error (€)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Linear Regression Error Distribution\nMean Error: €{lr_errors.mean():,.0f}', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Random Forest Error Distribution
    axes[1].hist(rf_errors, bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error (€)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Random Forest Error Distribution\nMean Error: €{rf_errors.mean():,.0f}', 
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Error distribution plot saved as 'error_distribution_comparison.png'")
    
    # Print error statistics
    print("\n" + "=" * 50)
    print("ERROR STATISTICS COMPARISON")
    print("=" * 50)
    print(f"\nLinear Regression:")
    print(f"  Mean Error: €{lr_errors.mean():,.2f}")
    print(f"  Std Dev: €{lr_errors.std():,.2f}")
    print(f"  Max Overestimate: €{lr_errors.min():,.2f}")
    print(f"  Max Underestimate: €{lr_errors.max():,.2f}")
    
    print(f"\nRandom Forest:")
    print(f"  Mean Error: €{rf_errors.mean():,.2f}")
    print(f"  Std Dev: €{rf_errors.std():,.2f}")
    print(f"  Max Overestimate: €{rf_errors.min():,.2f}")
    print(f"  Max Underestimate: €{rf_errors.max():,.2f}")
    print("=" * 50)

def predict_salah_value(model, df, feature_columns):
    """Predict future value for a specific player"""
    
    # Check if player exists
    if "name" not in df.columns:
        print("\nSkipping player prediction - 'name' column not available in dataframe")
        return
    
    salah_data = df[df["name"] == "Mohamed Salah"]
    
    if salah_data.empty:
        print("\nMohamed Salah not found in dataset")
        return
    
    salah_latest = salah_data.sort_values("year").iloc[-1]
    
    # Create prediction data using CURRENT performance rates (not cumulative stats!)
    next_year_age = salah_latest["age_at_valuation"] + 1
    
    next_year_data = pd.DataFrame({
        "age_at_valuation": [next_year_age],
        "age_squared": [next_year_age ** 2],
        "is_prime_age": [1 if 24 <= next_year_age <= 29 else 0],
        "height_in_cm": [salah_latest["height_in_cm"]],
        "goals_per_90": [salah_latest["goals_per_90"]],  # Keep same performance rate
        "assists_per_90": [salah_latest["assists_per_90"]],
        "goal_contributions_per_90": [salah_latest["goal_contributions_per_90"]],
        "career_games": [salah_latest["career_games"] + 38],  # Add one season (~38 games)
        "has_experience": [salah_latest["has_experience"]]
    })
    
    # Add position columns
    for col in feature_columns:
        if col.startswith("pos_"):
            next_year_data[col] = [salah_latest[col] if col in salah_latest.index else 0]
    
    # Ensure columns are in the same order as training
    next_year_data = next_year_data[feature_columns]
    
    predicted_log_value = model.predict(next_year_data)[0]
    predicted_value = np.expm1(predicted_log_value)  # Reverse log transformation
    
    print(f"\nPREDICTION FOR MO SALAH ({salah_latest['year'] + 1}):")
    print(f"Current Age: {salah_latest['age_at_valuation']}, Next Year: {next_year_age}")
    print(f"Performance: {salah_latest['goals_per_90']:.2f} goals/90min, {salah_latest['assists_per_90']:.2f} assists/90min")
    print(f"Current Market Value: €{salah_latest['market_value_in_eur_y']:,.2f}")
    print(f"Predicted Market Value: €{predicted_value:,.2f}")
    print(f"Projected Change: €{predicted_value - salah_latest['market_value_in_eur_y']:,.2f}")

def main():
    
    # Execute data preparation (same for both models)
    df, alphanumeric_df = prepare_data()
    
    print("\n" + "🔵" * 25)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("🔵" * 25)
    
    # Train and evaluate Linear Regression
    lr_model, X_train_lr, X_test_lr, y_train_lr, y_test_lr, y_pred_lr = train_model(df)
    evaluate_model(y_test_lr, y_pred_lr, lr_model, X_train_lr)
    
    print("\n" + "🟢" * 25)
    print("TRAINING RANDOM FOREST MODEL")
    print("🟢" * 25)
    
    # Train and evaluate Random Forest
    rf_model, X_train_rf, X_test_rf, y_train_rf, y_test_rf, y_pred_rf = train_random_forest(df)
    evaluate_random_forest(y_test_rf, y_pred_rf, rf_model, X_train_rf)
    
    # Compare models numerically
    print("\n" + "⚖️" * 25)
    print("MODEL COMPARISON")
    print("⚖️" * 25)
    lr_r2 = r2_score(y_test_lr, y_pred_lr)
    rf_r2 = r2_score(y_test_rf, y_pred_rf)
    
    print(f"Linear Regression R²: {lr_r2:.4f}")
    print(f"Random Forest R²:     {rf_r2:.4f}")
    print(f"\nImprovement: {((rf_r2 - lr_r2) / lr_r2 * 100):+.2f}%")
    
    if rf_r2 > lr_r2:
        print("🏆 Random Forest performs better!")
    else:
        print("🏆 Linear Regression performs better!")
    
    # Visualize comparisons
    print("\n" + "📊" * 25)
    print("GENERATING VISUALIZATIONS")
    print("📊" * 25)
    
    compare_models_visualization(y_test_lr, y_pred_lr, y_test_rf, y_pred_rf)
    compare_error_distribution(y_test_lr, y_pred_lr, y_test_rf, y_pred_rf)
    
    # Individual model visualization (optional)
    # visualize_predictions(y_test_lr, y_pred_lr)
    
    # Predict for Salah using both models
    predict_salah_value(lr_model, alphanumeric_df, X_train_lr.columns.tolist())


if __name__ == "__main__":
    main()