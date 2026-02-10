import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

    # Fill NaN with 0 (players with no appearance records)
    df["total_goals"] = df["total_goals"].fillna(0)
    df["total_assists"] = df["total_assists"].fillna(0)
    df["total_minutes"] = df["total_minutes"].fillna(0)


    # Convert date and extract year
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    # df["age"] = 2026 - df["date_of_birth"].dt.year

    df = df.dropna(subset=["date","date_of_birth"])

    df["age_at_valuation"] = ((df["date"] - df["date_of_birth"]).dt.days / 365.25).astype(int)



    """ Feature Engineering """
    features = [
        "age_at_valuation", 
        "height_in_cm", 
        "total_goals",      # Add goals as feature
        "total_assists",    # Add assists as feature
        "total_minutes",    # Add minutes as feature
        "market_value_in_eur_y"
    ]

    """ One-hot encode positions and add them into features """
    # df = df[["name","year","age","height_in_cm","market_value_in_eur_y"]].dropna()
    if "position" in df.columns:
        df = pd.get_dummies(df, columns=["position"], prefix="pos")
        features.extend([col for col in df.columns if col.startswith("pos_")])

    df = df[features].dropna()

    # Check if valuations are varied
    print(f"\nMarket Value Stats:")
    print(df["market_value_in_eur_y"].describe())   

    print("\nFeature Correlations with Market Value:")
    print(df.corr()["market_value_in_eur_y"].sort_values(ascending=False))

    return df


""" Step 2: Train Linear Regression Model """
def train_model(df):


    # Features (X) and Target (y)
    # X = df[["year", "age", "height_in_cm"]]
    X = df.drop("market_value_in_eur_y", axis=1)
    
    y = df["market_value_in_eur_y"]
    
    # Applying log transformation to handle right-skewed target variable
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


""" Step 3: Evaluate Model Performance """
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


def predict_salah_value(model, df):

    salah_latest = df[df["name"] == "Mohamed Salah"].sort_values("year").iloc[-1]

    next_year_data = pd.DataFrame({
        "year": [salah_latest["year"] + 1],
        "age": [salah_latest["age"] + 1],
        "height_in_cm": [salah_latest["height_in_cm"]]
    })

    predicted_value = model.predict(next_year_data)[0]

    print(f"\nPREDICTION FOR MO SALAH ({salah_latest['year'] + 1}):")
    print(f"Predicted Market Value: €{predicted_value:,.2f}")

def main():

    # Execute pipeline
    df = prepare_data()
    model, X_train, X_test, y_train, y_test, y_pred = train_model(df)
    evaluate_model(y_test, y_pred, model, X_train)
    visualize_predictions(y_test, y_pred)
    predict_salah_value(model, df)

if __name__ == "__main__":
    main()