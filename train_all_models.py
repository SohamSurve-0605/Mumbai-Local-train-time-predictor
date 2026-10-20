import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    print("1. Loading Data...")
    # NOTE: Change 'matches.csv' to 'matches (2).csv' if that's your file name
    delivery_df = pd.read_csv('deliveries.csv')
    match_df = pd.read_csv('matches.csv')

    print("2. Merging, Filtering, and Sorting Data...")
    delivery_df = delivery_df.merge(match_df, left_on='match_id', right_on='id', how='left')
    # Filter for 2nd innings only
    delivery_df = delivery_df[delivery_df['inning'] == 2].copy()
    # Sort chronologically (CRITICAL for cumsum)
    delivery_df = delivery_df.sort_values(by=['match_id', 'over', 'ball']).reset_index(drop=True)

    print("3. Feature Engineering...")
    # Current Score (If your column is 'total_runs_y', change it here)
    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs'].cumsum()

    # Balls Left
    delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])

    # Wickets Remaining
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == "0" else 1).astype(int)
    wickets_fallen = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
    delivery_df['wickets'] = 10 - wickets_fallen

    # Current Run Rate (CRR)
    delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])

    # Clean Infinity and NaN values
    delivery_df['crr'] = delivery_df['crr'].replace([np.inf, -np.inf], 0).fillna(0)
    delivery_df = delivery_df.fillna(0)

    # Target Variable: Final Score
    delivery_df['final_score'] = delivery_df.groupby('match_id')['current_score'].transform('max')

    print("4. Preparing Data for Modeling...")
    cols_to_encode = ['batting_team', 'bowling_team', 'city']
    for col in cols_to_encode:
        delivery_df[col] = delivery_df[col].astype(str)

    features = ['batting_team', 'bowling_team', 'city', 'current_score', 'balls_left', 'wickets', 'crr']
    X = delivery_df[features]
    y = delivery_df['final_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first'), cols_to_encode)
    ], remainder='passthrough')

    print("5. Setting up Models and Hyperparameter Grids...")
    # Dictionary of models and their hyperparameter grids for tuning
    models = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'step2__n_estimators': [50, 100],
                'step2__max_depth': [10, 20, None]
            }
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'step2__max_depth': [5, 10, 20, None]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, n_jobs=-1),
            'params': {
                'step2__n_estimators': [50, 100],
                'step2__learning_rate': [0.05, 0.1]
            }
        }
    }

    print("\n6. Tuning, Training, and Evaluating Models...")
    for name, config in models.items():
        print(f"\n--- Tuning {name} ---")

        # Create the pipeline
        pipe = Pipeline(steps=[('step1', trf), ('step2', config['model'])])

        # Apply GridSearchCV for Hyperparameter Tuning
        # cv=3 means 3-fold cross-validation, scoring='r2' optimizes for R-squared
        grid = GridSearchCV(pipe, config['params'], cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Get the best model from the grid search
        best_model = grid.best_estimator_

        # Evaluate on the test set
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ Best Params for {name}: {grid.best_params_}")
        print(f"✅ {name} MAE: {mae:.2f} runs")
        print(f"✅ {name} R2 Score: {r2:.4f}")

        # Save the best tuned model
        filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"   Saved as {filename}")

    print("\n✅ ALL MODELS TUNED, TRAINED, AND SAVED SUCCESSFULLY!")


if __name__ == "__main__":
    main()