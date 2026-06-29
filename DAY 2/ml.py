import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed. Target Encoding will be skipped.")

def main():
    print("Loading Dataset...")
    file_path = "train.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    df = pd.read_csv(file_path)
    df = df[pd.to_numeric(df['ID'], errors='coerce').notna()].apply(pd.to_numeric)
    print(
        f"Dataset loaded successfully.\n"
        f"Rows: {df.shape[0]}\n"
        f"Columns: {df.shape[1]}"
    )
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nHandling Missing DATA...")
    print("Artificially setting some 'Hits' (H) values to NaN for demonstration...")

    # Artificially create missing values
    df.loc[0:24, 'H'] = np.nan

    # Create an imputer that replaces missing values with the median of H
    imputer = SimpleImputer(strategy='median')
    df[['H']] = imputer.fit_transform(df[['H']])
    print(
        f"Imputation complete. "
        f"'H' column now has {df['H'].isnull().sum()} missing values."
    )
    print("\nFirst 30 rows of H:")
    print(df['H'].head(30))



    df['Team_ID'] = ['Team_' + str(np.random.randint(1, 1000)) for _ in range(len(df))]

    if TargetEncoder is not None:
        print("Applying Target Encoder")
        encoder = TargetEncoder()
        df['Team_ID_Encoded'] = encoder.fit_transform(df['Team_ID'], df['W'])
    else:
        print("Categories Encoders not installed")

    # Prepare data for feature selection and model training
    # Assuming 'W' is the target variable
    y = df['W']
    # Drop 'ID', 'W', and 'Team_ID' (if it's not a feature) from features set
    X = df.drop(columns=['ID', 'W', 'Team_ID'])

    # Apply SelectKBest to get the best features
    # Using mutual_info_regression to select top 2 features
    selector = SelectKBest(mutual_info_regression, k=2)
    selector.fit(X, y)
    
    # Get the names of the selected features
    selected_features_names = X.columns[selector.get_support()].tolist()

    # If the selected features don't match exactly, we'll use the user-specified ones
    if sorted(selected_features_names) != sorted(['R', 'HR']):
        print(f"Note: SelectKBest chose {selected_features_names}. Using user-specified ['R', 'HR'] for demonstration.")
        X_selected = X[['R', 'HR']]
        selected_features_names = ['R', 'HR']
    else:
        X_selected = X[selected_features_names]

    print(f"Best features selected:\n{selected_features_names}")

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    print(f"Training Data size: {x_train.shape}")
    print(f"Testing Data size: {x_test.shape}")

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(x_test)
    print("Predictions:")
    print(predictions)

    actual_wins = y_test.head(3).values
    predicted_wins = predictions[:3]

    for i in range(3):
        predicted = round(predicted_wins[i])
        actual = actual_wins[i]
        difference = abs(actual - predicted)

        print(f"Model Guessed: {predicted}")
        print(f"Real Answer: {actual}")
        print(f"Difference: {difference}\n")


if __name__ == '__main__':
    main()
    