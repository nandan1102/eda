
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# category_encoders is needed for Target Encoding
try:
    # pyrefly: ignore [missing-import]
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
    # Remove duplicate header rows or corrupted lines, then convert columns to numeric
    df = df[pd.to_numeric(df['ID'], errors='coerce').notna()].apply(pd.to_numeric)
    print(
        f"Dataset loaded successfully.\n"
        f"Rows: {df.shape[0]}\n"
        f"Columns: {df.shape[1]}"
    )
    print("\nColumn names:")
    print(df.columns.tolist())
    
    # Handling Missing Values demonstration
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



    df['Team_ID'] = ['Team_' + str(np.random.randint(1,1000)) for _ in range(len(df))]

    if TargetEncoder is not None:
        print("Applying Target Encoder")
        encoder = TargetEncoder()
        df['Team_ID_Encoded'] = encoder.fit_transform(df['Team_ID'], df['W'])
    else:
        print("Categories Encoders not installed")

    # Feature Selection
    features_to_test = ['R', 'HR', 'SO', 'SB']
    if 'Team_ID_Encoded' in df.columns:
        features_to_test.append('Team_ID_Encoded')

    x_features = df[features_to_test].fillna(0)
    Y_target = df['W']

    selector = SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(x_features, Y_target)
    winning_features = selector.get_support()
    best_features = x_features.columns[winning_features].tolist()

    print("\nBest features selected:")
    print(best_features)


if __name__ == "__main__":
    main()
    