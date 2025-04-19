import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    Encapsulates preprocessing steps for:
      - Logistic Regression: numeric feature selection
      - Traditional ML: one-hot encoding + alignment
      - Deep Learning: one-hot encoding + alignment + scaling
    """
    def __init__(self, reference_path: str):
        # Load reference DataFrame
        df_ref = pd.read_csv(reference_path)
        if 'TARGET' in df_ref:
            df_ref = df_ref.drop(columns=['TARGET'])

        # Numeric feature names (for logistic)
        self.numeric_cols = df_ref.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # One-hot columns reference (for alignment)
        df_ohe_ref = pd.get_dummies(df_ref)
        self.ohe_columns = df_ohe_ref.columns.tolist()

        # Scaler for deep models
        self.scaler = StandardScaler()
        self.scaler.fit(df_ohe_ref)

    def transform(self, df: pd.DataFrame):
        """
        Given a raw input DataFrame, returns:
         - X_numeric: DataFrame with numeric columns
         - X_ohe: one-hot encoded + aligned DataFrame
         - X_scaled: numpy array scaled for deep models
        """
        # Numeric subset
        X_numeric = df[self.numeric_cols].copy()

        # One-hot encode and align
        X_ohe = pd.get_dummies(df)
        # Reindex to reference columns, fill missing with 0
        X_ohe = X_ohe.reindex(columns=self.ohe_columns, fill_value=0)

        # Scale for deep models
        X_scaled = self.scaler.transform(X_ohe)

        return X_numeric, X_ohe, X_scaled


if __name__ == '__main__':
    # Paths
    processed_dir = os.path.join('data', 'processed')
    reference_csv = os.path.join(processed_dir, 'train.csv')
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize and fit processor
    print(f"Loading reference data from {reference_csv}")
    processor = DataProcessor(reference_csv)

    # Save the processor pipeline
    processor_path = os.path.join(output_dir, 'processor.pkl')
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    print(f"DataProcessor saved to {processor_path}")

# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 