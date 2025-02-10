import pandas as pd


class DataPreparationService:
    # @staticmethod
    # def prepare_column_for_clustering(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    #     """Prepare a specific column for clustering"""
    #     if column_name not in df.columns:
    #         raise ValueError(f"Column {column_name} not found in dataset")

    #     # Convert to numeric, forcing non-numeric to NaN
    #     df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    #     # Remove rows where the target column is NaN
    #     clean_df = df.dropna(subset=[column_name])

    #     return clean_df

    def prepare_column_for_clustering(
        self, df: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Prepare a column for clustering with better error handling
        """
        try:
            # If column doesn't exist, try to use the first column
            if column_name not in df.columns:
                if len(df.columns) > 0:
                    column_name = df.columns[0]
                else:
                    raise ValueError("No valid columns found in dataset")

            # Create a copy to avoid modifying original
            prepared_df = df.copy()

            # Clean the target column
            prepared_df[column_name] = prepared_df[column_name].astype(str).str.strip()

            # Remove rows where the target column is empty
            prepared_df = prepared_df[prepared_df[column_name].str.len() > 0]

            return prepared_df

        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")
