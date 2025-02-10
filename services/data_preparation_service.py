import pandas as pd


class DataPreparationService:

    def prepare_column_for_clustering(
        self, df: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Prepare a column for clustering with better error handling
        """
        try:
            # check if the column exists
            if column_name not in df.columns:
                raise ValueError(
                    f"Column '{column_name}' does not exist in the dataset"
                )

            # create a copy to avoid modifying original
            prepared_df = df.copy()

            # clean the target column
            prepared_df[column_name] = prepared_df[column_name].astype(str).str.strip()

            # remove rows where the target column is empty
            prepared_df = prepared_df[prepared_df[column_name].str.len() > 0]

            return prepared_df

        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")
