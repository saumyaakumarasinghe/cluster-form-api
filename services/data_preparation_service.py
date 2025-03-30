import pandas as pd


class DataPreparationService:

    def prepare_column_for_clustering(
        self, df: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Prepare a column for clustering with better error handling
        and track the number of empty and null values across all columns.
        """
        try:
            # check if the column exists in the dataframe
            if column_name not in df.columns:
                raise ValueError(
                    f"Column '{column_name}' does not exist in the dataset"
                )

            # Create a copy of the dataframe to avoid modifying the original dataframe
            prepared_df = df.copy()

            # normalize multi-line entries (replace newlines with spaces)
            prepared_df[column_name] = prepared_df[column_name].replace(
                {r"\n": " ", r"\r": " "}, regex=True
            )

            # handle "None" values as a string and empty strings
            # replacing "None" and empty strings with empty text
            prepared_df[column_name] = prepared_df[column_name].replace(
                {"None": "", "": ""}
            )

            # strip spaces and non-visible characters before counting empty entries
            prepared_df[column_name] = prepared_df[column_name].str.strip()

            # track the number of null and empty values for the specific column before cleaning
            null_count = prepared_df[column_name].isna().sum()
            empty_count = (prepared_df[column_name].str.len() == 0).sum()
            print(f"Null count (before cleaning): {str(null_count)}")
            print(f"Empty count (before cleaning): {str(empty_count)}")

            # remove rows where the target column is either null or empty
            # use 'notna()' to check for non-null values and check for string length > 0 for non-empty entries
            prepared_df = prepared_df[
                prepared_df[column_name].notna()
                & (prepared_df[column_name].str.len() > 0)
            ]

            # track the number of null and empty values for the specific column after cleaning
            null_count_after = prepared_df[column_name].isna().sum()
            empty_count_after = (prepared_df[column_name].str.len() == 0).sum()
            print(f"Null count (after cleaning): {str(null_count_after)}")
            print(f"Empty count (after cleaning): {str(empty_count_after)}")

            # return the cleaned dataframe
            return prepared_df

        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")
