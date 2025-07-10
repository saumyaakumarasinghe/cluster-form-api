# This service handles cleaning and preparing data from spreadsheets for clustering.
# It ensures the data is in a suitable format and handles missing/empty values robustly.

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class DataPreparationService:
    def __init__(self):
        # Ensure NLTK resources are available
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        # Stemming
        tokens = [self.stemmer.stem(word) for word in tokens]
        # Rejoin
        cleaned = ' '.join(tokens)
        return cleaned.strip()

    # Prepares a specific column in a DataFrame for clustering
    def prepare_column_for_clustering(
        self, df: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Prepare a column for clustering with better error handling
        and track the number of empty and null values across all columns.
        """
        try:
            # Check if the column exists in the dataframe
            if column_name not in df.columns:
                raise ValueError(
                    f"Column '{column_name}' does not exist in the dataset"
                )

            prepared_df = df.copy()  # Copy dataframe to avoid modifying original

            # Replace newlines with spaces
            prepared_df[column_name] = prepared_df[column_name].replace(
                {r"\n": " ", r"\r": " "}, regex=True
            )

            # Replace 'None' and empty strings with empty text
            prepared_df[column_name] = prepared_df[column_name].replace(
                {"None": "", "": ""}
            )

            # Strip spaces and non-visible characters
            prepared_df[column_name] = prepared_df[column_name].str.strip()

            # Count null and empty values before cleaning
            null_count = prepared_df[column_name].isna().sum()
            empty_count = (prepared_df[column_name].str.len() == 0).sum()
            print("\n" + "=" * 50)
            print(f"Null count (before cleaning): {str(null_count)}")
            print(f"Empty count (before cleaning): {str(empty_count)}")

            # Remove rows where the target column is null or empty
            prepared_df = prepared_df[
                prepared_df[column_name].notna()
                & (prepared_df[column_name].str.len() > 0)
            ]

            # Apply advanced cleaning to each response
            prepared_df[column_name] = prepared_df[column_name].apply(self.clean_text)

            # Remove very short responses (e.g., less than 3 characters)
            prepared_df = prepared_df[prepared_df[column_name].str.len() >= 3]

            # Count null and empty values after cleaning
            null_count_after = prepared_df[column_name].isna().sum()
            empty_count_after = (prepared_df[column_name].str.len() == 0).sum()
            print(f"Null count (after cleaning): {str(null_count_after)}")
            print(f"Empty count (after cleaning): {str(empty_count_after)}")

            return prepared_df  # Return cleaned dataframe

        except Exception as e:
            # Raise a clear error if anything goes wrong during preparation
            raise ValueError(f"Error preparing data: {str(e)}")
