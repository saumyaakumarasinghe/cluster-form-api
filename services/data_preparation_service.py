"""
Data Preparation Service for cleaning and preparing text data for clustering.
Handles text preprocessing, stopword removal, stemming, and data validation.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class DataPreparationService:
    """
    Service for preparing and cleaning text data for clustering analysis.
    Handles text preprocessing, normalization, and data validation.
    """

    def __init__(self):
        """Initialize the data preparation service with NLTK resources."""
        # Ensure NLTK resources are available
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        """
        Clean and preprocess text data for clustering using stemming.

        Args:
            text (str): Raw text input

        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters (keep only letters and spaces)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]

        # Stemming (keeping original logic for better silhouette scores)
        tokens = [self.stemmer.stem(word) for word in tokens]

        # Rejoin tokens into text
        cleaned = " ".join(tokens)
        return cleaned.strip()

    def prepare_column_for_clustering(
        self, df: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Prepare a specific column in a DataFrame for clustering analysis.

        This method:
        1. Validates the column exists
        2. Cleans and normalizes text data
        3. Removes empty/null entries
        4. Applies text preprocessing with stemming
        5. Filters out very short responses

        Args:
            df (pd.DataFrame): Input DataFrame
            column_name (str): Name of the column to prepare

        Returns:
            pd.DataFrame: Cleaned DataFrame ready for clustering

        Raises:
            ValueError: If column doesn't exist or data preparation fails
        """
        try:
            # Check if the column exists in the dataframe
            if column_name not in df.columns:
                raise ValueError(
                    f"Column '{column_name}' does not exist in the dataset"
                )

            prepared_df = df.copy()  # Copy dataframe to avoid modifying original

            # Replace newlines and carriage returns with spaces
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
            print(f"Data Cleaning Summary:")
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
            print(f"Final dataset size: {len(prepared_df)} records")
            print("=" * 50)

            return prepared_df

        except Exception as e:
            # Raise a clear error if anything goes wrong during preparation
            raise ValueError(f"Error preparing data: {str(e)}")
