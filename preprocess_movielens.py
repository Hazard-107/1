"""
MovieLens-1M Dataset Preprocessing Script

This script extracts, processes, and transforms the MovieLens-1M dataset
with proper encoding of categorical variables and comprehensive reporting.

Author: AI Assistant
Date: 2025-11-12
"""

import os
import sys
import zipfile
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.preprocessing import OrdinalEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MovieLensPreprocessor:
    """
    Preprocessor for MovieLens-1M dataset with comprehensive data handling
    and transformation capabilities.
    """
    
    def __init__(self, zip_path: str, output_dir: str = "output"):
        """
        Initialize the preprocessor.
        
        Args:
            zip_path: Path to the MovieLens-1M zip file
            output_dir: Directory to save processed data
        """
        self.zip_path = zip_path
        self.output_dir = output_dir
        self.extract_dir = "ml-1m"
        
        # DataFrames
        self.ratings_df = None
        self.users_df = None
        self.movies_df = None
        self.merged_df = None
        
        # Encoders (for potential inverse transformation)
        self.encoders = {}
        
        # Reports
        self.report_data = {
            'original_stats': {},
            'missing_values': {},
            'transformations': {},
            'final_stats': {}
        }
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        logger.info(f"Initialized MovieLensPreprocessor with zip: {zip_path}")
    
    def extract_dataset(self) -> None:
        """Extract the MovieLens-1M dataset from zip file."""
        try:
            if not os.path.exists(self.zip_path):
                raise FileNotFoundError(f"Zip file not found: {self.zip_path}")
            
            logger.info(f"Extracting {self.zip_path}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            logger.info("Extraction completed successfully")
            
        except zipfile.BadZipFile:
            logger.error(f"Corrupted zip file: {self.zip_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting dataset: {str(e)}")
            raise
    
    def load_data(self) -> None:
        """Load the three main data files from MovieLens-1M dataset."""
        try:
            # Define file paths
            ratings_path = os.path.join(self.extract_dir, 'ratings.dat')
            users_path = os.path.join(self.extract_dir, 'users.dat')
            movies_path = os.path.join(self.extract_dir, 'movies.dat')
            
            # Check if files exist
            for path in [ratings_path, users_path, movies_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Data file not found: {path}")
            
            # Load ratings.dat
            # Format: UserID::MovieID::Rating::Timestamp
            logger.info("Loading ratings.dat...")
            self.ratings_df = pd.read_csv(
                ratings_path,
                sep='::',
                engine='python',
                names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                encoding='latin-1'
            )
            logger.info(f"Loaded {len(self.ratings_df)} ratings")
            
            # Load users.dat
            # Format: UserID::Gender::Age::Occupation::Zip-code
            logger.info("Loading users.dat...")
            self.users_df = pd.read_csv(
                users_path,
                sep='::',
                engine='python',
                names=['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode'],
                encoding='latin-1'
            )
            logger.info(f"Loaded {len(self.users_df)} users")
            
            # Load movies.dat
            # Format: MovieID::Title::Genres
            logger.info("Loading movies.dat...")
            self.movies_df = pd.read_csv(
                movies_path,
                sep='::',
                engine='python',
                names=['MovieID', 'Title', 'Genres'],
                encoding='latin-1'
            )
            logger.info(f"Loaded {len(self.movies_df)} movies")
            
            # Store original statistics
            self._collect_original_stats()
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _collect_original_stats(self) -> None:
        """Collect statistics from original data."""
        self.report_data['original_stats'] = {
            'ratings': {
                'count': len(self.ratings_df),
                'shape': self.ratings_df.shape,
                'columns': list(self.ratings_df.columns),
                'dtypes': self.ratings_df.dtypes.to_dict(),
                'rating_distribution': self.ratings_df['Rating'].value_counts().to_dict(),
                'unique_users': self.ratings_df['UserID'].nunique(),
                'unique_movies': self.ratings_df['MovieID'].nunique()
            },
            'users': {
                'count': len(self.users_df),
                'shape': self.users_df.shape,
                'columns': list(self.users_df.columns),
                'dtypes': self.users_df.dtypes.to_dict(),
                'gender_distribution': self.users_df['Gender'].value_counts().to_dict(),
                'age_distribution': self.users_df['Age'].value_counts().to_dict(),
                'occupation_distribution': self.users_df['Occupation'].value_counts().to_dict()
            },
            'movies': {
                'count': len(self.movies_df),
                'shape': self.movies_df.shape,
                'columns': list(self.movies_df.columns),
                'dtypes': self.movies_df.dtypes.to_dict()
            }
        }
        logger.info("Collected original data statistics")
    
    def detect_missing_values(self) -> Dict[str, pd.Series]:
        """
        Detect and report missing values in all datasets.
        
        Returns:
            Dictionary containing missing value counts for each dataset
        """
        logger.info("Detecting missing values...")
        
        missing_values = {
            'ratings': self.ratings_df.isnull().sum(),
            'users': self.users_df.isnull().sum(),
            'movies': self.movies_df.isnull().sum()
        }
        
        self.report_data['missing_values'] = {
            'ratings': missing_values['ratings'].to_dict(),
            'users': missing_values['users'].to_dict(),
            'movies': missing_values['movies'].to_dict()
        }
        
        # Log missing values
        for dataset_name, missing in missing_values.items():
            total_missing = missing.sum()
            if total_missing > 0:
                logger.warning(f"Missing values in {dataset_name}: {total_missing}")
                logger.warning(f"Details:\n{missing[missing > 0]}")
            else:
                logger.info(f"No missing values in {dataset_name}")
        
        return missing_values
    
    def handle_missing_values(self) -> None:
        """Handle missing values appropriately."""
        logger.info("Handling missing values...")
        
        # For this dataset, there should be no missing values
        # But we'll handle them just in case
        
        # Ratings: Drop rows with missing critical values
        if self.ratings_df.isnull().any().any():
            before = len(self.ratings_df)
            self.ratings_df.dropna(subset=['UserID', 'MovieID', 'Rating'], inplace=True)
            after = len(self.ratings_df)
            logger.info(f"Dropped {before - after} rows with missing values in ratings")
        
        # Users: Drop rows with missing critical values
        if self.users_df.isnull().any().any():
            before = len(self.users_df)
            self.users_df.dropna(subset=['UserID', 'Gender', 'Age', 'Occupation'], inplace=True)
            after = len(self.users_df)
            logger.info(f"Dropped {before - after} rows with missing values in users")
        
        # Movies: Drop rows with missing critical values
        if self.movies_df.isnull().any().any():
            before = len(self.movies_df)
            self.movies_df.dropna(subset=['MovieID', 'Title', 'Genres'], inplace=True)
            after = len(self.movies_df)
            logger.info(f"Dropped {before - after} rows with missing values in movies")
    
    def encode_gender(self) -> None:
        """Encode gender using Label Encoding (M/F to 0/1)."""
        logger.info("Encoding gender with LabelEncoder...")
        
        le = LabelEncoder()
        self.users_df['Gender_Encoded'] = le.fit_transform(self.users_df['Gender'])
        self.encoders['gender'] = le
        
        encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
        self.report_data['transformations']['gender'] = {
            'method': 'LabelEncoder',
            'original_column': 'Gender',
            'new_column': 'Gender_Encoded',
            'encoding_map': encoding_map,
            'classes': list(le.classes_)
        }
        
        logger.info(f"Gender encoding complete: {encoding_map}")
    
    def encode_occupation(self) -> pd.DataFrame:
        """
        Encode occupation using One-Hot Encoding.
        
        Returns:
            DataFrame with one-hot encoded occupation columns
        """
        logger.info("Encoding occupation with OneHotEncoder...")
        
        # Get unique occupations
        occupations = self.users_df['Occupation'].unique()
        
        # Create one-hot encoded columns
        occupation_dummies = pd.get_dummies(
            self.users_df['Occupation'],
            prefix='Occupation'
        )
        
        # Add to users dataframe
        self.users_df = pd.concat([self.users_df, occupation_dummies], axis=1)
        
        self.report_data['transformations']['occupation'] = {
            'method': 'OneHotEncoding (get_dummies)',
            'original_column': 'Occupation',
            'new_columns': list(occupation_dummies.columns),
            'num_categories': len(occupations),
            'categories': sorted(list(occupations))
        }
        
        logger.info(f"Occupation encoding complete: {len(occupations)} categories")
        return occupation_dummies
    
    def encode_genres(self) -> pd.DataFrame:
        """
        Encode movie genres using Multi-Label Binarization.
        Genres are separated by '|' in the original data.
        
        Returns:
            DataFrame with binarized genre columns
        """
        logger.info("Encoding genres with MultiLabelBinarizer...")
        
        # Split genres (they are pipe-separated)
        genres_split = self.movies_df['Genres'].str.split('|')
        
        # Apply MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(genres_split)
        
        # Create DataFrame with genre columns
        genres_df = pd.DataFrame(
            genres_encoded,
            columns=[f'Genre_{genre}' for genre in mlb.classes_],
            index=self.movies_df.index
        )
        
        # Add to movies dataframe
        self.movies_df = pd.concat([self.movies_df, genres_df], axis=1)
        self.encoders['genres'] = mlb
        
        self.report_data['transformations']['genres'] = {
            'method': 'MultiLabelBinarizer',
            'original_column': 'Genres',
            'new_columns': list(genres_df.columns),
            'num_genres': len(mlb.classes_),
            'genres': list(mlb.classes_)
        }
        
        logger.info(f"Genre encoding complete: {len(mlb.classes_)} unique genres")
        return genres_df
    
    def encode_age(self) -> None:
        """
        Encode age groups using ordinal encoding.
        Age is already in grouped format (1, 18, 25, 35, 45, 50, 56).
        """
        logger.info("Encoding age groups...")
        
        # Age groups are already numerical but we can create ordinal encoding
        # The values represent: 1: "Under 18", 18: "18-24", 25: "25-34", 
        # 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"
        
        age_mapping = {
            1: 0,   # Under 18
            18: 1,  # 18-24
            25: 2,  # 25-34
            35: 3,  # 35-44
            45: 4,  # 45-49
            50: 5,  # 50-55
            56: 6   # 56+
        }
        
        self.users_df['Age_Ordinal'] = self.users_df['Age'].map(age_mapping)
        
        self.report_data['transformations']['age'] = {
            'method': 'Ordinal Encoding (Manual Mapping)',
            'original_column': 'Age',
            'new_column': 'Age_Ordinal',
            'mapping': age_mapping,
            'age_groups': {
                '0': 'Under 18',
                '1': '18-24',
                '2': '25-34',
                '3': '35-44',
                '4': '45-49',
                '5': '50-55',
                '6': '56+'
            }
        }
        
        logger.info("Age encoding complete")
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge ratings, users, and movies datasets.
        
        Returns:
            Merged DataFrame
        """
        logger.info("Merging datasets...")
        
        # First merge ratings with users
        merged = self.ratings_df.merge(
            self.users_df,
            on='UserID',
            how='left'
        )
        
        # Then merge with movies
        merged = merged.merge(
            self.movies_df,
            on='MovieID',
            how='left'
        )
        
        self.merged_df = merged
        
        logger.info(f"Merged dataset shape: {merged.shape}")
        logger.info(f"Merged dataset columns: {len(merged.columns)}")
        
        return merged
    
    def generate_report(self) -> str:
        """
        Generate comprehensive preprocessing report.
        
        Returns:
            Report as formatted string
        """
        logger.info("Generating comprehensive report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MOVIELENS-1M DATASET PREPROCESSING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Original Data Statistics
        report_lines.append("-" * 80)
        report_lines.append("1. ORIGINAL DATA STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        report_lines.append("RATINGS Dataset:")
        stats = self.report_data['original_stats']['ratings']
        report_lines.append(f"  - Total ratings: {stats['count']:,}")
        report_lines.append(f"  - Shape: {stats['shape']}")
        report_lines.append(f"  - Unique users: {stats['unique_users']:,}")
        report_lines.append(f"  - Unique movies: {stats['unique_movies']:,}")
        report_lines.append(f"  - Rating distribution: {stats['rating_distribution']}")
        report_lines.append("")
        
        report_lines.append("USERS Dataset:")
        stats = self.report_data['original_stats']['users']
        report_lines.append(f"  - Total users: {stats['count']:,}")
        report_lines.append(f"  - Shape: {stats['shape']}")
        report_lines.append(f"  - Gender distribution: {stats['gender_distribution']}")
        report_lines.append(f"  - Age distribution: {stats['age_distribution']}")
        report_lines.append("")
        
        report_lines.append("MOVIES Dataset:")
        stats = self.report_data['original_stats']['movies']
        report_lines.append(f"  - Total movies: {stats['count']:,}")
        report_lines.append(f"  - Shape: {stats['shape']}")
        report_lines.append("")
        
        # Missing Values Report
        report_lines.append("-" * 80)
        report_lines.append("2. MISSING VALUES REPORT")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        for dataset_name, missing in self.report_data['missing_values'].items():
            total = sum(missing.values())
            if total > 0:
                report_lines.append(f"{dataset_name.upper()}: {total} missing values found")
                for col, count in missing.items():
                    if count > 0:
                        report_lines.append(f"  - {col}: {count}")
            else:
                report_lines.append(f"{dataset_name.upper()}: No missing values")
        report_lines.append("")
        
        # Encoding Transformations
        report_lines.append("-" * 80)
        report_lines.append("3. ENCODING TRANSFORMATIONS APPLIED")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        for feature, transform in self.report_data['transformations'].items():
            report_lines.append(f"{feature.upper()}:")
            report_lines.append(f"  - Method: {transform['method']}")
            report_lines.append(f"  - Original column: {transform['original_column']}")
            if 'new_column' in transform:
                report_lines.append(f"  - New column: {transform['new_column']}")
            if 'new_columns' in transform:
                report_lines.append(f"  - New columns: {len(transform['new_columns'])} columns")
            if 'encoding_map' in transform:
                report_lines.append(f"  - Encoding map: {transform['encoding_map']}")
            if 'num_categories' in transform:
                report_lines.append(f"  - Number of categories: {transform['num_categories']}")
            if 'num_genres' in transform:
                report_lines.append(f"  - Number of genres: {transform['num_genres']}")
                report_lines.append(f"  - Genres: {', '.join(transform['genres'])}")
            report_lines.append("")
        
        # Final Dataset Statistics
        if self.merged_df is not None:
            report_lines.append("-" * 80)
            report_lines.append("4. FINAL PREPROCESSED DATASET")
            report_lines.append("-" * 80)
            report_lines.append("")
            report_lines.append(f"Shape: {self.merged_df.shape}")
            report_lines.append(f"Columns: {len(self.merged_df.columns)}")
            report_lines.append(f"Total records: {len(self.merged_df):,}")
            report_lines.append("")
            report_lines.append("Column names:")
            for i, col in enumerate(self.merged_df.columns, 1):
                report_lines.append(f"  {i:3d}. {col}")
            report_lines.append("")
            
            # Memory usage
            memory_mb = self.merged_df.memory_usage(deep=True).sum() / 1024 / 1024
            report_lines.append(f"Memory usage: {memory_mb:.2f} MB")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report to file
        report_path = os.path.join(self.output_dir, 'preprocessing_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Report saved to {report_path}")
        
        return report_text
    
    def save_preprocessed_data(self, filename: str = 'preprocessed_movielens.csv') -> None:
        """
        Save preprocessed data to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        if self.merged_df is None:
            raise ValueError("No merged dataset available. Run merge_datasets() first.")
        
        output_path = os.path.join(self.output_dir, filename)
        logger.info(f"Saving preprocessed data to {output_path}...")
        
        self.merged_df.to_csv(output_path, index=False)
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        logger.info(f"Saved {len(self.merged_df)} records ({file_size_mb:.2f} MB)")
    
    def save_sample_data(self, n_samples: int = 1000) -> None:
        """
        Save a sample of the preprocessed data for inspection.
        
        Args:
            n_samples: Number of samples to save
        """
        if self.merged_df is None:
            raise ValueError("No merged dataset available.")
        
        sample_df = self.merged_df.head(n_samples)
        sample_path = os.path.join(self.output_dir, 'preprocessed_sample.csv')
        sample_df.to_csv(sample_path, index=False)
        logger.info(f"Saved {n_samples} sample records to {sample_path}")
    
    def run_full_pipeline(self) -> None:
        """Execute the complete preprocessing pipeline."""
        logger.info("Starting full preprocessing pipeline...")
        
        try:
            # Step 1: Extract dataset
            self.extract_dataset()
            
            # Step 2: Load data
            self.load_data()
            
            # Step 3: Detect missing values
            self.detect_missing_values()
            
            # Step 4: Handle missing values
            self.handle_missing_values()
            
            # Step 5: Apply encodings
            self.encode_gender()
            self.encode_age()
            self.encode_occupation()
            self.encode_genres()
            
            # Step 6: Merge datasets
            self.merge_datasets()
            
            # Step 7: Generate report
            report = self.generate_report()
            print("\n" + report)
            
            # Step 8: Save results
            self.save_preprocessed_data()
            self.save_sample_data()
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point for the script."""
    try:
        # Configuration
        ZIP_PATH = 'ml-1m.zip'
        OUTPUT_DIR = 'output'
        
        # Check if zip file exists
        if not os.path.exists(ZIP_PATH):
            logger.error(f"Zip file not found: {ZIP_PATH}")
            logger.error("Please ensure 'ml-1m.zip' is in the current directory")
            sys.exit(1)
        
        # Create preprocessor and run pipeline
        preprocessor = MovieLensPreprocessor(ZIP_PATH, OUTPUT_DIR)
        preprocessor.run_full_pipeline()
        
        print("\n" + "=" * 80)
        print("SUCCESS! Preprocessing completed.")
        print("=" * 80)
        print(f"\nOutput files saved to '{OUTPUT_DIR}/' directory:")
        print(f"  - preprocessed_movielens.csv (full dataset)")
        print(f"  - preprocessed_sample.csv (sample for inspection)")
        print(f"  - preprocessing_report.txt (detailed report)")
        print(f"\nLog file: preprocessing.log")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
