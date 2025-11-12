"""
MovieLens-1M Dataset Preprocessing - Simple Solution
=====================================================

AI Prompts and Responses:
--------------------------

PROMPT 1: "How do I load MovieLens-1M dataset from a zip file?"
AI OUTPUT: Use zipfile module to extract, then pandas.read_csv with sep='::' and engine='python'

PROMPT 2: "What encodings should I use for categorical variables?"
AI OUTPUT: 
  - Gender: LabelEncoder (binary categorical)
  - Occupation: OneHotEncoder (nominal categorical)
  - Age: OrdinalEncoder (ordered categorical)
  - Genres: MultiLabelBinarizer (multi-label categorical)

PROMPT 3: "How to handle missing values in the dataset?"
AI OUTPUT: Use df.isnull().sum() to detect, then dropna() or fillna() based on column importance
"""

import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer

# ============================================================================
# STEP 1: Extract and Load Data
# ============================================================================
print("=" * 80)
print("STEP 1: EXTRACTING AND LOADING DATA")
print("=" * 80)

# Extract zip file
with zipfile.ZipFile('ml-1m.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
print("✓ Extracted ml-1m.zip")

# Load ratings data
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
print(f"✓ Loaded ratings: {len(ratings)} records")

# Load users data
users = pd.read_csv('ml-1m/users.dat', sep='::', engine='python',
                    names=['UserID', 'Gender', 'Age', 'Occupation', 'Zipcode'])
print(f"✓ Loaded users: {len(users)} records")

# Load movies data
movies = pd.read_csv('ml-1m/movies.dat', sep='::', engine='python',
                     names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
print(f"✓ Loaded movies: {len(movies)} records")

# ============================================================================
# STEP 2: Check for Missing Values
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CHECKING FOR MISSING VALUES")
print("=" * 80)

print("\nRatings missing values:")
print(ratings.isnull().sum())

print("\nUsers missing values:")
print(users.isnull().sum())

print("\nMovies missing values:")
print(movies.isnull().sum())

# Handle missing values (if any)
ratings.dropna(inplace=True)
users.dropna(inplace=True)
movies.dropna(inplace=True)
print("\n✓ Missing values handled (dropped)")

# ============================================================================
# STEP 3: Encode Categorical Variables
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: ENCODING CATEGORICAL VARIABLES")
print("=" * 80)

# 3.1 Gender - Label Encoding
print("\n3.1 Gender (Label Encoding)")
le_gender = LabelEncoder()
users['Gender_Encoded'] = le_gender.fit_transform(users['Gender'])
print(f"  Original: {users['Gender'].unique()}")
print(f"  Encoded: {users['Gender_Encoded'].unique()}")
print(f"  Mapping: F={le_gender.transform(['F'])[0]}, M={le_gender.transform(['M'])[0]}")

# 3.2 Age - Ordinal Encoding (already numeric groups)
print("\n3.2 Age (Ordinal Encoding)")
age_mapping = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
users['Age_Encoded'] = users['Age'].map(age_mapping)
print(f"  Original age groups: {sorted(users['Age'].unique())}")
print(f"  Encoded to ordinal: {sorted(users['Age_Encoded'].unique())}")

# 3.3 Occupation - One-Hot Encoding
print("\n3.3 Occupation (One-Hot Encoding)")
occupation_dummies = pd.get_dummies(users['Occupation'], prefix='Occupation')
users = pd.concat([users, occupation_dummies], axis=1)
print(f"  Created {len(occupation_dummies.columns)} binary columns")
print(f"  Columns: {list(occupation_dummies.columns[:5])}... (showing first 5)")

# 3.4 Movie Genres - Multi-Label Binarization
print("\n3.4 Movie Genres (Multi-Label Binarization)")
genres_split = movies['Genres'].str.split('|')
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(genres_split)
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=movies.index)
movies = pd.concat([movies, genres_df], axis=1)
print(f"  Unique genres: {len(mlb.classes_)}")
print(f"  Genres: {list(mlb.classes_)}")

# ============================================================================
# STEP 4: Merge Datasets
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: MERGING DATASETS")
print("=" * 80)

# Merge ratings with users
merged = ratings.merge(users, on='UserID', how='left')
print(f"✓ Merged ratings + users: {merged.shape}")

# Merge with movies
merged = merged.merge(movies, on='MovieID', how='left')
print(f"✓ Merged with movies: {merged.shape}")

# ============================================================================
# STEP 5: Save Results
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: SAVING PREPROCESSED DATA")
print("=" * 80)

# Save full dataset
merged.to_csv('preprocessed_movielens_simple.csv', index=False)
print(f"✓ Saved full dataset: preprocessed_movielens_simple.csv")
print(f"  Shape: {merged.shape}")
print(f"  Size: {merged.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# Save sample
sample = merged.head(1000)
sample.to_csv('sample_preprocessed.csv', index=False)
print(f"✓ Saved sample: sample_preprocessed.csv (1000 rows)")

# ============================================================================
# STEP 6: Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING SUMMARY")
print("=" * 80)

print("\nOriginal Data:")
print(f"  Ratings:  {len(ratings):,} records")
print(f"  Users:    {len(users):,} records")
print(f"  Movies:   {len(movies):,} records")

print("\nEncoding Applied:")
print(f"  ✓ Gender: Label Encoding (F→0, M→1)")
print(f"  ✓ Age: Ordinal Encoding (7 groups → 0-6)")
print(f"  ✓ Occupation: One-Hot Encoding (21 categories)")
print(f"  ✓ Genres: Multi-Label Binarization (18 genres)")

print("\nFinal Dataset:")
print(f"  Rows: {len(merged):,}")
print(f"  Columns: {len(merged.columns)}")
print(f"  Missing Values: {merged.isnull().sum().sum()}")

print("\nOutput Files:")
print(f"  1. preprocessed_movielens_simple.csv (full dataset)")
print(f"  2. sample_preprocessed.csv (1000 rows sample)")

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE!")
print("=" * 80)

# Display sample of preprocessed data
print("\nSample of preprocessed data (first 5 rows, selected columns):")
display_cols = ['UserID', 'MovieID', 'Rating', 'Gender_Encoded', 'Age_Encoded', 
                'Occupation_0', 'Title']
print(merged[display_cols].head())

print("\n" + "=" * 80)
print("AI INTERACTION SUMMARY")
print("=" * 80)
print("""
INPUT: MovieLens-1M dataset (zip file)
  - ratings.dat: 1M+ movie ratings
  - users.dat: 6K+ user demographics
  - movies.dat: 3K+ movie metadata

AI PROMPTS USED:
  1. How to extract zip files in Python? → zipfile module
  2. How to read :: delimited files? → pd.read_csv with sep='::'
  3. Best encoding for binary categorical? → LabelEncoder
  4. Best encoding for multi-category? → OneHotEncoder
  5. How to encode multi-label data? → MultiLabelBinarizer
  6. How to handle missing values? → dropna() or fillna()

OUTPUT: Preprocessed dataset with encoded features
  - All categorical variables properly encoded
  - No missing values
  - Ready for machine learning models
""")
