# MovieLens-1M Dataset Preprocessing

A production-ready Python solution for preprocessing the MovieLens-1M dataset with comprehensive data handling, encoding, and reporting capabilities.

## Overview

This solution extracts, processes, and transforms the MovieLens-1M dataset from zip format, applying various encoding strategies to categorical variables and generating detailed reports.

## Features

- ✅ **Complete Data Pipeline**: Extract, load, process, merge, and save
- ✅ **Missing Value Handling**: Detection, reporting, and appropriate handling
- ✅ **Categorical Encoding**:
  - Gender: Label encoding (M/F → 0/1)
  - Occupation: One-hot encoding (21 categories)
  - Movie Genres: Multi-label binarization (18 genres)
  - Age Groups: Ordinal encoding (0-6 scale)
- ✅ **Comprehensive Reporting**: Original stats, transformations, final dataset info
- ✅ **Production-Ready**: Error handling, logging, modular design
- ✅ **Well-Documented**: Code comments, docstrings, and AI interaction log

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Ensure ml-1m.zip is in the current directory
python preprocess_movielens.py
```

### Expected Output

The script will:
1. Extract the MovieLens-1M dataset from `ml-1m.zip`
2. Load ratings.dat, users.dat, and movies.dat
3. Detect and report any missing values
4. Apply all encoding transformations
5. Merge all datasets into one
6. Generate a comprehensive report
7. Save output files to `output/` directory

### Output Files

All outputs are saved to the `output/` directory:

- **preprocessed_movielens.csv**: Complete preprocessed dataset (~230 MB)
  - 1,000,209 rows × 51 columns
  - All encodings included
  
- **preprocessed_sample.csv**: Sample of first 1,000 records (~234 KB)
  - For quick inspection and validation
  
- **preprocessing_report.txt**: Detailed report including:
  - Original data statistics
  - Missing values analysis
  - Encoding transformations applied
  - Final dataset information
  
- **preprocessing.log**: Execution log with timestamps
  - All processing steps
  - Warnings and errors
  - Performance metrics

## Dataset Structure

### Original Files (from ml-1m.zip)

1. **ratings.dat**: UserID::MovieID::Rating::Timestamp
2. **users.dat**: UserID::Gender::Age::Occupation::Zip-code
3. **movies.dat**: MovieID::Title::Genres

### Preprocessed Dataset Columns (51 total)

**From Ratings (4 columns):**
- UserID, MovieID, Rating, Timestamp

**From Users (28 columns):**
- Original: Gender, Age, Occupation, Zipcode
- Encoded: Gender_Encoded, Age_Ordinal
- One-hot: Occupation_0 through Occupation_20

**From Movies (19 columns):**
- Original: Title, Genres
- Encoded: Genre_Action, Genre_Adventure, Genre_Animation, Genre_Children's, Genre_Comedy, Genre_Crime, Genre_Documentary, Genre_Drama, Genre_Fantasy, Genre_Film-Noir, Genre_Horror, Genre_Musical, Genre_Mystery, Genre_Romance, Genre_Sci-Fi, Genre_Thriller, Genre_War, Genre_Western

## Encoding Details

### Gender (Label Encoding)
- Method: `sklearn.preprocessing.LabelEncoder`
- Mapping: F → 0, M → 1
- Output column: `Gender_Encoded`

### Age (Ordinal Encoding)
- Method: Manual mapping
- Mapping:
  - 1 (Under 18) → 0
  - 18 (18-24) → 1
  - 25 (25-34) → 2
  - 35 (35-44) → 3
  - 45 (45-49) → 4
  - 50 (50-55) → 5
  - 56 (56+) → 6
- Output column: `Age_Ordinal`

### Occupation (One-Hot Encoding)
- Method: `pandas.get_dummies`
- Categories: 0-20 (21 total occupations)
- Output columns: `Occupation_0` through `Occupation_20`

### Genres (Multi-Label Binarization)
- Method: `sklearn.preprocessing.MultiLabelBinarizer`
- Genres: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western
- Output columns: `Genre_<name>` for each genre (18 columns)

## Architecture

### MovieLensPreprocessor Class

Object-oriented design with methods for each processing step:

```python
preprocessor = MovieLensPreprocessor('ml-1m.zip', 'output')
preprocessor.run_full_pipeline()  # Executes complete workflow
```

### Individual Methods

```python
# Step-by-step processing
preprocessor.extract_dataset()      # Extract zip file
preprocessor.load_data()            # Load .dat files
preprocessor.detect_missing_values()  # Scan for missing data
preprocessor.handle_missing_values()  # Clean data
preprocessor.encode_gender()        # Apply label encoding
preprocessor.encode_age()           # Apply ordinal encoding
preprocessor.encode_occupation()    # Apply one-hot encoding
preprocessor.encode_genres()        # Apply multi-label binarization
preprocessor.merge_datasets()       # Combine all datasets
preprocessor.generate_report()      # Create report
preprocessor.save_preprocessed_data()  # Export to CSV
```

## Error Handling

The script includes comprehensive error handling for:
- Missing zip file
- Corrupted zip archive
- Missing .dat files
- Encoding errors (uses latin-1)
- Missing values in data
- File I/O errors
- Memory issues

All errors are logged with descriptive messages.

## Logging

Dual logging system:
- **Console**: Real-time progress updates
- **File** (preprocessing.log): Complete audit trail with timestamps

Log levels:
- INFO: Normal processing steps
- WARNING: Data quality issues (e.g., missing values)
- ERROR: Failures and exceptions

## Performance

Typical processing time: ~10-15 seconds
- Extraction: ~1 second
- Loading: ~2-3 seconds
- Encoding: ~2-3 seconds
- Merging: ~3-5 seconds
- Saving: ~5-10 seconds

Memory usage: ~450 MB in-memory, ~230 MB on-disk

## Documentation

- **AI_INTERACTION_LOG.md**: Complete documentation of the development process
  - Initial problem statement
  - Data exploration findings
  - Design decisions and rationale
  - Transformation details
  - Sample outputs

## Project Structure

```
.
├── preprocess_movielens.py      # Main preprocessing script
├── requirements.txt              # Python dependencies
├── ml-1m.zip                    # Input dataset (user-provided)
├── AI_INTERACTION_LOG.md        # Complete documentation
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── output/                      # Generated files (gitignored)
│   ├── preprocessed_movielens.csv
│   ├── preprocessed_sample.csv
│   └── preprocessing_report.txt
└── preprocessing.log            # Execution log (gitignored)
```

## Example Output

```
================================================================================
MOVIELENS-1M DATASET PREPROCESSING REPORT
================================================================================

RATINGS Dataset:
  - Total ratings: 1,000,209
  - Unique users: 6,040
  - Unique movies: 3,706

USERS Dataset:
  - Total users: 6,040
  - Gender distribution: {'M': 4331, 'F': 1709}

MOVIES Dataset:
  - Total movies: 3,883

MISSING VALUES REPORT:
RATINGS: No missing values
USERS: No missing values
MOVIES: No missing values

ENCODING TRANSFORMATIONS APPLIED:
GENDER: LabelEncoder - {'F': 0, 'M': 1}
OCCUPATION: OneHotEncoding - 21 categories
GENRES: MultiLabelBinarizer - 18 genres

FINAL PREPROCESSED DATASET:
Shape: (1000209, 51)
Total records: 1,000,209
Memory usage: 447.48 MB
================================================================================
```

## Use Cases

This preprocessed dataset is ready for:
- Movie recommendation systems
- Collaborative filtering algorithms
- User behavior analysis
- Content-based filtering
- Machine learning model training
- Educational purposes

## License

This preprocessing script is provided as-is. The MovieLens-1M dataset has its own license from GroupLens Research.

## Author

Created by AI Assistant for MovieLens-1M dataset preprocessing.

## Support

For issues or questions, please refer to:
- AI_INTERACTION_LOG.md for detailed documentation
- preprocessing.log for execution details
- preprocessing_report.txt for data statistics
