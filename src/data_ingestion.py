"""
Data Ingestion Pipeline for Vehicle Registration Project
=======================================================

This module handles:
1. Loading and cleaning CSV files
2. Standardizing column names and data types
3. Merging all data into a single SQLite database
4. Data validation and quality checks
"""

import pandas as pd
import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VehicleDataIngestion:
    """Main class for handling vehicle registration data ingestion."""
    
    def __init__(self, data_dir: str = "../data", db_path: str = "../db/vehicles.db"):
        """Initialize the data ingestion pipeline."""
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.csv_files = list(self.data_dir.glob("*.csv"))
        
        # Column mapping for standardization
        self.column_mapping = {
            'slno': 'serial_number',
            'registrationNo': 'registration_number',
            'reserve_no': 'reserve_number',
            'regvalidfrom': 'registration_valid_from',
            'regvalidto': 'registration_valid_to',
            'makerName': 'maker_name',
            'modelDesc': 'model_description',
            'bodyType': 'body_type',
            'cc': 'engine_cc',
            'cylinder': 'cylinder_count',
            'fuel': 'fuel_type',
            'hp': 'horsepower',
            'seatCapacity': 'seat_capacity',
            'OfficeCd': 'office_code',
            'fromdate': 'data_from_date',
            'todate': 'data_to_date'
        }
        
        # Date columns that need parsing
        self.date_columns = [
            'registration_valid_from',
            'registration_valid_to', 
            'data_from_date',
            'data_to_date'
        ]
        
        logger.info(f"Initialized with {len(self.csv_files)} CSV files")
    
    def load_csv_file(self, file_path: Path, sample_size: int = None) -> pd.DataFrame:
        """Load a CSV file with error handling."""
        try:
            if sample_size:
                df = pd.read_csv(file_path, nrows=sample_size)
                logger.info(f"Loaded sample of {sample_size} rows from {file_path.name}")
            else:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {str(e)}")
            raise
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across all datasets."""
        df_clean = df.rename(columns=self.column_mapping)
        
        # Ensure all expected columns exist
        expected_columns = set(self.column_mapping.values())
        missing_columns = expected_columns - set(df_clean.columns)
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            for col in missing_columns:
                df_clean[col] = None
        
        return df_clean
    
    def clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data types."""
        df_clean = df.copy()
        
        # Parse date columns
        for col in self.date_columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', format='%d/%m/%Y')
                    logger.info(f"Parsed date column: {col}")
                except Exception as e:
                    logger.warning(f"Could not parse date column {col}: {str(e)}")
        
        # Clean string columns
        string_columns = ['registration_number', 'maker_name', 'model_description', 
                         'body_type', 'fuel_type', 'office_code']
        
        for col in string_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.upper()
                df_clean[col] = df_clean[col].replace(['NAN', 'NONE', ''], None)
        
        # Clean numeric columns
        numeric_columns = ['serial_number', 'engine_cc', 'cylinder_count', 
                          'horsepower', 'seat_capacity']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform data validation and return quality metrics."""
        validation_results = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'date_range': {},
            'data_quality_score': 0
        }
        
        # Check date ranges
        for col in self.date_columns:
            if col in df.columns and not df[col].isnull().all():
                validation_results['date_range'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'null_count': df[col].isnull().sum()
                }
        
        # Calculate data quality score (0-100)
        total_cells = len(df) * len(df.columns)
        null_cells = df.isnull().sum().sum()
        duplicate_penalty = validation_results['duplicate_rows'] * len(df.columns)
        
        quality_score = max(0, 100 - ((null_cells + duplicate_penalty) / total_cells * 100))
        validation_results['data_quality_score'] = round(quality_score, 2)
        
        return validation_results
    
    def process_single_file(self, file_path: Path, sample_size: int = None) -> pd.DataFrame:
        """Process a single CSV file through the complete pipeline."""
        logger.info(f"Processing {file_path.name}...")
        
        # Load data
        df = self.load_csv_file(file_path, sample_size)
        
        # Standardize columns
        df = self.standardize_columns(df)
        
        # Clean data types
        df = self.clean_data_types(df)
        
        # Add source file information
        df['source_file'] = file_path.name
        df['processed_at'] = datetime.now()
        
        # Validate data
        validation = self.validate_data(df)
        logger.info(f"Data quality score for {file_path.name}: {validation['data_quality_score']}%")
        
        return df
    
    def create_database(self) -> sqlite3.Connection:
        """Create SQLite database and connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        logger.info(f"Created database connection: {self.db_path}")
        return conn
    
    def load_to_database(self, df: pd.DataFrame, conn: sqlite3.Connection, 
                        table_name: str = "vehicle_registrations", 
                        if_exists: str = "replace") -> None:
        """Load DataFrame to SQLite database."""
        try:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            logger.info(f"Loaded {len(df)} rows to table '{table_name}'")
        except Exception as e:
            logger.error(f"Error loading to database: {str(e)}")
            raise
    
    def create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for better query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_registration_number ON vehicle_registrations(registration_number)",
            "CREATE INDEX IF NOT EXISTS idx_maker_name ON vehicle_registrations(maker_name)",
            "CREATE INDEX IF NOT EXISTS idx_office_code ON vehicle_registrations(office_code)",
            "CREATE INDEX IF NOT EXISTS idx_registration_valid_from ON vehicle_registrations(registration_valid_from)",
            "CREATE INDEX IF NOT EXISTS idx_fuel_type ON vehicle_registrations(fuel_type)",
            "CREATE INDEX IF NOT EXISTS idx_body_type ON vehicle_registrations(body_type)"
        ]
        
        cursor = conn.cursor()
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                logger.warning(f"Could not create index: {str(e)}")
        
        conn.commit()
    
    def run_full_pipeline(self, sample_size: int = None, 
                         validate_only: bool = False) -> Dict[str, Any]:
        """Run the complete data ingestion pipeline."""
        logger.info("Starting full data ingestion pipeline...")
        
        results = {
            'files_processed': 0,
            'total_rows': 0,
            'errors': [],
            'validation_results': {},
            'database_created': False
        }
        
        try:
            # Process all CSV files
            all_dataframes = []
            
            for file_path in self.csv_files:
                try:
                    df = self.process_single_file(file_path, sample_size)
                    all_dataframes.append(df)
                    results['files_processed'] += 1
                    results['total_rows'] += len(df)
                    
                    # Store validation results
                    results['validation_results'][file_path.name] = self.validate_data(df)
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path.name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            if not validate_only and all_dataframes:
                # Combine all dataframes
                logger.info("Combining all datasets...")
                combined_df = pd.concat(all_dataframes, ignore_index=True)
                
                # Final validation
                final_validation = self.validate_data(combined_df)
                results['final_validation'] = final_validation
                
                # Load to database
                logger.info("Loading to SQLite database...")
                conn = self.create_database()
                self.load_to_database(combined_df, conn, if_exists="replace")
                
                # Create indexes for better query performance
                self.create_indexes(conn)
                
                conn.close()
                results['database_created'] = True
                
                logger.info(f"Pipeline completed successfully!")
                logger.info(f"Total rows processed: {results['total_rows']}")
                logger.info(f"Final data quality score: {final_validation['data_quality_score']}%")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get summary statistics from the created database."""
        if not self.db_path.exists():
            return {"error": "Database does not exist"}
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get table info
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vehicle_registrations")
            total_rows = cursor.fetchone()[0]
            
            # Get column info
            cursor.execute("PRAGMA table_info(vehicle_registrations)")
            columns = cursor.fetchall()
            
            # Get sample data
            cursor.execute("SELECT * FROM vehicle_registrations LIMIT 5")
            sample_data = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_rows": total_rows,
                "columns": [col[1] for col in columns],
                "sample_data": sample_data
            }
            
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main function to run the data ingestion pipeline."""
    # Initialize the pipeline
    pipeline = VehicleDataIngestion()
    
    # Run the pipeline
    print("🚀 Starting Vehicle Registration Data Ingestion Pipeline")
    print("=" * 60)
    
    # First, run validation only to check data quality
    print("📊 Running data validation...")
    validation_results = pipeline.run_full_pipeline(sample_size=1000, validate_only=True)
    
    print(f"✅ Validation complete!")
    print(f"   Files processed: {validation_results['files_processed']}")
    print(f"   Total sample rows: {validation_results['total_rows']}")
    
    if validation_results['errors']:
        print(f"⚠️  Errors found: {len(validation_results['errors'])}")
        for error in validation_results['errors']:
            print(f"   - {error}")
    
    # Show data quality scores
    print("\n📈 Data Quality Scores:")
    for file_name, validation in validation_results['validation_results'].items():
        print(f"   {file_name}: {validation['data_quality_score']}%")
    
    # Ask user if they want to proceed with full processing
    print("\n" + "=" * 60)
    response = input("Proceed with full data processing? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n🔄 Running full data processing...")
        full_results = pipeline.run_full_pipeline()
        
        if full_results['database_created']:
            print("✅ Database created successfully!")
            
            # Get and display summary
            summary = pipeline.get_database_summary()
            if 'error' not in summary:
                print(f"\n📊 Database Summary:")
                print(f"   Total rows: {summary['total_rows']:,}")
                print(f"   Columns: {len(summary['columns'])}")
                print(f"   Database location: {pipeline.db_path}")
        else:
            print("❌ Database creation failed!")
            for error in full_results['errors']:
                print(f"   - {error}")
    else:
        print("⏹️  Pipeline stopped by user.")


if __name__ == "__main__":
    main()
