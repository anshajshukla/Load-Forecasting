"""
Phase 1.1 - Data Quality Validator
Validates migrated historical data quality and completeness
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

class DataQualityValidator:
    """Validate historical data quality and completeness."""
    
    def __init__(self):
        """Initialize data validator."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load data quality thresholds
        self.thresholds = {
            'min_delhi_load': 2000,    # MW
            'max_delhi_load': 8000,    # MW
            'min_area_load': 100,      # MW
            'max_area_load': 3000,     # MW
            'completeness_threshold': 0.95,  # 95% data completeness
            'max_missing_hours': 24    # Max consecutive missing hours
        }
        
        print("🔍 Data Quality Validator for Phase 1.1")
        print("📊 Validating 6-column load data quality")
    
    def get_data_from_supabase(self, limit=None):
        """Get data from Supabase for validation."""
        try:
            conn = psycopg2.connect(self.database_url)
            
            # Build query
            query = """
                SELECT 
                    timestamp,
                    delhi_load,
                    brpl_load,
                    bypl_load,
                    ndpl_load,
                    ndmc_load,
                    mes_load,
                    data_source,
                    quality_score
                FROM load_data 
                ORDER BY timestamp
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Load into DataFrame
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                self.logger.warning("⚠️ No data found in database")
                return None
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"📊 Loaded {len(df)} records for validation")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            return None
    
    def validate_data_ranges(self, df):
        """Validate that load values are within reasonable ranges."""
        print("\n🔍 RANGE VALIDATION")
        print("=" * 50)
        
        validation_results = {}
        load_columns = ['delhi_load', 'brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
        
        for col in load_columns:
            col_min = df[col].min()
            col_max = df[col].max()
            col_mean = df[col].mean()
            
            # Check ranges
            if col == 'delhi_load':
                range_ok = (col_min >= self.thresholds['min_delhi_load'] and 
                           col_max <= self.thresholds['max_delhi_load'])
            else:
                range_ok = (col_min >= self.thresholds['min_area_load'] and 
                           col_max <= self.thresholds['max_area_load'])
            
            # Count outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            outlier_pct = len(outliers) / len(df) * 100
            
            status = "✅" if range_ok and outlier_pct < 5 else "⚠️"
            
            print(f"{status} {col}:")
            print(f"   Range: {col_min:.1f} - {col_max:.1f} MW")
            print(f"   Mean: {col_mean:.1f} MW")
            print(f"   Outliers: {len(outliers)} ({outlier_pct:.1f}%)")
            
            validation_results[col] = {
                'range_ok': range_ok,
                'outlier_count': len(outliers),
                'outlier_percentage': outlier_pct,
                'min': col_min,
                'max': col_max,
                'mean': col_mean
            }
        
        return validation_results
    
    def validate_completeness(self, df):
        """Validate data completeness and identify gaps."""
        print("\n📊 COMPLETENESS VALIDATION")
        print("=" * 50)
        
        # Expected time range
        start_time = df.index.min()
        end_time = df.index.max()
        
        # Generate expected timestamps (30-minute intervals)
        expected_range = pd.date_range(start=start_time, end=end_time, freq='30T')
        expected_count = len(expected_range)
        actual_count = len(df)
        
        completeness = actual_count / expected_count
        
        print(f"📅 Time range: {start_time} to {end_time}")
        print(f"📊 Expected records: {expected_count:,}")
        print(f"📊 Actual records: {actual_count:,}")
        print(f"📊 Completeness: {completeness:.2%}")
        
        # Find missing timestamps
        missing_timestamps = expected_range.difference(df.index)
        missing_count = len(missing_timestamps)
        
        if missing_count > 0:
            print(f"⚠️ Missing {missing_count:,} timestamps")
        else:
            print("✅ No missing timestamps")
        
        status = "✅" if completeness >= self.thresholds['completeness_threshold'] else "⚠️"
        
        return {
            'completeness': completeness,
            'missing_count': missing_count,
            'status': status
        }
    
    def validate_consistency(self, df):
        """Validate data consistency and relationships."""
        print("\n🔄 CONSISTENCY VALIDATION")
        print("=" * 50)
        
        load_columns = ['brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
        
        # Check if area loads sum reasonably to Delhi total
        df['areas_sum'] = df[load_columns].sum(axis=1)
        df['sum_diff'] = abs(df['delhi_load'] - df['areas_sum'])
        df['sum_diff_pct'] = (df['sum_diff'] / df['delhi_load']) * 100
        
        # Calculate consistency metrics
        avg_diff_pct = df['sum_diff_pct'].mean()
        max_diff_pct = df['sum_diff_pct'].max()
        consistent_records = (df['sum_diff_pct'] <= 15).sum()  # Within 15%
        consistency_rate = consistent_records / len(df)
        
        print(f"📊 Area loads vs Delhi total:")
        print(f"   Average difference: {avg_diff_pct:.1f}%")
        print(f"   Maximum difference: {max_diff_pct:.1f}%")
        print(f"   Consistent records: {consistent_records:,} ({consistency_rate:.1%})")
        
        # Check for negative values
        negative_counts = {}
        for col in ['delhi_load'] + load_columns:
            negative_count = (df[col] < 0).sum()
            negative_counts[col] = negative_count
            if negative_count > 0:
                print(f"⚠️ {col}: {negative_count} negative values")
        
        status = "✅" if consistency_rate >= 0.85 and sum(negative_counts.values()) == 0 else "⚠️"
        
        return {
            'consistency_rate': consistency_rate,
            'avg_diff_percentage': avg_diff_pct,
            'negative_values': negative_counts,
            'status': status
        }
    
    def generate_validation_report(self, df):
        """Generate comprehensive validation report."""
        print("\n" + "="*70)
        print("📋 PHASE 1.1 DATA VALIDATION REPORT")
        print("="*70)
        
        # Run all validations
        range_results = self.validate_data_ranges(df)
        completeness_results = self.validate_completeness(df)
        consistency_results = self.validate_consistency(df)
        
        # Overall assessment
        print("\n🎯 OVERALL ASSESSMENT")
        print("=" * 50)
        
        range_status = all(result['outlier_percentage'] < 5 for result in range_results.values())
        completeness_ok = completeness_results['completeness'] >= self.thresholds['completeness_threshold']
        consistency_ok = consistency_results['consistency_rate'] >= 0.85
        
        overall_score = sum([range_status, completeness_ok, consistency_ok]) / 3
        
        print(f"📊 Range validation: {'✅ PASS' if range_status else '⚠️ ISSUES'}")
        print(f"📊 Completeness: {'✅ PASS' if completeness_ok else '⚠️ ISSUES'}")
        print(f"📊 Consistency: {'✅ PASS' if consistency_ok else '⚠️ ISSUES'}")
        print(f"🏆 Overall score: {overall_score:.1%}")
        
        if overall_score >= 0.8:
            print("✅ Data quality is GOOD - Ready for Phase 1.2")
        elif overall_score >= 0.6:
            print("⚠️ Data quality is ACCEPTABLE - Some improvements needed")
        else:
            print("❌ Data quality NEEDS IMPROVEMENT - Review data sources")
        
        # Save detailed report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'time_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            },
            'range_validation': range_results,
            'completeness': completeness_results,
            'consistency': consistency_results,
            'overall_score': overall_score
        }
        
        return report


def main():
    """Main function to run data validation."""
    print("=" * 70)
    print("🔍 PHASE 1.1 - DATA VALIDATION")
    print("=" * 70)
    print("📊 Validating historical data in Supabase")
    print("🗄️ Checking 6-column load data quality")
    print("=" * 70)
    
    try:
        # Initialize validator
        validator = DataQualityValidator()
        
        # Load data from Supabase
        print("\n📥 Loading data from Supabase...")
        df = validator.get_data_from_supabase()
        
        if df is None:
            print("❌ No data to validate!")
            return
        
        # Generate validation report
        report = validator.generate_validation_report(df)
        
        # Save report
        import json
        report_path = Path("docs/phase_1_1_validation_report.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Detailed report saved to {report_path}")
        
        print("\n🎉 Phase 1.1 data validation completed!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
