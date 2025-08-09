# Phase 1.1 Setup Guide - Historical Data Migration

## 📋 Overview

Phase 1.1 focuses on migrating 3 years of historical load data from Delhi SLDC website to Supabase PostgreSQL database. This establishes the foundation for machine learning model training with 6-column load data structure.

## 🎯 Target Data Structure

| Column | Description | Unit |
|--------|-------------|------|
| DELHI  | Total Delhi load | MW |
| BRPL   | Bharti Airtel area load | MW |
| BYPL   | BSES Yamuna Power Limited area load | MW |
| NDPL   | North Delhi Power Limited area load | MW |
| NDMC   | New Delhi Municipal Council area load | MW |
| MES    | Military Engineer Services area load | MW |

## 🔧 Setup Instructions

### 1. Environment Configuration

Create or update your `.env` file:

```bash
# Database Configuration
DATABASE_URL=postgresql://[user]:[password]@[host]:[port]/[database]

# Optional: Weather API (for future phases)
OPENWEATHER_API_KEY=your_api_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `psycopg2-binary` - PostgreSQL database adapter
- `beautifulsoup4` - HTML parsing for web scraping
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `requests` - HTTP library for API calls

### 3. Test Phase 1.1 Components

Run the testing script to verify all components work:

```bash
python scripts/run_phase_1_1.py
```

This will test:
- ✅ Database connection and schema management
- ✅ Historical data fetching capabilities
- ✅ Data quality validation framework

## 🚀 Execution Steps

### Step 1: Database Schema Setup

```bash
python src/data/database_schema_manager.py
```

This creates:
- `load_data` table with 6 load columns
- `weather_data` table for future integration
- `predictions` table for model outputs
- Performance indexes and views
- Sample data for testing

### Step 2: Historical Data Migration

```bash
python src/data/historical_data_migrator.py
```

This will:
- Fetch 3 years of historical data from Delhi SLDC
- Parse HTML tables and extract load values
- Generate realistic simulations for missing data
- Save data to Supabase in batches
- Log progress and handle errors gracefully

**⚠️ Note**: This process may take 30-60 minutes depending on network speed and website availability.

### Step 3: Data Quality Validation

```bash
python src/data/data_quality_validator.py
```

This validates:
- Data ranges (2000-8000 MW for Delhi total)
- Completeness (target: 95%+)
- Consistency (area loads should sum to ~Delhi total)
- Generates detailed quality report

## 📊 Expected Results

After successful completion:

- **Database**: ~150,000 records of historical load data
- **Time Range**: 3 years of data at 30-minute intervals
- **Quality**: >90% data completeness with validation
- **Performance**: <100ms query response times

## 🔍 Verification

Check your data migration success:

```sql
-- Connect to your Supabase database and run:
SELECT COUNT(*) as total_records FROM load_data;
SELECT MIN(timestamp), MAX(timestamp) FROM load_data;
SELECT data_source, COUNT(*) FROM load_data GROUP BY data_source;
```

Expected output:
- **Total records**: ~150,000+
- **Date range**: ~3 years from current date
- **Data sources**: Mix of real scraping and realistic simulation

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check `DATABASE_URL` in `.env` file
   - Verify Supabase credentials and network access
   - Test connection: `python -c "import psycopg2; psycopg2.connect('your_database_url')"`

2. **Web Scraping Timeouts**
   - Normal behavior - script automatically falls back to simulation
   - Check internet connection stability
   - Review logs in `data_migration.log`

3. **Import Errors**
   - Install missing dependencies: `pip install psycopg2-binary beautifulsoup4`
   - Use virtual environment for clean package management

4. **Low Data Quality**
   - Review validation report in `docs/phase_1_1_validation_report.json`
   - Check if simulation fallback is working correctly
   - Verify data source patterns match expected format

## 📁 File Structure

```
src/data/
├── database_schema_manager.py    # Database setup and schema creation
├── historical_data_migrator.py   # Web scraping and data migration
└── data_quality_validator.py     # Data quality validation and reporting

scripts/
└── run_phase_1_1.py             # Testing suite for Phase 1.1

learning/phases/
└── phase_1_1_learnings.md       # Comprehensive learning documentation
```

## 🎯 Success Criteria

Phase 1.1 is complete when:

- ✅ Database schema created with 6-column load structure
- ✅ Historical data migrated (>90% completeness)
- ✅ Data quality validated and documented
- ✅ Performance benchmarks met (<100ms queries)
- ✅ Learning documentation updated

## 🚀 Next Phase

Once Phase 1.1 is complete, proceed to **Phase 1.2**: Database optimization and monitoring setup.

---

*For detailed technical insights and lessons learned, see `learning/phases/phase_1_1_learnings.md`*
