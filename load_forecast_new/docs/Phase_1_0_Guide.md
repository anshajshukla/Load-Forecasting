# SIH 2024 Phase 1.0: Production Database Setup Guide

## 🚀 Step-by-Step Implementation

### Step 1: Create Supabase Account & Project

#### 1.1 Sign Up for Supabase
1. Go to [https://supabase.com](https://supabase.com)
2. Click "Start your project"
3. Sign up with GitHub (recommended) or Google
4. Verify your email if required

#### 1.2 Create New Project
1. Click "New Project" 
2. Select your organization (or create one)
3. Fill in project details:
   ```
   Name: delhi-load-forecasting-sih2024
   Database Password: [Create a strong password - save this!]
   Region: Asia Pacific (Mumbai) [Closest to Delhi]
   Pricing Plan: Free (Perfect for SIH 2024)
   ```
4. Click "Create new project"
5. Wait 2-3 minutes for project provisioning

#### 1.3 Get Your Credentials
1. Go to Settings → API
2. Copy these values:
   ```
   Project URL: https://xxxxx.supabase.co
   anon public key: eyJhbGc...
   service_role key: eyJhbGc...
   ```
3. Go to Settings → Database
4. Copy the connection string:
   ```
   URI: postgresql://postgres:[YOUR-PASSWORD]@db.xxxxx.supabase.co:5432/postgres
   ```

### Step 2: Configure Local Environment

#### 2.1 Install Dependencies
```bash
pip install -r requirements_production.txt
```

#### 2.2 Setup Environment File
1. Copy `.env.template` to `.env`
2. Update `.env` with your Supabase credentials:
   ```
   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_ANON_KEY=eyJhbGc...
   SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...
   DATABASE_URL=postgresql://postgres:yourpassword@db.xxxxx.supabase.co:5432/postgres
   ```

### Step 3: Run Database Migration

#### 3.1 Test Connection
```bash
python -c "from scripts.migrate_to_supabase import SIHDatabaseMigration; m=SIHDatabaseMigration(); print('✅ Connected!' if m.test_connection() else '❌ Failed')"
```

#### 3.2 Create Database Schema
```bash
python scripts/migrate_to_supabase.py
```

#### 3.3 Verify Setup
Check your Supabase dashboard → Table Editor to see the new tables.

### Step 4: Update Application Code

#### 4.1 Create Production Database Manager
This will replace your SQLite setup with cloud PostgreSQL.

#### 4.2 Test SIH Integration
Verify that all components can connect to the cloud database.

---

## 🎯 Let's Start Implementation

### Current Status Check
- [x] Created migration scripts
- [x] Created requirements file
- [x] Created environment template
- [ ] **Next: Install dependencies**
- [ ] **Next: Setup Supabase account**
- [ ] **Next: Configure environment**
- [ ] **Next: Run migration**

---

## 📝 Important Notes

### Why Supabase for SIH 2024?
✅ **Free Tier**: 500MB database + 2GB storage  
✅ **Production Ready**: Real-time capabilities  
✅ **Location**: Mumbai servers (low latency for Delhi)  
✅ **No Credit Card**: Required for free tier  
✅ **Scalable**: Easy to upgrade if needed  

### Storage Calculation
- **Current data**: Likely < 100MB
- **15-minute intervals**: ~87MB per year
- **5 years of data**: ~450MB (fits in free tier)
- **Buffer for features**: Additional ~1GB storage

---

## 🚨 Troubleshooting

### Common Issues:
1. **"psycopg2 not found"**: Install with `pip install psycopg2-binary`
2. **"Connection failed"**: Check DATABASE_URL format
3. **"Permission denied"**: Verify service_role key
4. **"Region error"**: Select Mumbai region for best performance

---

Ready to start? Let's begin with Step 1!
