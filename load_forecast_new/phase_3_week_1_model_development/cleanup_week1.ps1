# Week 1 Model Development Directory Cleanup Script (PowerShell)
# ================================================================
#
# This script removes unnecessary, duplicate, and redundant files from the 
# phase_3_week_1_model_development directory to clean up the workspace.
#
# Author: Cleanup Script
# Date: August 8, 2025

param(
    [switch]$WhatIf = $false,
    [switch]$Force = $false
)

function Write-Status {
    param($Message, $Type = "Info")
    $timestamp = Get-Date -Format "HH:mm:ss"
    switch ($Type) {
        "Success" { Write-Host "[$timestamp] ✓ $Message" -ForegroundColor Green }
        "Warning" { Write-Host "[$timestamp] ⚠ $Message" -ForegroundColor Yellow }
        "Error"   { Write-Host "[$timestamp] ✗ $Message" -ForegroundColor Red }
        "Info"    { Write-Host "[$timestamp] ℹ $Message" -ForegroundColor Cyan }
        default   { Write-Host "[$timestamp] $Message" }
    }
}

function Remove-ItemSafe {
    param(
        [string]$Path,
        [string]$Type = "File"
    )
    
    try {
        if (Test-Path $Path) {
            if ($WhatIf) {
                Write-Status "Would remove ${Type}: $Path" "Info"
                return $true
            } else {
                Remove-Item -Path $Path -Recurse -Force
                Write-Status "Removed ${Type}: $Path" "Success"
                return $true
            }
        } else {
            Write-Status "${Type} not found: $Path" "Warning"
            return $false
        }
    } catch {
        Write-Status "Error removing ${Type} ${Path}: $($_.Exception.Message)" "Error"
        return $false
    }
}

# Main cleanup logic
Write-Host "🧹 Starting Week 1 Model Development Directory Cleanup" -ForegroundColor Magenta
Write-Host "=" * 60 -ForegroundColor Magenta

# Get the current directory (should be the week 1 directory)
$BaseDir = $PSScriptRoot
$ScriptsDir = Join-Path $BaseDir "scripts"

Write-Status "Base directory: $BaseDir"
Write-Status "Scripts directory: $ScriptsDir"

if ($WhatIf) {
    Write-Host "`n🔍 DRY RUN MODE - No files will be actually deleted" -ForegroundColor Yellow
}

# Files to remove from scripts directory (checking what actually exists)
$PotentialFilesToRemove = @(
    "scripts\02_fast_baselines.py",
    "scripts\02_linear_tree_baselines_fast.py", 
    "scripts\02_linear_tree_baselines_fixed.py",
    "scripts\02_super_fast_baselines.py",
    "scripts\03_gradient_boosting_time_series_fast.py",
    "scripts\03_gradient_boosting_time_series_fixed.py",
    "scripts\04_baseline_evaluation_documentation_fixed.py",
    "scripts\00_week1_complete_pipeline_fixed.py",
    "scripts\00_week1_fast_pipeline.py",
    "scripts\01_data_preparation_pipeline_fixed.py"
)

# Directories to remove
$DirectoriesToRemove = @(
    "notebooks",
    "scripts\data",
    "scripts\logs",
    "phase_3_week_1_model_development",
    "scripts\__pycache__"
)

# Execution log files to remove
$LogFilesToRemove = @(
    "WEEK_1_FAST_EXECUTION_LOG.txt",
    "WEEK_1_FAST_EXECUTION_SUMMARY.json"
)

# Counters
$RemovedFiles = 0
$RemovedDirs = 0
$RemovedLogs = 0

# Remove duplicate script files
Write-Host "`n📁 Processing duplicate script files..." -ForegroundColor Blue
foreach ($FileRelPath in $PotentialFilesToRemove) {
    $FilePath = Join-Path $BaseDir $FileRelPath
    if (Remove-ItemSafe -Path $FilePath -Type "File") {
        $RemovedFiles++
    }
}

# Remove execution log files
Write-Host "`n📄 Processing redundant execution logs..." -ForegroundColor Blue
foreach ($LogFile in $LogFilesToRemove) {
    $LogPath = Join-Path $BaseDir $LogFile
    if (Remove-ItemSafe -Path $LogPath -Type "Log file") {
        $RemovedLogs++
    }
}

# Remove empty/redundant directories
Write-Host "`n📂 Processing empty/redundant directories..." -ForegroundColor Blue
foreach ($DirRelPath in $DirectoriesToRemove) {
    $DirPath = Join-Path $BaseDir $DirRelPath
    if (Remove-ItemSafe -Path $DirPath -Type "Directory") {
        $RemovedDirs++
    }
}

# Additional cleanup: Look for any remaining _fixed, _fast, _super_fast files
Write-Host "`n🔍 Scanning for additional duplicate files..." -ForegroundColor Blue
if (Test-Path $ScriptsDir) {
    $AdditionalFiles = Get-ChildItem -Path $ScriptsDir -Recurse -File | 
                      Where-Object { $_.Name -match '_fixed\.py$|_fast\.py$|_super_fast\.py$' }
    
    if ($AdditionalFiles) {
        Write-Status "Found $($AdditionalFiles.Count) additional duplicate files"
        foreach ($File in $AdditionalFiles) {
            if (Remove-ItemSafe -Path $File.FullName -Type "Additional duplicate file") {
                $RemovedFiles++
            }
        }
    } else {
        Write-Status "No additional duplicate files found"
    }
}

# Summary
Write-Host "`n" + ("=" * 60) -ForegroundColor Magenta
Write-Host "🎉 Cleanup Summary:" -ForegroundColor Magenta
Write-Host "   • Processed $RemovedFiles duplicate script files" -ForegroundColor Green
Write-Host "   • Processed $RemovedLogs redundant log files" -ForegroundColor Green
Write-Host "   • Processed $RemovedDirs redundant directories" -ForegroundColor Green
Write-Host "   • Total items processed: $($RemovedFiles + $RemovedLogs + $RemovedDirs)" -ForegroundColor Green

if ($WhatIf) {
    Write-Host "`n💡 To actually perform the cleanup, run this script without the -WhatIf parameter" -ForegroundColor Yellow
} else {
    Write-Host "`n✅ Week 1 directory cleanup completed successfully!" -ForegroundColor Green
}

# Show remaining structure
Write-Host "`n📋 Current structure in scripts directory:" -ForegroundColor Blue
if (Test-Path $ScriptsDir) {
    Get-ChildItem -Path $ScriptsDir | Sort-Object Name | ForEach-Object {
        if ($_.PSIsContainer) {
            Write-Host "   📁 $($_.Name)/" -ForegroundColor Yellow
        } elseif ($_.Extension -eq '.py') {
            Write-Host "   📄 $($_.Name)" -ForegroundColor White
        }
    }
}
