# Week 2 Scripts Cleanup - Remove Irrelevant Files
# Keep only the main 02_lstm_gru_models.py and essential files

$scriptDir = "C:\Users\ansha\Desktop\SIH_new\load_forecast\phase_3_week_2_neural_networks\scripts"
$logFile = "C:\Users\ansha\Desktop\SIH_new\load_forecast\phase_3_week_2_neural_networks\cleanup_week2_log.txt"

# Files to keep (main implementation)
$keepFiles = @(
    "02_lstm_gru_models.py",           # Main LSTM/GRU implementation
    "01_neural_network_data_preparation.py",  # Data preparation (if needed)
    "03_final_evaluation_ensemble.py"  # Final evaluation (if needed)
)

# Files to remove (redundant/test files)
$removeFiles = @(
    "00_week2_fast_implementation.py",
    "00_week2_neural_network_pipeline.py", 
    "01_fast_data_prep.py",
    "02_fast_lstm_gru.py",
    "quick_neural_network_demo.py",
    "quick_test_optimized.py", 
    "simple_nn_demo.py",
    "week2_master_fast.py"
)

Write-Host "🧹 Week 2 Scripts Cleanup" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

# Initialize log
"Week 2 Scripts Cleanup - $(Get-Date)" | Out-File -FilePath $logFile -Encoding UTF8
"=" * 50 | Out-File -FilePath $logFile -Append -Encoding UTF8

$removedCount = 0
$errorCount = 0

# Function to safely remove files
function Remove-FilesSafely {
    param([string[]]$FilesToRemove, [string]$Directory)
    
    foreach ($file in $FilesToRemove) {
        $fullPath = Join-Path $Directory $file
        
        if (Test-Path $fullPath) {
            try {
                Remove-Item $fullPath -Force
                Write-Host "✅ Removed: $file" -ForegroundColor Green
                "REMOVED: $file" | Out-File -FilePath $logFile -Append -Encoding UTF8
                $script:removedCount++
            }
            catch {
                Write-Host "❌ Error removing $file : $($_.Exception.Message)" -ForegroundColor Red
                "ERROR: $file - $($_.Exception.Message)" | Out-File -FilePath $logFile -Append -Encoding UTF8
                $script:errorCount++
            }
        }
        else {
            Write-Host "⚠️  File not found: $file" -ForegroundColor Yellow
            "NOT FOUND: $file" | Out-File -FilePath $logFile -Append -Encoding UTF8
        }
    }
}

# Show files to keep
Write-Host "`n📌 Keeping these essential files:" -ForegroundColor Yellow
foreach ($file in $keepFiles) {
    $fullPath = Join-Path $scriptDir $file
    if (Test-Path $fullPath) {
        Write-Host "   ✓ $file" -ForegroundColor Green
        "KEEPING: $file" | Out-File -FilePath $logFile -Append -Encoding UTF8
    }
    else {
        Write-Host "   ⚠️  $file (not found)" -ForegroundColor Yellow
        "KEEP BUT NOT FOUND: $file" | Out-File -FilePath $logFile -Append -Encoding UTF8
    }
}

# Remove irrelevant files
Write-Host "`n🗑️  Removing irrelevant files:" -ForegroundColor Red
"" | Out-File -FilePath $logFile -Append -Encoding UTF8
"FILES TO REMOVE:" | Out-File -FilePath $logFile -Append -Encoding UTF8
Remove-FilesSafely -FilesToRemove $removeFiles -Directory $scriptDir

# Summary
Write-Host "`n📊 CLEANUP SUMMARY:" -ForegroundColor Cyan
Write-Host "   Files removed: $removedCount" -ForegroundColor Green
Write-Host "   Errors: $errorCount" -ForegroundColor Red
Write-Host "   Main file: 02_lstm_gru_models.py ✓" -ForegroundColor Green

"" | Out-File -FilePath $logFile -Append -Encoding UTF8
"CLEANUP SUMMARY:" | Out-File -FilePath $logFile -Append -Encoding UTF8
"Files removed: $removedCount" | Out-File -FilePath $logFile -Append -Encoding UTF8
"Errors: $errorCount" | Out-File -FilePath $logFile -Append -Encoding UTF8
"Completed: $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding UTF8

# List remaining files
Write-Host "`n📁 Remaining files in scripts directory:" -ForegroundColor Cyan
$remainingFiles = Get-ChildItem $scriptDir -File | Select-Object -ExpandProperty Name
foreach ($file in $remainingFiles) {
    Write-Host "   📄 $file" -ForegroundColor White
}

Write-Host "`n✅ Week 2 cleanup completed!" -ForegroundColor Green
Write-Host "📝 Log saved to: $logFile" -ForegroundColor Gray

# Show disk space saved (approximate)
$savedSpace = $removedCount * 50  # Rough estimate of KB per file
Write-Host "💾 Approximate space saved: $savedSpace KB" -ForegroundColor Cyan
