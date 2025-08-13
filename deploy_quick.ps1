# 🚀 Delhi Load Forecasting Project - Quick Deploy Script
# PowerShell script for rapid GitHub deployment

param(
    [string]$RepoUrl = "https://github.com/anshajshukla/SIH2024---Delhi-Load-Forecasting.git",
    [switch]$DryRun
)

Write-Host "🚀 Delhi Load Forecasting Project - GitHub Deployment" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

$SourceDir = "C:\Users\ansha\Desktop\SIH_new"
$TempDir = "temp_deployment"

# Check if source directory exists
if (-not (Test-Path $SourceDir)) {
    Write-Host "❌ Source directory not found: $SourceDir" -ForegroundColor Red
    exit 1
}

Write-Host "📁 Source Directory: $SourceDir" -ForegroundColor Cyan
Write-Host "🎯 Target Repository: $RepoUrl" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "🔍 DRY RUN MODE - No actual deployment" -ForegroundColor Yellow
    Write-Host ""
    
    # Run Python script in dry-run mode
    python deploy_to_github.py --source $SourceDir --repo $RepoUrl --dry-run
} else {
    Write-Host "⚠️  LIVE DEPLOYMENT - Will push to GitHub" -ForegroundColor Yellow
    Write-Host ""
    
    # Confirm deployment
    $confirm = Read-Host "Continue with deployment? (y/N)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Host "❌ Deployment cancelled" -ForegroundColor Red
        exit 0
    }
    
    # Run Python script
    python deploy_to_github.py --source $SourceDir --repo $RepoUrl
}

Write-Host ""
Write-Host "🎉 Script execution completed!" -ForegroundColor Green

# Instructions for next steps
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Check the GitHub repository: $RepoUrl" -ForegroundColor White
Write-Host "2. Deploy dashboard on Streamlit Cloud" -ForegroundColor White
Write-Host "3. Update repository settings and description" -ForegroundColor White
Write-Host "4. Add collaborators if needed" -ForegroundColor White

if (-not $DryRun) {
    Write-Host ""
    Write-Host "🌐 Dashboard Deployment:" -ForegroundColor Cyan
    Write-Host "- Visit https://share.streamlit.io/" -ForegroundColor White
    Write-Host "- Connect your GitHub repository" -ForegroundColor White
    Write-Host "- Set main file: load_forecast/delhi_forecasting_dashboard/main.py" -ForegroundColor White
    Write-Host "- Deploy automatically" -ForegroundColor White
}
