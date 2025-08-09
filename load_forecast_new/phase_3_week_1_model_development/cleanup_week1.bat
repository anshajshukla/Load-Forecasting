@echo off
REM Week 1 Model Development Directory Cleanup Batch Script
REM ========================================================
REM
REM This batch script provides easy access to cleanup the week 1 directory.
REM It will run the PowerShell cleanup script with appropriate parameters.
REM
REM Author: Cleanup Script
REM Date: August 8, 2025

echo.
echo ================================================================
echo         Week 1 Model Development Directory Cleanup
echo ================================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

:menu
echo Please choose an option:
echo.
echo 1. Preview cleanup (dry run) - Shows what will be deleted
echo 2. Perform actual cleanup - Deletes the files
echo 3. Run Python cleanup script
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto preview
if "%choice%"=="2" goto cleanup
if "%choice%"=="3" goto python
if "%choice%"=="4" goto exit
echo Invalid choice. Please try again.
echo.
goto menu

:preview
echo.
echo ========================================
echo          PREVIEW MODE (DRY RUN)
echo ========================================
echo.
powershell -ExecutionPolicy Bypass -File "cleanup_week1_files.ps1" -WhatIf
echo.
pause
goto menu

:cleanup
echo.
echo ========================================
echo         PERFORMING ACTUAL CLEANUP
echo ========================================
echo.
echo WARNING: This will permanently delete the identified files!
set /p confirm="Are you sure you want to continue? (y/N): "
if /i not "%confirm%"=="y" (
    echo Cleanup cancelled.
    echo.
    goto menu
)
echo.
powershell -ExecutionPolicy Bypass -File "cleanup_week1_files.ps1"
echo.
pause
goto menu

:python
echo.
echo ========================================
echo        RUNNING PYTHON CLEANUP SCRIPT
echo ========================================
echo.
python cleanup_week1_files.py
echo.
pause
goto menu

:exit
echo.
echo Cleanup script finished. Goodbye!
exit /b 0
