@echo off
title GNN Social Brain Dashboard
:menu
cls
echo ==================================================
echo       GNN-RWKV SOCIAL BRAIN DASHBOARD
echo ==================================================
echo  1. START CHAT (Use existing knowledge)
echo  2. TRAIN BASE LANGUAGE MODEL
echo  3. TRAIN ON PLAIN TEXT
echo  4. EXIT
echo ==================================================
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto chat
if "%choice%"=="2" goto base
if "%choice%"=="3" goto train
if "%choice%"=="4" goto end

:chat
cls
echo Launching Chat Mode...
uv run python chat.py
pause
goto menu

:base
cls
echo.
set /p epochs="Enter base training epochs (default 8): "
if "%epochs%"=="" set epochs=8
echo Training small base language model for %epochs% epochs...
uv run python train.py --base --epochs %epochs%
echo.
echo Base Model Training Finished!
pause
goto menu

:train
cls
echo.
set /p filename="Enter the name of the .txt file (e.g. story.txt): "
set /p epochs="Enter number of epochs (default 10): "
if "%epochs%"=="" set epochs=10
echo Training on %filename% for %epochs% epochs...
uv run python train.py %filename% --epochs %epochs%
echo.
echo Training Finished!
pause
goto menu

:end
exit
