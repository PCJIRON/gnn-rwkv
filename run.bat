@echo off
title GNN Social Brain Dashboard
:menu
cls
echo ==================================================
echo       GNN-RWKV SOCIAL BRAIN DASHBOARD
echo ==================================================
echo  1. START CHAT (Use existing knowledge)
echo  2. TRAIN ON NEW DATA (Add a new .txt file)
echo  3. EXIT
echo ==================================================
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto chat
if "%choice%"=="2" goto train
if "%choice%"=="3" goto end

:chat
cls
echo Launching Chat Mode...
uv run python chat.py
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
