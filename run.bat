@echo off

echo Virtual Enviroment is active
call .\.venv\Scripts\activate

set PYTHONPATH=%CD%
echo PYTHONPATH is set to %PYTHONPATH%

echo Opening Windows Terminal with multiple tabs...

wt new-tab -p "PowerShell" -d . --title "Camera00" cmd /k python ./src/identify00.py ^
    ; new-tab -p "PowerShell" -d . --title "Camera01" cmd /k python ./src/identify01.py ^