echo Opening Windows Terminal with multiple tabs...

wt new-tab -p "PowerShell" -d . --title "Camera00" cmd /k python ./src/identify00.py ^
    ; new-tab -p "PowerShell" -d . --title "Camera01" cmd /k python ./src/identify01.py ^