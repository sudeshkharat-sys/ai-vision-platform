@echo off
rem cmd forgets commands when the window closes. Run this before closing to append
rem this window's commands to %USERPROFILE%\cmd-history.txt (copy from there later).
doskey /history >> "%USERPROFILE%\cmd-history.txt"
echo Saved to %USERPROFILE%\cmd-history.txt
