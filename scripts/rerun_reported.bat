@echo off
REM Launch the re-run inside WSL, where the GPU environment lives.
REM Usage:  scripts\rerun_reported.bat            (everything, then push)
REM         scripts\rerun_reported.bat split      (one stage)
setlocal
set STAGES=%*
if "%STAGES%"=="" set STAGES=all
wsl bash -lc "cd /mnt/d/FA019/new && ./scripts/rerun_reported.sh %STAGES%"
endlocal
