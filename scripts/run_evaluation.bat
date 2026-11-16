@echo off
REM Evaluate the trained checkpoints inside WSL, where the GPU environment lives.
REM Usage:  scripts\run_evaluation.bat             (all stages, then push)
REM         scripts\run_evaluation.bat kd nopush
setlocal
set ARGS=%*
if "%ARGS%"=="" set ARGS=all
wsl bash -lc "cd /mnt/d/FA019/new && ./scripts/run_evaluation.sh %ARGS%"
endlocal
