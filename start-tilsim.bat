@echo off
rem Tilsim ovoz serveri + tunnel. Bu oyna ochiq turganda sayt ovoz yaratadi. To'xtatish: Ctrl+C.
title Tilsim ovoz serveri
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-tilsim.ps1"
pause
