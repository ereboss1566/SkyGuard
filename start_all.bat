@echo off
echo Starting SkyGuard System...
echo.

echo Starting backend server...
cd web\backend
start "Backend" /MIN python server.py
cd ..\..

timeout /t 5 /nobreak >nul

echo Starting frontend server...
cd web\frontend
start "Frontend" /MIN npm start
cd ..\..

echo.
echo Servers started!
echo.
echo Access URLs:
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:8000
echo   Health:   http://localhost:8000/health
echo.
echo Press any key to stop both servers...
pause >nul

echo Stopping servers...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Backend*"
taskkill /f /im node.exe /fi "WINDOWTITLE eq Frontend*"