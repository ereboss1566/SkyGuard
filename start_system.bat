@echo off
echo Starting SkyGuard Real-time Weather Prediction System...
echo.

echo 1. Installing backend dependencies...
cd backend
pip install -r requirements.txt
echo Backend dependencies installed.
echo.

echo 2. Starting backend server...
start cmd /k "python server.py"
timeout /t 5 /nobreak >nul
echo Backend server started on http://localhost:8000
echo.

echo 3. Installing frontend dependencies...
cd ..\frontend
npm install
echo Frontend dependencies installed.
echo.

echo 4. Starting frontend development server...
start cmd /k "npm start"
timeout /t 10 /nobreak >nul
echo Frontend server started on http://localhost:3000
echo.

echo System startup complete!
echo.
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8000
echo Backend Health: http://localhost:8000/health
echo.
echo Press any key to exit...
pause >nul