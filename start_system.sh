#!/bin/bash

echo "Starting SkyGuard Real-time Weather Prediction System..."
echo

echo "1. Installing backend dependencies..."
cd backend
pip install -r requirements.txt
echo "Backend dependencies installed."
echo

echo "2. Starting backend server..."
python server.py &
BACKEND_PID=$!
sleep 5
echo "Backend server started on http://localhost:8000"
echo

echo "3. Installing frontend dependencies..."
cd ../frontend
npm install
echo "Frontend dependencies installed."
echo

echo "4. Starting frontend development server..."
npm start &
FRONTEND_PID=$!
sleep 10
echo "Frontend server started on http://localhost:3000"
echo

echo "System startup complete!"
echo
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "Backend Health: http://localhost:8000/health"
echo
echo "Press Ctrl+C to stop both servers..."

# Wait for both processes
wait $BACKEND_PID
wait $FRONTEND_PID