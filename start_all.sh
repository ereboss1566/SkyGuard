#!/bin/bash

echo "Starting SkyGuard System..."
echo

# Start backend in background
echo "Starting backend server..."
cd web/backend
python server.py > backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

# Wait a bit for backend to start
sleep 5

# Start frontend in background
echo "Starting frontend server..."
cd web/frontend
npm start > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

echo
echo "Servers started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo
echo "Access URLs:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  Health:   http://localhost:8000/health"
echo
echo "Logs are being written to backend.log and frontend.log"
echo "Press Ctrl+C to stop both servers"
echo

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID