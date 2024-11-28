# Run Instructions

This guide provides simple commands to set up and run the ANPR system. All commands are intended to be run from the **project root directory** (`ANPR/`).

## 1. Prerequisites
- **Python 3.10+**
- **Node.js & npm** (for the frontend)
- **CUDA 12.1** (optional, for GPU acceleration)

## 2. One-Time Setup

### Backend Setup
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Frontend Setup
Install the required JavaScript libraries:
```bash
npm install --prefix static
```

---

## 3. Running the Application

You will need two terminal windows (one for the backend, one for the frontend).

### Terminal 1: Start the Backend API
This starts the Python server on `http://127.0.0.1:8000`.
```bash
python -m src.main --mode api
```

### Terminal 2: Start the Frontend UI
This starts the web interface on `http://localhost:3000`.
```bash
npm run dev --prefix static
```

---

## 4. Accessing the App
Once both are running, open your browser and go to:
**http://localhost:3000**

The frontend will automatically talk to the backend API.
