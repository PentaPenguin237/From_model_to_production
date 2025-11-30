# Real-Time Anomaly Detection in an IoT-Enabled Factory
**Course:** DLBDSMBTP01 - From Model to Production
**Author:** Guglielmo Luraschi Sicca

---

## 1. Project Overview

This project implements a production-grade Machine Learning pipeline for anomaly detection in an industrial setting. It utilizes a **Microservices Architecture** to separate the intelligence (ML Model) from the data source (Sensors).

The core objective is to detect potential equipment failures using an **Isolation Forest** algorithm exposed via a standardized **RESTful API**. The system is fully containerized using Docker, ensuring reproducibility and scalability.

---

## 2. File Structure

\`\`\`text
/University_Project/
├── docker/
│   └── Dockerfile             <-- Docker configuration (Python 3.10 Slim)
├── data/
│   └── predictive_maintenance.csv  <-- Automatically downloaded on startup
├── results/
│   └── isolation_forest_model.joblib <-- Persisted model artifact
├── .gitignore                 <-- Git exclusion rules
├── docker-compose.yml         <-- Orchestration (API Service)
├── main.py                    <-- FastAPI Server (The "Production" App)
├── simulate_sensors.py        <-- Client Simulation Script (The "Factory")
├── requirements.txt           <-- Dependencies (FastAPI, Scikit-learn, etc.)
└── README.md                  <-- Project documentation
\`\`\`

---

## 3. Installation & Setup

### Prerequisites
* [Docker Desktop](https://www.docker.com/get-started) (Running and updated)
* [Docker Compose](https://docs.docker.com/compose/install/)
* Python 3.10+ (for running the client simulation locally)

### Setup Steps
1.  **Clone the Repository**
    \`\`\`bash
    git clone [REPOSITORY_URL]
    cd University_Project
    \`\`\`

2.  **Data Setup**
    * **Manual action is NOT required.**
    * The application includes an automated ingestion module. On first launch, it checks the \`./data\` volume. If the dataset is missing, it automatically authenticates with KaggleHub, downloads the dataset, and persists it to the local disk.

---

## 4. How to Run (Production Simulation)

This project uses a Client-Server architecture. You will run the **API Server** in Docker and the **Sensor Simulation** on your local machine.

### Step 1: Start the API Server
This builds the container and starts the FastAPI service on Port 8000.

\`\`\`bash
# Build and Start
docker compose up --build
\`\`\`

*Wait until you see:* \`Uvicorn running on http://0.0.0.0:8000\`

### Step 2: Run the Sensor Simulation
Open a **new terminal window**. This script acts as the IoT Gateway, generating synthetic sensor data and sending it to the API.

\`\`\`bash
# Install client requirements (if needed)
pip install requests

# Start the factory simulation
python simulate_sensors.py
\`\`\`

### Step 3: Observe Real-Time Monitoring
Watch the output in your simulation terminal. You will see real-time JSON responses from the Docker container:

\`\`\`text
Sensor Sending: {'temperature_k': 301.4, 'rotational_speed_rpm': 1496.2}
API Response:   NORMAL (Score: 0.157)
...
--- INJECTING ANOMALY (High RPM) ---
Sensor Sending: {'temperature_k': 301.4, 'rotational_speed_rpm': 2800}
API Response:   ALERT (Score: -0.18)
\`\`\`

### Step 4: Clean Up
To stop the server and free resources:
\`\`\`bash
docker compose down
\`\`\`

---

## 5. Architecture & Logic

* **The Server (`main.py`):** A FastAPI application that loads the trained Isolation Forest model. It exposes a \`/predict\` endpoint that accepts JSON payloads validated by **Pydantic** models.
* **The Model:** An unsupervised Isolation Forest trained on the *Predictive Maintenance* dataset. It learns the distribution of "normal" temperature and vibration (RPM) levels.
* **The Client (`simulate_sensors.py`):** A Python script that generates a continuous stream of data using sine waves and random noise. It periodically injects synthetic anomalies (spikes in RPM) to validate the system's detection capabilities.

---

## 6. Results

The system successfully identifies anomalies in real-time with low latency. The Isolation Forest model correctly flags the injected high-RPM events as anomalies (Negative Scores), while classifying standard operating parameters as normal (Positive Scores).