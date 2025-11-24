# Real-Time Anomaly Detection in an IoT-Enabled Factory
**Course:** DLBDSMBTP01 - From Model to Production
**Author:** Guglielmo Luraschi Sicca - Matriculation no.92125339

---

## 1. Project Overview

This project implements a complete Machine Learning pipeline for anomaly detection in an industrial setting. It simulates a "Model-to-Production" workflow where sensor data is ingested, processed, and analyzed in real-time.

The core objective is to detect potential equipment failures using an **Isolation Forest** algorithm. The system is fully containerized using Docker, ensuring reproducibility across different environments. It features automated data ingestion via the Kaggle API, robust feature engineering (handling Kelvin/Celsius conversion and synthetic humidity generation), and persistent model storage.

---

## 2. File Structure

\`\`\`text
/University_Project/
├── docker/
│   └── Dockerfile             <-- Docker configuration (Python 3.10 Slim)
├── data/
│   └── predictive_maintenance.csv  <-- Automatically downloaded on first run
├── results/
│   └── isolation_forest_model.joblib <-- Persisted model artifact
├── .gitignore                 <-- Git exclusion rules
├── docker-compose.yml         <-- Container orchestration config
├── from_model_to_production.py  <-- Main pipeline script
├── requirements.txt             <-- Python dependencies
└── README.md                    <-- Project documentation
\`\`\`

---

## 3. Installation & Setup

### Prerequisites
* [Docker Desktop](https://www.docker.com/get-started) (Running and updated)
* [Docker Compose](https://docs.docker.com/compose/install/)
* Internet connection (for initial data download)

### Setup Steps
1.  **Clone the Repository**
    \`\`\`bash
    git clone [REPOSITORY_URL]
    cd University_Project
    \`\`\`

2.  **Data Setup**
    * **Manual action is NOT required.**
    * The application includes an automated ingestion module. On the first launch, it checks the \`./data\` volume. If the dataset is missing, it automatically authenticates with KaggleHub, downloads the dataset, and persists it to the local disk.

---

## 4. How to Run

This project is optimized for Docker Compose.

1.  **Build the Environment**
    Build the Docker image and install dependencies.
    \`\`\`bash
    docker compose build
    \`\`\`

2.  **Execute the Pipeline**
    Start the container. This triggers the full lifecycle: Data Loading -> Feature Engineering -> Training -> Simulation.
    \`\`\`bash
    docker compose up
    \`\`\`

    *Output:* The terminal will display the training logs followed by a real-time simulation of sensor data streams. Anomalies will be flagged with a 🚨 alert.

3.  **Clean Up**
    To stop the execution and remove the container resources:
    \`\`\`bash
    docker compose down
    \`\`\`

---

## 5. Development & Debugging

For developers wishing to modify the pipeline or debug interactively without rebuilding:

**Interactive Shell**
To open a terminal inside the container environment:
\`\`\`bash
docker compose run --rm --entrypoint bash app
\`\`\`

**Workflow**
The project uses volume mapping, meaning changes made to \`from_model_to_production.py\` on the host machine are immediately reflected inside the container.
1.  Edit the script locally.
2.  Run \`python from_model_to_production.py\` inside the interactive shell.

---

## 6. Results

The model successfully identifies anomalies based on deviations in temperature, rotational speed (proxy for sound), and synthetic humidity. The final trained model is serialized and saved to \`results/isolation_forest_model.joblib\` for potential future deployment.