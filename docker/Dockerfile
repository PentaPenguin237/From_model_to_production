# Use a lightweight and official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project's source code into the container
COPY . .

# Define container-relative paths as environment variables
ENV DATA_DIR="/data"
ENV RESULTS_DIR="/results"

# Set the default command to run when the container starts
CMD ["python", "from_model_to_production.py"]
