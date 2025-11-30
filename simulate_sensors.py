import time
import requests
import random
import math

API_URL = "http://localhost:8000/predict"

def generate_stream():
    """Generates an infinite stream of sensor data."""
    t = 0
    while True:
        # 1. Generate Synthetic Data (Normal)
        # Temp: Sine wave centered at 300K (approx 27C)
        temp_k = 300 + (2 * math.sin(t * 0.1)) + random.uniform(-0.5, 0.5)
        
        # RPM: Normal around 1500
        rpm = 1500 + random.uniform(-50, 50)
        
        # 2. Inject Anomaly (Every 10th step)
        if t % 10 == 0 and t > 0:
            print("\n--- INJECTING ANOMALY (High RPM) ---")
            rpm = 2800 # Spike!
            
        yield {"temperature_k": temp_k, "rotational_speed_rpm": rpm}
        t += 1

def main():
    print(f"Connecting to Factory API at {API_URL}...")
    
    # Wait for server
    try:
        requests.get("http://localhost:8000/")
        print("Server is Online.")
    except:
        print("Server is offline. Is Docker running?")
        return

    stream = generate_stream()
    
    for sensor_data in stream:
        print(f"Sensor Sending: {sensor_data}")
        
        try:
            response = requests.post(API_URL, json=sensor_data)
            if response.status_code == 200:
                result = response.json()
                status_icon = "ERROR" if result['is_anomaly'] else "WORKING"
                print(f"API Response:   {status_icon} {result['status']} (Score: {result['anomaly_score']})")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Connection Error: {e}")
            
        time.sleep(1) # simulate 1Hz sensor rate

if __name__ == "__main__":
    main()
