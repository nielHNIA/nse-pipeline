import json
import joblib
from azureml.core.model import Model
from datetime import datetime, timedelta
import pandas as pd



# Called when the service is loaded
def init():
    global arima_model
    
    # Get the path to the registered ARIMA model file and load it
    model_path = Model.get_model_path('reliance_model3')
    arima_model = joblib.load(model_path)


# Called when a request is received
def run(raw_data):
    try:
        # Parse input JSON
        input_data = json.loads(raw_data)
        
        # Extract number of steps to forecast
        forecast_steps = input_data.get('steps', 1)  # Default to 1 if not provided

        # Ensure forecast_steps is a positive integer
        if not isinstance(forecast_steps, int) or forecast_steps <= 0:
            return json.dumps({"error": "Invalid 'steps' value. Must be a positive integer."})

        # Generate forecast
        forecast = arima_model.forecast(steps=forecast_steps)

        # Convert forecast to JSON format
        forecast_results = {"forecast": forecast.tolist()}

        return json.dumps(forecast_results)
    
    except Exception as e:
        return json.dumps({"error": str(e)})