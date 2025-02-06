import json
from azureml.core import Workspace, Datastore, Dataset
import pandas as pd
from datetime import timedelta
import numpy as np

ws = Workspace.from_config()


dataset_name = "final_data_stocks"  # Replace with your dataset name
dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)

# Convert to Pandas DataFrame (if it's a tabular dataset)
df = dataset.to_pandas_dataframe()
print(df.head())




# Access the service end points
print("Accessing the service end-points")
service = ws.webservices['stock-service11']

# Prepare input data with number of forecast steps
steps = 5  # Number of forecast steps
json_data = json.dumps({"steps": steps})

# Call the web service
print("Calling the service...")
response = service.run(input_data=json_data)

# Collect and convert the response into a local variable
forecast_results = json.loads(response)

# Now we need to get the last known stock price from your data

# Get the last known stock price from the dataset
last_known_price = df['Reliance'].iloc[-1]  # Assuming 'Close' column contains the stock prices

# Generate forecast dates
last_known_date = pd.to_datetime(df['Date'].max())  # Last date in the dataset
forecast_dates = [(last_known_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, steps + 1)]

# Convert log-returns back to stock prices using the formula:
forecast_prices = [last_known_price * np.exp(log_return) for log_return in forecast_results['forecast']]

# Add dates and prices to the forecasted results
forecast_with_dates_and_prices = {"forecast": []}

for i, price in enumerate(forecast_prices):
    forecast_with_dates_and_prices['forecast'].append({
        "date": forecast_dates[i],
        "price": price
    })

# Print the forecasted values with dates and original stock prices
print("\nForecasted values with dates and original stock prices:")
for item in forecast_with_dates_and_prices['forecast']:
    print(f"Date: {item['date']}, Price: {item['price']}")
