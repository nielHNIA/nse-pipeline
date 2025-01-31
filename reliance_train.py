from azureml.core import Workspace, Dataset, Datastore, Experiment, Run
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
#import joblib

ws = Workspace.from_config()
az_store = Datastore.get(ws, "azure_sdk_nse_blob")
az_dataset = Dataset.get_by_name(ws, "final_data_stocks")

new_run = Run.get_context()


df = az_dataset.to_pandas_dataframe()
reliance_df = df[['Date', 'RELIANCE_log_returns']]
reliance_df.set_index('Date', inplace=True)

train_size = int(0.8 * len(reliance_df))
train, test = reliance_df[:train_size], reliance_df[train_size:]

# You can change the (p, d, q) parameters as needed
model = ARIMA(train['RELIANCE_log_returns'], order=(5, 1, 0))  # Adjust (p, d, q) as needed
fitted_model = model.fit()

# Forecast for the test period
forecast = fitted_model.forecast(steps=len(test))

# Add forecast to test set for comparison
test['forecast'] = forecast.values

# Evaluate Model
mse = mean_squared_error(test['RELIANCE_log_returns'], test['forecast'])
#print(f"Mean Squared Error: {mse}")
new_run.log("Mean Squared Error:", mse)


new_run.complete()