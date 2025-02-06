# %% [markdown]
# 

# %%
from azureml.core import Workspace, Dataset, Datastore, Experiment, Run



# %%
ws = Workspace.from_config()
az_store = Datastore.get(ws, "azure_sdk_nse_blob")
az_dataset = Dataset.get_by_name(ws, "final_data_stocks")



# %%
new_run = Run.get_context()

# %%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = az_dataset.to_pandas_dataframe()
df.set_index("Date", inplace=True)
df = df.asfreq('D')
df.fillna(method='ffill', inplace=True)
#df = df.asfreq('D')
# Extract individual log return series
print(df.info())

print(df.isna().sum())




# %%
# Split data into train (90%) and test (10%)
train_size = int(len(df) * 0.9)
train, test = df.iloc[:train_size], df.iloc[train_size:]

model = ARIMA(train["Reliance_log"], order=(2,1,0))  # Change order based on ACF/PACF analysis
model_fit = model.fit()

# Predict on test set
test_predictions = model_fit.forecast(steps=len(test))

# Model evaluation
mae = mean_absolute_error(test["Reliance_log"], test_predictions)
rmse = np.sqrt(mean_squared_error(test["Reliance_log"], test_predictions))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# %%
# Forecast next 5 months (approximately 150 days)
future_steps = 150
future_forecast = model_fit.forecast(steps=future_steps)

# Create future dates
future_dates = pd.date_range(df.index[-1], periods=future_steps + 1, freq="D")[1:]

# Save forecast data to CSV
forecast_df = pd.DataFrame({"date": future_dates, "predicted_stock": future_forecast})

#forecast_df.to_csv("reliance_stock_forecast.csv", index=False)

#print("5-month forecast saved as 'reliance_stock_forecast.csv'")

# Save the trained ARIMA model as .pkl file
#joblib.dump(model_fit, "reliance_arima_model.pkl")
#print("ARIMA model saved as 'reliance_arima_model.pkl'")

# %%
#experiment2 = Experiment(workspace=ws, name="Stocks-exp02")

# %%
#running an experiment
#new_run = experiment.start_logging(snapshot_directory=None)

# %%
#df = az_dataset.to_pandas_dataframe()
#total_observations = len(df)

# %%
new_run.log("total observations", mae)

# %%
new_run.complete()

# %%



