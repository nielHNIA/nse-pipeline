from azureml.core. import Dataset
import pandas as pd
# Access the CSV file in the container
csv_path = "azureml://subscriptions/a9be2bae-a7e2-4890-9232-48dc82eb3c71/resourcegroups/mlproject/workspaces/mlprojectexperiments/datastores/nsemldatastore/paths/final_nse.csv/"
dataset = Dataset.Tabular.from_delimited_files(path=(blob_datastore, csv_path))

# Convert the dataset to a pandas DataFrame (optional)
df = dataset.to_pandas_dataframe()
print(df.head())
