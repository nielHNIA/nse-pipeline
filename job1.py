# Import the necessary Azure ML classes
from azureml.core import Workspace, Dataset, Datastore, Experiment, Environment, ScriptRunConfig
from azureml.core.environment import CondaDependencies
from azureml.core import Dataset, Datastore, Run
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Access the Azure ML workspace
ws = Workspace.from_config()

# Create/access the experiment from the workspace
new_experiment = Experiment(workspace=ws, name="Training_Script")

# -------------------------------------------------
# Create custom environment
# -------------------------------------------------
# Define the environment
myenv = Environment(name="MyEnvironment")

# Create the dependencies object with necessary packages (such as scikit-learn, pandas, statsmodels)
myenv_dep = CondaDependencies.create(
    conda_packages=['scikit-learn', 'statsmodels', 'pandas', 'numpy'],
    pip_packages=['azureml-sdk']
)

# Set the environment dependencies
myenv.python.conda_dependencies = myenv_dep

# Register the environment
myenv.register(ws)
# -------------------------------------------------

# Create a script configuration using the custom environment
script_config = ScriptRunConfig(
    source_directory=".",  # The directory where your script is located
    script="reliance_train.py",  # The training script to run
    environment=myenv  # The custom environment for the run
)

# Submit a new run using the ScriptRunConfig
new_run = new_experiment.submit(config=script_config)

# Monitor and wait for the run to complete
new_run.wait_for_completion(show_output=True)
