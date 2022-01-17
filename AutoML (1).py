# Databricks notebook source
# MAGIC %run ./Data_Eng/configuration

# COMMAND ----------

#Reading from the created table in data enigneering and put it into dataframe
data = spark.read.table("sales_data")

# COMMAND ----------

#selecting only the features that we are using in our modeling and putting into a variable 
factors_df = data.select(["Weekly_Sales","Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"])
(factors_df.write
 .format("delta")
 .mode("overwrite")
 .save(goldPath + "selected_factors")
)

# COMMAND ----------

#Since We wanna keep the original data we used in our modeling we are putting the fetured and agg data into goldPath

df_loaded = (data.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable("salesdb.gold_df")
 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

df_loaded = (spark.read
 .table("salesdb.gold_df")
 .toPandas()
)
transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean for consistency

# COMMAND ----------

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputer", SimpleImputer(strategy="mean"))
])

transformers.append(("numerical", numerical_pipeline, ["CPI", "Fuel_Price", "Unemployment", "Store", "Temperature"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(handle_unknown="ignore")

transformers.append(("onehot", one_hot_encoder, ["Date", "Holiday_Flag"]))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature standardization
# MAGIC Scale all feature columns to be centered around zero with unit variance.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

df_loaded.columns

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = df_loaded.drop(['Weekly_Sales'], axis=1)
split_y = df_loaded['Weekly_Sales']

# Split out train data
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, random_state=979224757)

# Split remaining data equally for validation and test
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, random_state=979224757)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train regression model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/3359699395656437/s?orderByKey=metrics.%60val_r2_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

import mlflow
import pandas as pd
from xgboost import XGBRegressor
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display='diagram')

xgb_regressor = XGBRegressor(
  colsample_bytree=0.6669908680393172,
  learning_rate=0.22560423988961822,
  max_depth=3,
  min_child_weight=7,
  n_estimators=328,
  n_jobs=100,
  subsample=0.22601343271363417,
  verbosity=0,
  random_state=979224757,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("regressor", xgb_regressor),
])

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
])

mlflow.sklearn.autolog(disable=True)
X_val_processed = pipeline.fit_transform(X_val, y_val)

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="xgboost") as mlflow_run:
    model.fit(X_train, y_train, regressor__early_stopping_rounds=5, regressor__eval_set=[(X_val_processed,y_val)], regressor__verbose=False)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    xgb_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    xgb_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    xgb_val_metrics = {k.replace("val_", ""): v for k, v in xgb_val_metrics.items()}
    xgb_test_metrics = {k.replace("test_", ""): v for k, v in xgb_test_metrics.items()}
    display(pd.DataFrame([xgb_val_metrics, xgb_test_metrics], index=["validation", "test"]))

# COMMAND ----------

# Patch requisite packages to the model environment YAML for model serving
import os
import shutil
import uuid
import yaml

None

import xgboost
from mlflow.tracking import MlflowClient

xgb_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], str(uuid.uuid4())[:8])
os.makedirs(xgb_temp_dir)
xgb_client = MlflowClient()
xgb_model_env_path = xgb_client.download_artifacts(mlflow_run.info.run_id, "model/conda.yaml", xgb_temp_dir)
xgb_model_env_str = open(xgb_model_env_path)
xgb_parsed_model_env_str = yaml.load(xgb_model_env_str, Loader=yaml.FullLoader)

xgb_parsed_model_env_str["dependencies"][-1]["pip"].append(f"xgboost=={xgboost.__version__}")

with open(xgb_model_env_path, "w") as f:
  f.write(yaml.dump(xgb_parsed_model_env_str))
xgb_client.log_artifact(run_id=mlflow_run.info.run_id, local_path=xgb_model_env_path, artifact_path="model")
shutil.rmtree(xgb_temp_dir)

# COMMAND ----------

shap_enabled = True
if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(200, len(X_train.index)))

    # Sample a single example from the validation set to explain. Increase the sample size and rerun for more thorough results.
    example = X_val.sample(n=1)

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example)

# COMMAND ----------

import mlflow
import mlflow.sklearn
from shap import KernelExplainer, summary_plot
from sklearn.model_selection import train_test_split
import pandas as pd
import shap

mlflow.autolog(disable=True)

# be sure to change the following run URI to match the best model generated by AutoML
model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
#model_uri = "runs:/[your run ID here!]/model"

sample = spark.read.table("salesdb.gold_df").sample(0.01, seed=42).toPandas()
data = sample.drop(["Weekly_Sales"], axis=1)
labels = sample["Holiday_Flag"]
X_background, X_example, _, y_example = train_test_split(data, labels, train_size=0.25, random_state=42, stratify=labels)

model = mlflow.sklearn.load_model(model_uri)

predict = lambda x: model.predict(pd.DataFrame(x, columns=X_background.columns))
explainer = KernelExplainer(predict, X_background)
shap_values = explainer.shap_values(X=X_example, nsamples=100)

# COMMAND ----------

summary_plot(shap_values, features=X_example)

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC # Automated Testing
# MAGIC 
# MAGIC This section is derived from the auto-generated batch inference notebook, from the MLflow Model Registry. It loads the latest Staging candidate model and, in addition to running inference on a data set, assesses model metrics on that result and from the training run. If successful, the model is promoted to Production. This is scheduled to run as a Job, triggered manually or on a schedule - or by a webhook set up to respond to state changes in the registry.
# MAGIC 
# MAGIC Load the model and set up the environment it defines:

# COMMAND ----------

from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os

local_path = ModelsArtifactRepository(f"models:/xgbSales/staging").download_artifacts("")

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

# MAGIC %md
# MAGIC Assert that the model accuracy was at least 80% at training time:

# COMMAND ----------

import mlflow.tracking

client = mlflow.tracking.MlflowClient()
latest_model_detail = client.get_latest_versions("xgbSales", stages=['Staging'])[0]
accuracy = mlflow.get_run(latest_model_detail.run_id).data.metrics['training_accuracy_score']
print(f"Training accuracy: {accuracy}")
assert(accuracy >= 0.8)
