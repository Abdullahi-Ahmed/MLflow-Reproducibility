# Databricks notebook source
artifact_path: model
databricks_runtime: 10.2.x-cpu-ml-scala2.12
flavors:
    python_function:
        env: conda.yaml
        loader_module: mlflow.sklearn
        model_path: model.pkl
        python_version: 3.8.10
    sklearn:
        pickled_model: model.pkl
        serialization_format: cloudpickle
        sklearn_version: 0.24.1
run_id: 1071d690ce374e6d8a2e12bdda2a106e
saved_input_example_info:
    artifact_path: input_example.json
    pandas_orient: split
    type: dataframe
signature:
    inputs: '[{"name": "Store", "type": "integer"}, {"name": "Date", "type": "string"},
    {"name": "Holiday_Flag", "type": "integer"}, {"name": "Temperature", "type": "double"},
    {"name": "Fuel_Price", "type": "double"}, {"name": "CPI", "type": "double"}, {"name":
    "Unemployment", "type": "double"}]'
    outputs: '[{"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1]}}]'
utc_time_created: '2022-01-17 20:29:00.995838'
