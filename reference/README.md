# ModelManager Python Client
 
 Python client wrappers for the ModelManager backend API.
 
 This repository contains a lightweight HTTP client (built on `requests`) that:
 - Creates/updates/deletes usecases (projects) and models
 - Uploads datasets / model artifacts by file path
 - Supports AzureML and MLFlow workflows
 - Exposes optional governance / visualization endpoints (WIT, Netron, reports)
 - Supports versioning endpoints (Git/DVC integration)
 
 ## Table of Contents
 - [Install](#install)
 - [Authentication & Base URL](#authentication--base-url)
 - [Quickstart](#quickstart)
 - [Supported Client Classes](#supported-client-classes)
 - [Usecases (Projects)](#usecases-projects)
 - [Models](#models)
 - [Database Metadata (Tables/Fields)](#database-metadata-tablesfields)
 - [AzureML Integration](#azureml-integration)
 - [MLFlow Integration](#mlflow-integration)
 - [What-If / Netron / Data Distribution](#what-if--netron--data-distribution)
 - [Version Control (Git/DVC)](#version-control-gitdvc)
 - [Model Card](#model-card)
 - [Logging](#logging)
 
 ## Install
 
 This workspace does not include a top-level packaging file (e.g. `pyproject.toml`).
 
 - If you are using a published package:
 
   ```bash
   pip install mmanager
   ```
 
 - If you are using the code in this repo directly:
 
   - Add `mmanager-test/` to your `PYTHONPATH`, or
   - Install from a folder that contains a `setup.py` (for example under `versions/`).
 
 ## Authentication & Base URL
 
 All requests use an `Authorization` header in the format:
 
 ```text
 Authorization: secret-key <YOUR_SECRET_KEY>
 ```
 
 `base_url` should include scheme + host (+ port), for example:
 
 ```text
 http://localhost:8000
 https://api.example.com
 ```
 
 ## Quickstart
 
 ```python
 from mmanager.mmanager import Usecase, Model
 
 secret_key = "YOUR_SECRET_KEY"
 base_url = "http://localhost:8000"
 
 usecase = Usecase(secret_key=secret_key, base_url=base_url)
 model = Model(secret_key=secret_key, base_url=base_url)
 
 # Create a usecase ("project")
 usecase_resp = usecase.post_usecase({
     "name": "Fraud Detection",
     "description": "Detect fraud in transactions",
 })
 
 # Create a model (upload file paths)
 model_resp = model.post_model({
     "project": "<usecase-id>",
     "transformerType": "Classification",  # also commonly: "Regression", "Forecasting"
     "datasetinsertionType": "Manual",      # "Manual", "AzureML", "MLFlow"
     "training_dataset": "/path/to/train.csv",
     "test_dataset": "/path/to/test.csv",
     "pred_dataset": "/path/to/pred.csv",
     "actual_dataset": "/path/to/truth.csv",
     "model_file_path": "/path/to/model.pkl",
     "target_column": "Class",
 })
 ```
 
 ## Supported Client Classes
 
 The main client classes live in `mmanager/mmanager.py`:
 - **Core**: `ModelManager`
 - **Usecases**: `Usecase`
 - **Models**: `Model`
 - **Database metadata**: `TableInfo`, `FieldInfo`
 - **External databases**: `ExternalDatabase` (legacy), `RelatedDatabase`, `DatabaseLink`
 - **Integrations**: `MLFlow`, `VersionControl`
 - **What-if resources**: `WhatIf`
 - **Other**: `Applications`, `ReleaseTable`, `LLMCreds`, `ModelCard`
 
 ## Usecases (Projects)
 
 ```python
 from mmanager.mmanager import Usecase
 
 secret_key = "YOUR_SECRET_KEY"
 base_url = "http://localhost:8000"
 api = Usecase(secret_key=secret_key, base_url=base_url)
 
 # Create
 api.post_usecase({
     "name": "My Usecase",
     "description": "Short description",
 })
 
 # List (usecases uploaded by authenticated user)
 api.get_usecases()
 
 # Detail
 api.get_detail(usecase_id="<usecase-id>")
 
 # Update (supports optional 'image' and 'banner' file paths)
 api.patch_usecase({"description": "Updated"}, usecase_id="<usecase-id>")
 
 # Delete
 api.delete_usecase(usecase_id="<usecase-id>")
 
 # Models under a usecase
 api.get_models(usecase_id="<usecase-id>")
 
 # Load database cache
 api.load_cache(usecase_id="<usecase-id>")
 ```
 
 ### Forecasting usecase
 
 `post_usecase()` accepts optional `forecasting_fields` and `forecasting_feature_tabs` when `usecase_type == "Forecasting"`.
 
 ```python
 from mmanager.mmanager import Usecase
 
 api = Usecase(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 
 usecase_info = {
     "name": "Sales Forecast",
     "usecase_type": "Forecasting",
     "author": "Jane Doe",
     "description": "Forecast future sales",
 }
 
 forecasting_fields = {
     "forecasting_template": "two_conditions",
     "notification_emails": ["jane.doe@example.com"],
 }
 
 forecasting_feature_tabs = {
     "result_tab": True,
     "series_tab": True,
     "condition_tab": True,
     "performance_tab": True,
     "ab_testing_tab": True,
     "release_tab": True,
 }
 
 api.post_usecase(usecase_info, forecasting_fields, forecasting_feature_tabs)
 ```
 
 ## Models
 
 ### Create (upload by file path)
 
 `Model.post_model()` opens the files you pass (e.g. `training_dataset`) and uploads them. Ensure paths exist.
 
 ```python
 from mmanager.mmanager import Model
 
 api = Model(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 
 api.post_model({
     "project": "<usecase-id>",
     "transformerType": "Classification",  # also commonly: "Regression", "Forecasting"
     "datasetinsertionType": "Manual",      # "Manual", "AzureML", "MLFlow"
     "training_dataset": "/path/to/train.csv",
     "test_dataset": "/path/to/test.csv",
     "pred_dataset": "/path/to/pred.csv",
     "actual_dataset": "/path/to/truth.csv",
     "model_file_path": "/path/to/model.pkl",
     "target_column": "Class",
 })
 ```
 
 ### Update / delete / detail
 
 ```python
 from mmanager.mmanager import Model
 
 api = Model(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 
 api.patch_model({
     "target_column": "Class",
     "training_dataset": "/path/to/train_updated.csv",
 }, model_id="<model-id>")
 
 api.get_details(model_id="<model-id>")
 api.delete_model(model_id="<model-id>")
 ```
 
 ### Metrics and reports
 
 ```python
 from mmanager.mmanager import Model
 
 api = Model(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 
 api.get_latest_metrics(model_id="<model-id>", metric_type="<metric-type>")
 api.generate_report(model_id="<model-id>")
 api.get_all_reports(model_id="<model-id>")
 ```
 
 ## Database Metadata (Tables/Fields)
 
 ```python
 from mmanager.mmanager import TableInfo, FieldInfo
 
 base = dict(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 
 TableInfo(**base).post_table_info({
     "table_type": "actual",
     "table_name": "daily_act2",
     "db_link": 11,
 })
 
 FieldInfo(**base).post_field_info({
     "table_id": 9,
     "display_name": "actual2",
     "field_type": "",
     "field_name": "",
 })
 ```
 
 ## AzureML Integration
 
 AzureML flows use `ml_options["credPath"]` pointing to a JSON file. The client loads it and sends it as `amlCred`.
 
 Example credential file:
 
 ```json
 {
   "subscription_id": "<subscription-id>",
   "resource_group": "<resource-group>",
   "workspace_name": "<workspace-name>",
   "tenant-id": "<tenant-id>",
   "datastore_name": "<datastore-name>"
 }
 ```
 
 Fetch from AzureML:
 
 ```python
 from mmanager.mmanager import Model
 
 model_data = {
     "project": "<usecase-id>",
     "transformerType": "Classification",
     "datasetinsertionType": "AzureML",
     "target_column": "Class",
 }
 
 ml_options = {
     "credPath": "config.json",
     "fetchOption": ["Model"],
     "modelName": "<registered-model-name>",
     "dataPath": "<dataset-name>",
 }
 
 Model(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000").post_model(model_data, ml_options)
 ```
 
 ## MLFlow Integration
 
 1) Create MLFlow creds
 
 ```python
 from mmanager.mmanager import MLFlow
 
 api = MLFlow(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 api.post_mlflow_creds({
     "name": "Test Credentials",
     "tracking_uri": "https://example.mlflow.com",
     "mlflow_s3_endpoint_url": "https://sfo3.digitaloceanspaces.com",
     "artifact_path": "pathtomodelfiles",
     "aws_access_key_id": "",
     "aws_secret_access_key": "",
     "usecase": "<usecase-id>",
 })
 ```
 
 2) Download dataset/model pointers
 
 ```python
 from mmanager.mmanager import MLFlow
 
 api = MLFlow(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 resp = api.download_dataset_model(mlflow_cred_id="<mlflow-cred-id>", exp_name="<experiment-name>")
 ```
 
 3) Create model with `datasetinsertionType == "MLFlow"` and pass returned paths
 
 ## What-If / Netron / Data Distribution
 
 These endpoints return `IPython.display.IFrame` (intended for notebooks):
 
 ```python
 from mmanager.mmanager import Model
 
 api = Model(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 api.get_wit(model_id="<model-id>")
 api.get_netron(model_id="<model-id>")
 api.get_data_distribution(model_id="<model-id>")
 ```
 
 ## Version Control (Git/DVC)
 
 ```python
 from mmanager.mmanager import VersionControl
 
 api = VersionControl(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 
 api.git_config({
     "tag": "dvc_example",
     "git_repo": "https://github.com/jhondoe/example.git",
     "git_branch": "main",
     "username": "jhondoe",
     "email": "jhondoe@gmail.com",
     "access_token": "",
     "is_active": True,
 })
 
 api.dvc_set(git_config_id="<git-config-id>")
 api.get_version_tags(model_id="<model-id>", usecase_id="<usecase-id>")
 api.get_version_details(tag_name="<tag>")
 api.switch_data_version(model_id="<model-id>", usecase_id="<usecase-id>", tag_name="<tag>")
 api.export_datasets(model_id="<model-id>", usecase_id="<usecase-id>", tag_name="<tag>")
 ```
 
 ## Model Card
 
 ```python
 from mmanager.mmanager import ModelCard
 
 api = ModelCard(secret_key="YOUR_SECRET_KEY", base_url="http://localhost:8000")
 api.create_modelcard({"usecase_id": 118, "series": "ED Visits"})
 api.create_modelcard_bulk(usecase_id="118")
 api.get_modelcard_data({"usecase_id": 118, "model_id": 97})
 ```
 
 ## Logging
 
 Logs are written to `mmanager_log.log` (rotating) and also emitted to stdout.
 Set the log level via:
 
 ```bash
 export MMANAGER_LOG_LEVEL=DEBUG
 ```
