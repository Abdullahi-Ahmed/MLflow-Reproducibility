# Databricks notebook source
# MAGIC %run ./utilities

# COMMAND ----------

# MAGIC %md 
# MAGIC * make the nootbook idempotent

# COMMAND ----------

dbutils.fs.rm(projectPath, recurse= True)

# COMMAND ----------

# MAGIC %md
# MAGIC * Remember process file is already define in utilities notebook now we retrieve the Data that is in our dbfs #we have to be explicit with the file name we are interested in  
# MAGIC * We're putting in a new data path so that when we're manuplating we keep the original data.  
# MAGIC * And the last is the Registering for the metadata in the delta lake format

# COMMAND ----------

process_file(
  "Walmart_Store_sales.csv",
  silverPath,
  "Sales_data"
)
