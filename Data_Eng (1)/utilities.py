# Databricks notebook source
# MAGIC %md
# MAGIC We're running our configuration nootbook to identfy the path and our new database to use in this notebook

# COMMAND ----------

# MAGIC %run ./configuration

# COMMAND ----------

# MAGIC %md
# MAGIC The csv file is provided in this repo  
# MAGIC import data to your databricks following this steps
# MAGIC * Click the Icon "Data" in the sidebar.  
# MAGIC * Click the DBFS button at the top of the page.  
# MAGIC * Click the Upload button at the top of the page.  
# MAGIC * On the Upload Data to DBFS dialog, (optionally) select a target directory or enter a new one. for me I use default directory  
# MAGIC * In the Files box, drag and drop or use the file browser to select the local file to upload.  
# MAGIC   

# COMMAND ----------

# MAGIC %md
# MAGIC A retrieve data function that retrieve the file and puts into loadingPath.  
# MAGIC We create another function that read the  file into spark dataframe and then loads into a delta while saving the delta table path.  
# MAGIC The process file function combines all the above function and register in as a table in the created database.

# COMMAND ----------

def retrieve_data(file: str) -> bool:
  """Load the data from your local """
 
  
  loadPath = "dbfs:/FileStore/tables/" + file
  "Load a csv file as a parquet file"
  dbfsPath   = loadingPath + file
  dbutils.fs.cp(loadPath , dbfsPath, recurse= True)
  return True

def load_delta_table(file: str, delta_table_path: str) -> bool:
  "Load a  file as a Delta table."
  parquet_df = spark.read.format("csv").options(header = "True", inferSchema = "True").load(loadingPath + file)
  parquet_df.write.format("delta").save(delta_table_path)
  return True

def process_file(file_name: str, path: str,  table_name: str) -> bool:
  """
  1. retrieve file
  2. load as delta table
  3. register table in the metastore
  """

  retrieve_data(file_name)
  print(f"Retrieve {file_name}.")

  load_delta_table(file_name, path)
  print(f"Load {file_name} to {path}")

  spark.sql(f"""
  DROP TABLE IF EXISTS {table_name}
  """)

  spark.sql(f"""
  CREATE TABLE {table_name}
  USING DELTA
  LOCATION "{path}"
  """)

  print(f"Register {table_name} using path: {path}")


# COMMAND ----------


