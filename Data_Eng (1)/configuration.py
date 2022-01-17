# Databricks notebook source
# MAGIC %md
# MAGIC #Data Path
# MAGIC We are Identifying Data Path 

# COMMAND ----------

projectPath     = f"/Users/Sales/"
loadingPath     = projectPath + "loading/"
silverPath      = projectPath + "silver/"
goldPath        = projectPath + "gold/"

# COMMAND ----------

# MAGIC %md
# MAGIC #Configuring Database  
# MAGIC By Default databricks create for you database Called default  
# MAGIC But we want to create a new database to store our delta tables

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS Salesdb")
spark.sql(f"USE Salesdb");


# COMMAND ----------


