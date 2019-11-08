import pca
from pyspark.sql import SparkSession
import pyspark
import pandas as pd

spark = SparkSession.builder.appName("PCA").getOrCreate()

filePath = "C:\\Users\\M.O.ERBOY\\Desktop\\Datas"

df_pandas = pd.read_csv(filepath_or_buffer=filePath+'\\muskX.csv', sep=',', header=None)
df = spark.createDataFrame(df_pandas)
column_names = df.columns
print(column_names)
print(df.head(3))

confidence_level = float(input("Please enter the confidence level : (formatted like 0.1) : "))

anomaly_value_indices = pca.anomalyDetectionWithPCA(dataFrame=df, selected_column_names=column_names, confidence_level=confidence_level)

anomaly_column = []

for ind in range(df.count()):
    anomaly_column.append(0)

for ind in range(len(anomaly_value_indices)):
    anomaly_column[anomaly_value_indices[ind]] = 1

print(anomaly_column)

df_result = pd.read_csv(filepath_or_buffer=filePath+'\\muskY.csv', sep=',', header=None).values
print(df_result)

