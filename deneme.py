import pca
from pyspark.sql import SparkSession
import pandas as pd

filePath = "C:\\Users\\M.O.ERBOY\\Desktop\\Datas"

spark = SparkSession.builder.appName("PCA").getOrCreate()

df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    filePath + "\\creditcard.csv")

print(df.head())
column_names = df.columns
print(column_names)


selected_colum_names = column_names[1:-1]
print(selected_colum_names)

confidence_level = float(input("Please enter the confidence level : (formatted like 0.1) : "))
print(confidence_level)


anomaly_value_indices = pca.anomalyDetectionWithPCA(dataFrame=df, selected_column_names=selected_colum_names, confidence_level=confidence_level)
print("detected anomaly count : " + str(len(anomaly_value_indices)))

anomaly_column = []

for ind in range(df.count()):
    anomaly_column.append(0)

for ind in range(len(anomaly_value_indices)):
    anomaly_column[anomaly_value_indices[ind]] = 1

#print(anomaly_column)

#real_anomaly_values = pd.read_csv(filepath_or_buffer=filePath+'\\muskY.csv', sep=',', header=None).values.tolist()
real_anomaly_values = df.select("Class").toPandas().values.tolist()

#print(real_anomaly_values)
print(len(real_anomaly_values))
print(len(anomaly_column))

real_anomaly_values_array = []
for ind in range(len(real_anomaly_values)):
    real_anomaly_values_array.append(real_anomaly_values[ind][0])

real_anomaly_indices = []
for ind in range(len(real_anomaly_values)):
    if real_anomaly_values_array[ind] == 1 :
        real_anomaly_indices.append(ind)

print("\n\n\n")
print("Calculated anomaly values : ")
print(anomaly_value_indices)
print("\n\nReal anomaly values : ")
print(real_anomaly_indices)

real_is_anomaly_calculated_is_anomaly = []
real_is_normal_calculated_is_normal = []
real_is_anomaly_calculated_is_normal = []
real_is_normal_calculated_is_anomaly = []

real_is_anomaly_calculated_is_anomaly_count = 0
real_is_normal_calculated_is_normal_count = 0
real_is_anomaly_calculated_is_normal_count = 0
real_is_normal_calculated_is_anomaly_count = 0

for ind in range(df.count()):
    real_is_anomaly_calculated_is_anomaly.append(0)
    real_is_anomaly_calculated_is_normal.append(0)
    real_is_normal_calculated_is_anomaly.append(0)
    real_is_normal_calculated_is_normal.append(0)



for ind in range(len(real_anomaly_values)):
    #print("[" + str(ind) + "] \t anomaly_column : " + str(anomaly_column[ind]) + " \t real_anomaly_value : " + str(real_anomaly_values_array[ind]))
    if anomaly_column[ind] == 0 and real_anomaly_values_array[ind] == 0:
        print("real_is_normal_calculated_is_normal")
        real_is_normal_calculated_is_normal_count += 1
        real_is_normal_calculated_is_normal.append(ind)

    if anomaly_column[ind] == 1 and real_anomaly_values_array[ind] == 1:
        print("real_is_anomaly_calculated_is_anomaly")
        real_is_anomaly_calculated_is_anomaly_count += 1
        real_is_anomaly_calculated_is_anomaly.append(ind)

    if anomaly_column[ind] == 0 and real_anomaly_values_array[ind] == 1:
        print("real_is_anomaly_calculated_is_normal")
        real_is_anomaly_calculated_is_normal_count += 1
        real_is_anomaly_calculated_is_normal.append(ind)

    if anomaly_column[ind] == 1 and real_anomaly_values_array[ind] == 0:
        print("real_is_normal_calculated_is_anomaly")
        real_is_normal_calculated_is_anomaly_count += 1
        real_is_normal_calculated_is_anomaly.append(ind)

print("real_is_anomaly_calculated_is_anomaly : " + str(len(real_is_anomaly_calculated_is_anomaly)))
print("real_is_normal_calculated_is_normal : " + str(len(real_is_normal_calculated_is_normal)))
print("real_is_anomaly_calculated_is_normal : " + str(len(real_is_anomaly_calculated_is_normal)))
print("real_is_normal_calculated_is_anomaly : " + str(len(real_is_normal_calculated_is_anomaly)))

print("\n\n\n\n")

print("real_is_anomaly_calculated_is_anomaly rate : " + str(len(real_is_anomaly_calculated_is_anomaly) / df.count()))
print("real_is_normal_calculated_is_normal rate : " + str(len(real_is_normal_calculated_is_normal) / df.count()))
print("real_is_anomaly_calculated_is_normal rate : " + str(len(real_is_anomaly_calculated_is_normal) / df.count()))
print("real_is_normal_calculated_is_anomaly rate : " + str(len(real_is_normal_calculated_is_anomaly) / df.count()))

print("\n\n\n\n")
print("with counts")

print("real_is_anomaly_calculated_is_anomaly_count : " + str(real_is_anomaly_calculated_is_anomaly_count))
print("real_is_normal_calculated_is_normal_count : " + str(real_is_normal_calculated_is_normal_count))
print("real_is_anomaly_calculated_is_normal_count : " + str(real_is_anomaly_calculated_is_normal_count))
print("real_is_normal_calculated_is_anomaly_count : " + str(real_is_normal_calculated_is_anomaly_count))

print("real_is_anomaly_calculated_is_anomaly rate : " + str(real_is_anomaly_calculated_is_anomaly_count / df.count()))
print("real_is_normal_calculated_is_normal rate : " + str(real_is_normal_calculated_is_normal_count / df.count()))
print("real_is_anomaly_calculated_is_normal rate : " + str(real_is_anomaly_calculated_is_normal_count / df.count()))
print("real_is_normal_calculated_is_anomaly rate : " + str(real_is_normal_calculated_is_anomaly_count / df.count()))
