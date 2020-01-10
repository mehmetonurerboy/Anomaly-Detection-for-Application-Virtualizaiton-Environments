from pyspark.sql import SparkSession
import os
import pca
import lr
import data_cleaner
import timeit
import time
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import context as sqlContext
import pandas as pd



# This funciton find all .csv files' names
# INPUTS :
# filePath -> Which folder must be detected.

# OUTPUT
# onlyFiles : Returns .csv files' names (without .csv parts)
def csvFileDetecter (filePath):
    # This code is extract all files at the path named filePath
    onlyFiles = [f for f in os.listdir(filePath) if os.path.isfile(os.path.join(filePath, f))]

    # indis is used for scanning onlyFiles elements
    indis = 0
    # Some file that is not .csv file must be deleted on onlyFiles array
    # remove() function is used for deleting file that is not .csv file.
    # But this function changes the length of array. So, while loop is used.
    while indis < len(onlyFiles):
        # .csv file is detected. So, it is removed but indis value must be the same.
        if(onlyFiles[indis].find('.csv') == -1):
            onlyFiles.remove(onlyFiles[indis])
        # Otherwise indis value should be increased.
        else :
            indis = indis + 1

    # This function's return used for detecting relevant excels' name
    # So, we have to clear '.csv' parts
    for indis in range(len(onlyFiles)):
        onlyFiles[indis] = onlyFiles[indis][:onlyFiles[indis].find('.csv')]

    return onlyFiles

# spark session is began.
spark = SparkSession.builder.appName("PCA").getOrCreate()

# The path of data and the path of outputs are declared.
directory_path = os.getcwd()
print("directory path")
print(directory_path)





from pyspark.ml.feature import PCA,StandardScaler,VectorAssembler

# This function
def wrongNumberException(arrayLength,textString) :
    value = int(input(textString))
    while(value < 0 or value > arrayLength):
        print("YOU ENTERED WRONG NUMBER! PLEASE ENTER VALID NUMBER.")
        value = int(input("Enter file name indis (the number that wrote on the left of its name) : "))
    return value

# This function provides the printing data of array

# INPUT
# array : The array that will print.

# OUTPUT
# console print out of data.
def valuePrinting(array) :
    for ind in range(len(array)):
        print('[' + str(ind) + '] : ' + array[ind])

# This function collect the content of data that is labelled as anomaly by PCA algorithm.

# INPUT
# dataFrame :
# selected_column_names :
# anomaly_indices : The indices of anomaly values came from PCA algorithm.

# OUTPUT
# anomaly_values : The 2-D array that keeps data that is labelled as anomaly by PCA.
def obtain_anomaly_values(dataframe, selected_column_names, anomaly_indices):
    # We extract the data from dataFrame according to selected_column_names and we transform it to list.
    dataSet = dataframe.select(selected_column_names).toPandas().values.tolist()

    # anomaly_values keeps the return anomaly data values.
    anomaly_values = [[]]
    for indis in range(len(anomaly_indices)):
        anomaly_values.insert(indis,dataSet[anomaly_indices[indis]])

    anomaly_values.remove(anomaly_values[len(anomaly_values)-1])

    return anomaly_values

def out_bound_of_normal_distribution(anomaly_values, mean, std):
    out_bound_of_anomaly_indices = []

    for row in range(len(anomaly_values)):
        is_out_of_bound = 0
        for indis in range(len(mean)):
            if anomaly_values[row][indis] < (mean[indis] - std[indis]) or \
                    anomaly_values[row][indis] > (mean[indis] + std[indis]):
                is_out_of_bound = 1
        if is_out_of_bound == 1:
            out_bound_of_anomaly_indices.append(row)

    return out_bound_of_anomaly_indices

def cleaned_from_anomalies_dataframe(dataframe, column_names, selected_column_indice, anomaly_values, anomaly_indices, mean_array, std_array, arranged_data_path):
    pandas_df = dataframe.toPandas()

    for row in range(len(anomaly_values)):
        for indis in range(len(selected_column_indice)):
            if anomaly_values[row][indis] < (mean_array[indis] - std_array[indis]):
                #pandas_df.iloc[anomaly_indices[row]][selected_column_indice[indis]] = mean_array[indis] - std_array[indis]
                pandas_df.at[anomaly_indices[row],selected_column_indice[indis]] = mean_array[indis] - std_array[indis]
            if anomaly_values[row][indis] > (mean_array[indis] + std_array[indis]):
                #pandas_df.iloc[anomaly_indices[row]][selected_column_indice[indis]] = mean_array[indis] + std_array[indis]
                pandas_df.at[anomaly_indices[row],selected_column_indice[indis]] = mean_array[indis] + std_array[indis]
    pandas_df.to_excel(arranged_data_path + '/' + 'cleaned_data.xlsx')


csvFileNames = csvFileDetecter(directory_path)

print("In data path, there are these files.\n")
valuePrinting(csvFileNames)
print("\n")
print("Select one of them.")
csvInput = wrongNumberException(len(csvFileNames),
                                "Enter file name indis (the number that wrote on the left of its name) : ")

print('\n\n\n')

df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    directory_path + "/" + csvFileNames[csvInput] + ".csv")
column_names = df.columns
print("There are these columns in data.\n\nCOLUMN NAMES :")
valuePrinting(column_names)

col_count = int(input("Enter the number of column that you want to use in calculation : "))
#selected_column_names = column_input_enterance(df, column_names, col_count)
selected_column_names = []
selected_column_indices = input("Enter indice values : ").split(',')
for indis in range(len(selected_column_indices)):
    selected_column_names.append(column_names[int(selected_column_indices[indis])])

time_col = int(input("Enter the indice of dimension that represents the time value : "))
time_col_name = column_names[time_col]



confidence_level = float(input("Please enter the confidence level : (formatted like 0.1) : "))





start_time = time.time()
anomaly_value_indices = pca.anomalyDetectionWithPCA(dataFrame=df, selected_column_names=selected_column_names, confidence_level=confidence_level, time_column_name=time_col_name)
end_time = time.time()
print("PCA performance measurements : ")
execution_time_of_pca = end_time - start_time
print("PCA elapsed time average : ")
print(execution_time_of_pca)

anomaly_column = []
normal_column = []

for ind in range(df.count()):
    anomaly_column.append(0)

last_element = 0
for indis in range(len(anomaly_value_indices)):
    for element in range(anomaly_value_indices[indis] - last_element - 1):
        normal_column.append(element+last_element+1)
    last_element = anomaly_value_indices[indis]

for element in range(df.count() - last_element - 1):
    normal_column.append(last_element+element+1)

last_anomaly_indice = 0

for ind in range(len(anomaly_value_indices)):
    anomaly_column[anomaly_value_indices[ind]] = 1

print(anomaly_column)
print(normal_column)
print("len : " + str(len(anomaly_value_indices) + len(normal_column)))

df_pandas = df.select(selected_column_names).toPandas()
#print(df_pandas)
print(type(df_pandas))
df_pandas['outlier_status'] = anomaly_column

print(df_pandas)

mean, std = data_cleaner.getStatisticsOfDataFrame(df,selected_column_names)
print("\n\n")
print("Mean values : ")
print(mean)
print("\n\n")
print("standard deviation values : ")
print(std)

print("\n\n")
for indis in range(len(selected_column_names)):
    print("Column | " + selected_column_names[indis] + " : ")
    print("Lower limit : " + str(mean[indis] - 3 * std[indis]))
    print("Upper limit : " + str(mean[indis] + 3 * std[indis]))
    print("\n")

print("\n\n\n")
print("Anomaly values : ")


anomaly_values = obtain_anomaly_values(dataframe=df, selected_column_names=selected_column_names, anomaly_indices=anomaly_value_indices)
for indis in range(len(anomaly_values)):
    print(anomaly_values[indis])

out_bound_anomalies = out_bound_of_normal_distribution(anomaly_values,mean,std)
print("\n\n")
print("len of anomalies : " + str(len(anomaly_values)))
print("\n\n")
print("len of out bound anomalies : " + str(len(out_bound_anomalies)))

output_cleaned_data_path = "E:\\PERSONAL ITEMS\\LESSON ITEMS\\SEVENTH TERM\\BİTİRME\\Analiz\\Temizlenmis Veri"

#df_new = cleaned_from_anomalies_dataframe(df,column_names,selected_column_indices,anomaly_values,anomaly_value_indices,mean,std,output_cleaned_data_path)


from pyspark.sql import Row

df_x4 = spark.createDataFrame([Row(**{'Anomaly': x}) for x in anomaly_column])

def flatten_row(r):
    r_ =  r.features.asDict()
    r_.update({'row_num': r.row_num})
    return Row(**r_)

def add_row_num(df):
    df_row_num = df.rdd.zipWithIndex().toDF(['features', 'row_num'])
    df_out = df_row_num.rdd.map(lambda x : flatten_row(x)).toDF()
    return df_out

df = add_row_num(df)
df_x4 = add_row_num(df_x4)
df = df.join(df_x4, on='row_num').drop('row_num')


#anomaly_df = spark.createDataFrame(pd.DataFrame(added_anomalies))
#print(anomaly_df.head())

print(df)
print(df.columns)

print(df_pandas)


print("normal columns")
print(normal_column)

print("anomaly columns")
print(anomaly_value_indices)


print(" ---------------------- %80 TRAIN DATA - %20 TEST DATA OPTIONS ---------------------- ")

train_dataset_indices = []
size_limit_for_normal = int(len(normal_column) * 0.8)
size_limit_for_anomaly = int(len(anomaly_value_indices) * 0.8)
print("size limiter")
print(size_limit_for_normal)
print(size_limit_for_anomaly)
train_dataset_indices.extend(normal_column[:size_limit_for_normal])
train_dataset_indices.extend(anomaly_value_indices[:size_limit_for_anomaly])
#train_dataset_indices.sort()

test_dataset_indices = []
test_dataset_indices.extend(normal_column[size_limit_for_normal:])
test_dataset_indices.extend(anomaly_value_indices[size_limit_for_anomaly:])
#test_dataset_indices.sort()

print("size of train")
print(len(train_dataset_indices))
print("size of test")
print(len(test_dataset_indices))
print("size of dataframe")
print(len(anomaly_values))

print("train indices :")
print(train_dataset_indices)
print("test indices : ")
print(test_dataset_indices)

df_train = df_pandas.take(train_dataset_indices)
print(df_train)
print('\n\n\n')
real_df_test = df_pandas.take(test_dataset_indices)


from pyspark.sql.types import *

pyspark_train_df = spark.createDataFrame(df_train)
df_test = real_df_test[selected_column_names]
print(df_test)
pyspark_test_df = spark.createDataFrame(df_test)
print("train dataset colummns")
print(pyspark_train_df.columns)

"""
print("train")
print(pyspark_train_df)
print("\n\n")
print("test")
print(pyspark_test_df)
"""

print("Logistic Regression performance measurement : ")
start_time = time.time()
logistic_regression_accuracy_summary = lr.calculateLogisticRegression(pyspark_train_df,pyspark_test_df,real_df_test)
end_time = time.time()
logistic_regression_exec_time = end_time - start_time
print("Execution time of logistic regression : ")
print(logistic_regression_exec_time)

accuracy_rate = (logistic_regression_accuracy_summary[3] + logistic_regression_accuracy_summary[0]) / len(test_dataset_indices)
print("accuracy rate  :")
print(accuracy_rate)

f_score = 2 * logistic_regression_accuracy_summary[0] / (2 * logistic_regression_accuracy_summary[0] + logistic_regression_accuracy_summary[1] + logistic_regression_accuracy_summary[2])
print("f-score : ")
print(f_score)


print("-------------------------------------------------------------------------------------------------------------------------")
size_limit_for_normal = int(len(normal_column) * 0.1)
size_limit_for_anomaly = int(len(anomaly_value_indices) * 0.1)

print("-------len--------")
print("normal " + str(len(normal_column)))
print("anomaly" + str(len(anomaly_value_indices)))
print("-----------------")

print("normal")
print(size_limit_for_normal)
print(normal_column)
print("anomaly")
print(size_limit_for_anomaly)
print(anomaly_value_indices)
print("--------")
data_indices = [[]]
for indis in range(9):
    temp = []
    temp.extend(normal_column[indis*size_limit_for_normal:(indis+1)*size_limit_for_normal])
    temp.extend(anomaly_value_indices[indis*size_limit_for_anomaly:(indis+1)*size_limit_for_anomaly])
    temp.sort()
    data_indices.append(temp)

temp = []
temp.extend(normal_column[9*size_limit_for_normal:])
print("-----temp------")
print(temp)
temp.extend(anomaly_value_indices[9*size_limit_for_anomaly:])
temp.sort()
print("\n\n")
data_indices.append(temp)
data_indices.remove(data_indices[0])

print(data_indices)

for indis in range(len(data_indices)):
    print(data_indices[indis])
    print("size : " + str(len(data_indices[indis])))

confusion_matrix_values = [[]]
accuracy_rates = []
f_scores = []
exec_time = []

for indis in range(len(data_indices)):
    train_dataset_indices = []
    test_dataset_indices = []

    # Obtaining train and test dataset indexes.
    for indis2 in range(len(data_indices)):
        if indis != indis2 :
            train_dataset_indices.extend(data_indices[indis])
        else:
            test_dataset_indices.extend(data_indices[indis])

    # Obtained data from pandas DataFrame object according to index informations
    df_train = df_pandas.take(train_dataset_indices)
    real_df_test = df_pandas.take(test_dataset_indices)

    pyspark_train_df = spark.createDataFrame(df_train)
    df_test = real_df_test[selected_column_names]
    pyspark_test_df = spark.createDataFrame(df_test)

    start_time = time.time()
    logistic_regression_accuracy_summary = lr.calculateLogisticRegression(pyspark_train_df,pyspark_test_df,real_df_test)
    end_time = time.time()
    exec_time.append(end_time - start_time)

    accuracy_rate = (logistic_regression_accuracy_summary[3] + logistic_regression_accuracy_summary[0]) / len(
        test_dataset_indices)
    print("accuracy rate : " + str(accuracy_rate))

    f_score = 2 * logistic_regression_accuracy_summary[0] / (
                2 * logistic_regression_accuracy_summary[0] + logistic_regression_accuracy_summary[1] +
                logistic_regression_accuracy_summary[2])
    print("f-score : " + str(f_score))

    confusion_matrix_values.append(logistic_regression_accuracy_summary)
    accuracy_rates.append(accuracy_rate)
    f_scores.append(f_score)

confusion_matrix_values.remove(confusion_matrix_values[0])

ave_confusion_matrix = []
print("about confusion matrix")
print(len(confusion_matrix_values[0]))
print(len(confusion_matrix_values))

for indis in range(len(confusion_matrix_values[0])):
    sum = 0
    for indis2 in range(len(confusion_matrix_values)):
        sum += confusion_matrix_values[indis2][indis]
    print("sum" + str(sum/10))
    ave_confusion_matrix.append(sum/(len(confusion_matrix_values)))

print("ave confusion matrix")
print(ave_confusion_matrix)
print("----------------------")

sum = 0
for indis in range(len(accuracy_rates)):
    sum += accuracy_rates[indis]

ave_accuracy_rate = sum / len(accuracy_rates)

print("ave accuracy rates")
print(ave_accuracy_rate)
print("----------------------")

sum = 0
for indis in range(len(f_scores)):
    sum += f_scores[indis]

ave_f_score = sum / len(f_scores)


print("ave f-score")
print(ave_f_score)
print("----------------------")

sum = 0
for indis in range(len(exec_time)):
    sum += exec_time[indis]

ave_exec_time = sum / len(exec_time)

print("ave exec time")
print(ave_exec_time)







"""
columns = []
columns.extend(selected_column_names)
columns.append("Anomaly")
df_lr = df.select([c for c in df.columns if c in columns]).show()
print(df_lr)
"""

# Configuration an ML pipeline, which consists of tree stages : tokenizer, hashingTF and lr.





