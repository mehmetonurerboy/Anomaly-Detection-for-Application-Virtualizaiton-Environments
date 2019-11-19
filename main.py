from pyspark.sql import SparkSession
import os
import pca
import data_cleaner



# This funciton find all .csv files' names
# PARAMETERS :
# filePath -> Which folder must be detected.
# It returns .csv files' names (without .csv parts)
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


spark = SparkSession.builder.appName("PCA").getOrCreate()

dataFilePath = "E:\\PERSONAL ITEMS\\LESSON ITEMS\\SEVENTH TERM\\BİTİRME\\Data\\AnomalyDetection\\dataset"
outputExcelPath = "E:\\PERSONAL ITEMS\\LESSON ITEMS\\SEVENTH TERM\\BİTİRME\\Analiz\\Excel Results"

#data = spark.read.option("header","true").option("inferSchema","true").format("csv")
df = spark.read.option("header","true").option("inferSchema","true").csv(dataFilePath + "\\1.csv")

from pyspark.ml.feature import PCA,StandardScaler,VectorAssembler

def PCA_Implementation(csvFileNames, columns, testNumbers, pcaKValues,dataPath, outputExcelPath) :
    for csv in range(len(csvFileNames)):
        df = spark.read.option("header","true").option("inferSchema","true").csv(dataPath + "\\" + csvFileNames[csv] + ".csv")
        for col in range(len(columns)):
            for test in range(testNumbers[col]):

                # VectorAssembler reduced columns that should be multiple columns to one column value
                assembler = VectorAssembler(
                    #inputCols=["ram_usage","cpu_percent","io_usage","network_usage"],
                    inputCols=columns[col],
                    outputCol="features"
                )

                # It transformed.
                output = assembler.transform(df)

                scaler = StandardScaler(inputCol="features",
                                        outputCol="scaledFeatures",
                                        withStd=True,
                                        withMean=False)

                # Compute summary statistics by fitting the StandardScaler
                scalerModel = scaler.fit(output)

                # Normalize each feature to have unit standard deviation.
                scaledData = scalerModel.transform(output)
                #scaledData.show()

                pca = PCA(k=pcaKValues[col][test], inputCol="features", outputCol="pca_features").fit(scaledData)

                pcaDf = pca.transform(scaledData)
                results = pcaDf.select("pca_features")
                #results.show()

                # Results extracted to excel file
                # Here excel file's name arrangement
                # Format :
                # [csv_file_name]_[column_name]_K=[pca_k_value].xlsx
                # OR for multiple column usage :
                # [csv_file_name]_[column_name]+[column_name]+[column_name]_K=[pca_k_value].xlsx

                # As a string, csv file name is assign as a first value.
                fileName = csvFileNames[csv]
                # According to multiple column name usage, they are added.
                for ind in range(len(columns[col])):
                    # For multiple column usage, '+' added to between 2 column name
                    if(ind > 0):
                        fileName += '+' + columns[col][ind]
                    # For first column name, '_' is used as reagent
                    else :
                        fileName += '_' + columns[col][ind]

                # Finally used k value at PCA algorithm and file tpye is added.
                fileName += '_K=' + str(pcaKValues[col][test]) + '.xlsx'

                # Here, file is extracted to target path.
                results.toPandas().to_excel(fileName)
                print(fileName + " is created at " + outputExcelPath)
                print('\n')

def wrongNumberException(arrayLength,textString) :
    value = int(input(textString))
    while(value < 0 or value > arrayLength):
        print("YOU ENTERED WRONG NUMBER! PLEASE ENTER VALID NUMBER.")
        value = int(input("Enter file name indis (the number that wrote on the left of its name) : "))
    return value

def valuePrinting(array) :
    for ind in range(len(array)):
        print('[' + str(ind) + '] : ' + array[ind])

def excelOutputs(pca_data_frame,csvFileName,column_names,pca_k_value,output_path):
    # Results extracted to excel file
    # Here excel file's name arrangement
    # Format :
    # [csv_file_name]_[column_name]_K=[pca_k_value].xlsx
    # OR for multiple column usage :
    # [csv_file_name]_[column_name]+[column_name]+[column_name]_K=[pca_k_value].xlsx

    # As a string, csv file name is assign as a first value.
    fileName = csvFileName
    # According to multiple column name usage, they are added.
    for ind in range(len(column_names)):
        # For multiple column usage, '+' added to between 2 column name
        if (ind > 0):
            fileName += '+' + column_names[ind]
        # For first column name, '_' is used as reagent
        else:
            fileName += '_' + column_names[ind]

    # Finally used k value at PCA algorithm and file tpye is added.
    fileName += '_K=' + str(pca_k_value) + '.xlsx'

    # Here, file is extracted to target path.
    pca_data_frame.toPandas().to_excel(output_path + '\\' + fileName)
    print(fileName + " is created at " + outputExcelPath)
    print('\n')

def PCA_calculation(dataFrame,csvFileName,column_values,pca_k_value,output_path):
    # VectorAssembler reduced columns that should be multiple columns to one column value
    assembler = VectorAssembler(
        # inputCols=["ram_usage","cpu_percent","io_usage","network_usage"],
        inputCols=column_values,
        outputCol="features"
    )

    # It transformed.
    output = assembler.transform(dataFrame)

    scaler = StandardScaler(inputCol="features",
                            outputCol="scaledFeatures",
                            withStd=True,
                            withMean=False)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(output)

    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(output)
    # scaledData.show()

    pca = PCA(k=pca_k_value, inputCol="features", outputCol="pca_features").fit(scaledData)

    pcaDf = pca.transform(scaledData)
    results = pcaDf.select("pca_features")
    # results.show()

    print("echo success")
    return results

# This function checks the equivalance between the number of indices that user entered and his/her indice enterance
def entered_column_count_control(input_string, column_count):
    if input_string.count(',') == (column_count-1) :
        return True
    elif input_string.count(',') < (column_count-1):
        print("YOU ENTERED INDICE VALUES BUT THERE ARE MINUS VALUES! PLEASE ENTER AGAIN!")
        return False
    else :
        print("YOU ENTERD EXTRA INDICE VALUES! PLEASE ENTER AGAIN!")
        return False

def entered_column_indice_control(input_string, dataFrame_column_count):
    indis = 0
    indice_values = input_string.split(',')

    while (indis < dataFrame_column_count) and (int(indice_values[indis]) >= 0) and (int(indice_values[indis]) < dataFrame_column_count) :
        indis = indis + 1

    if indis == dataFrame_column_count :
        return int(indice_values)
    else :
        print("YOU ENTERD WRONG INDICE VALUE! PLEASE ENTER AGAIN!")
        return []

def column_input_enterance(dataFrame, column_names, selected_count_number):
    input_string = input("Enter the indices of column that you want to use at PCA calculation. "
                         "[For seperating indices, you have to use comma(',') | Ex : 1,2,3] : ")

    while(entered_column_count_control(input_string,selected_count_number) == False) :
        input_string = input("Enter the indices of column that you want to use at PCA calculation. "
                             "[For seperating indices, you have to use comma(',') | Ex : 1,2,3] : ")

    entered_column_indices = entered_column_indice_control(input_string,len(column_names))
    while(entered_column_indices.count() == 0):
        input_string = input("Enter the indices of column that you want to use at PCA calculation. "
                             "[For seperating indices, you have to use comma(',') | Ex : 1,2,3] : ")
        entered_column_indices = entered_column_indice_control(input_string,column_names.count())

    selected_column_names = []

    for indis in range(len(entered_column_indices)) :
        selected_column_names.append(column_names[entered_column_indices[indis]])

    return entered_column_indices,selected_column_names

def obtain_anomaly_values(dataframe, selected_column_names, anomaly_indices):
    dataSet = dataframe.select(selected_column_names).toPandas().values.tolist()

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
    pandas_df.to_excel(arranged_data_path + '\\' + 'cleaned_data.xlsx')

csvFileNames = csvFileDetecter(dataFilePath)
#print(csvFileNames)

#print(df.columns)
relevant_cols = [['cpu_percent'],['ram_usage'],['network_usage'],['io_usage'],["cpu_percent","ram_usage"]]
#print(relevant_cols)

pca_test_numbers = [1,1,1,1,2]
pca_tests_k_values = [[1],[1],[1],[1],[1,2]]
#print(pca_test_number)
#print(pca_tests_k_values)

#PCA_Implementation(csvFileNames, relevant_cols, pca_test_numbers, pca_tests_k_values, dataFilePath , outputExcelPath)

"""
print("In data path, there are these files.\n")
valuePrinting(csvFileNames)
print("\n")
print("Select one of them.")
csvInput = wrongNumberException(len(csvFileNames),"Enter file name indis (the number that wrote on the left of its name) : ")

print('\n\n\n')

df = spark.read.option("header", "true").option("inferSchema", "true").csv(dataFilePath + "\\" + csvFileNames[csvInput] + ".csv")
column_names = df.columns
print("There are these columns in data.\n\nCOLUMN NAMES :")
valuePrinting(column_names)

col_count = int(input("Enter the number of column that you want to reduce : "))
col_values = []
for ind in range(col_count):
    col_values.append(column_names[wrongNumberException(len(column_names),"Enter the column indis : ")])
print(col_values)

pca_k_value = int(input("Enter the K value for PCA algorithm : "))

pca_result = PCA_calculation(df,csvFileNames[csvInput],col_values,pca_k_value,outputExcelPath)

print(pca_result.values)
"""

print("In data path, there are these files.\n")
valuePrinting(csvFileNames)
print("\n")
print("Select one of them.")
csvInput = wrongNumberException(len(csvFileNames),
                                "Enter file name indis (the number that wrote on the left of its name) : ")

print('\n\n\n')

df = spark.read.option("header", "true").option("inferSchema", "true").csv(
    dataFilePath + "\\" + csvFileNames[csvInput] + ".csv")
column_names = df.columns
print("There are these columns in data.\n\nCOLUMN NAMES :")
valuePrinting(column_names)

col_count = int(input("Enter the number of column that you want to use in calculation : "))
#selected_column_names = column_input_enterance(df, column_names, col_count)
selected_column_names = []
selected_column_indices = input("Enter indice values : ").split(',')
for indis in range(len(selected_column_indices)):
    selected_column_names.append(column_names[int(selected_column_indices[indis])])

confidence_level = float(input("Please enter the confidence level : (formatted like 0.1) : "))

anomaly_value_indices = pca.anomalyDetectionWithPCA(dataFrame=df, selected_column_names=selected_column_names, confidence_level=confidence_level)

anomaly_column = []

for ind in range(df.count()):
    anomaly_column.append(0)

for ind in range(len(anomaly_value_indices)):
    anomaly_column[anomaly_value_indices[ind]] = 1

print(anomaly_column)

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

df_new = cleaned_from_anomalies_dataframe(df,column_names,selected_column_indices,anomaly_values,anomaly_value_indices,mean,std,output_cleaned_data_path)


