from pyspark.sql import SparkSession
import pyspark
import os



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
