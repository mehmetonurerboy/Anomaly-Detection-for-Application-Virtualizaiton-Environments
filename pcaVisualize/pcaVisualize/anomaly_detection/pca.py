from numpy import linalg as LA
import numpy as np
import pandas as pd
import datetime
from statistics import mean as getMean

# This function return the normalized data.

# INPUTS
# dataFrame : The spark.DataFrame object that keeps normal data
# selected_column_names : The column names array that PCA calculation is implemented on.

# OUTPUTS
# queried_matrix : This return data keeps the normalized data. (The data that substracted from the column mean value)
def obtain_normalized_data(dataFrame, selected_column_names):
    # The data that is located on selected columns is extracted.
    queried_values = dataFrame.select(selected_column_names).rdd.flatMap(lambda x: x).collect()

    # This queried_values is an array. I have to obtaint an 2-D array structure. np.reshape helps for this aim.
    queried_matrix = np.reshape(np.array(queried_values), (-1, len(selected_column_names)))

    # column_means keeps the mean value for each selected column data.
    column_means = []

    # np.mean calculates this mean values and this values are assigned to column_means array.
    for column in range(len(selected_column_names)):
        column_means.append(np.mean(queried_matrix[:,column]))

   # The extracted data have to be normalized. So, the difference between real value and column mean value is assigned to its location.
    for row in range(len(queried_matrix)):
        for column in range(len(selected_column_names)):
            queried_matrix[row,column] -= column_means[column]

    return queried_matrix

# This function obtains the correlation matrix that shows the relation between columns.

# INPUTS
# normalized_value_matriz : The normalized data. (It has to be 2-D array)
# selected_column_names : The column names array that PCA calculation is implemented on.

# OUTPUT
# covariance_matrix : The calculated covariance matrix.
def getCorrelationMatrix(normalized_value_matrix, selected_column_names) :
    # The output is the 2-D array that it has the same number of columns and rows and their number equals to number of selected column names.
    # We have initialize this covariance_matrix. With np.zero, we fulfill the 0 values of this 2-D array.
    covariance_matrix = np.zeros((len(selected_column_names),len(selected_column_names)))
    print(covariance_matrix)

    # covariance calculated with the normalized values multiplication.
    # In here, this operation is implemented.
    for row in range(len(normalized_value_matrix)):
        for ind1 in range(len(selected_column_names)):
             for ind2 in range(len(selected_column_names)):
                 covariance_matrix[ind1,ind2] += normalized_value_matrix[row,ind1]*normalized_value_matrix[row,ind2]

    # After before loop, we have to divide result to the 1 minus value of length normalized_value_matrix
    for ind1 in range(len(selected_column_names)):
        for ind2 in range(len(selected_column_names)):
            covariance_matrix[ind1,ind2] /= len(normalized_value_matrix)-1

    return covariance_matrix

# This function return the formatted 2-D array.

# INPUT
# dataFrame : The spark.DataFrame object that keeps normal data
# selected_column_names : The column names array that PCA calculation is implemented on.

# OUTPUT
# exact_result : The arranged dataFrame datas.
def obtainY_values(dataFrame, selected_column_names) :
    # The data that is located on selected columns is extracted.
    queried_values = dataFrame.select(selected_column_names).rdd.flatMap(lambda x : x).collect()

    # This queried_values is an array. I have to obtaint an 2-D array structure. np.reshape helps for this aim.
    queried_matrix = np.reshape(np.array(queried_values), (-1, len(selected_column_names)))

    # The queried_matrix is arranged the wanted order
    exact_result = np.reshape(queried_matrix, (dataFrame.count(),len(selected_column_names)) , order=(0,3,2,1))

    return exact_result

# This function returns the indice values that anomaly data occured.

# INPUTS
# y_value_matrix : The 2-D data array that keeps the selected column data. (This data formatted as len(data) x len(selected_column_names)
# threshold_value : The calculated threshold limit value that represents the anomaly on the greater values than this.
# projection_matrix : The obtained 2-D array after PCA calculation.

# OUTPUT
# return_values : This 1-D array keeps the indices of anomaly data.
def detectAnomalyValueIndices(y_value_matrix, threshold_value, projection_matrix):
    # return_values will keep the indices values of anomalies.
    return_values = []

    for indis in range(len(y_value_matrix)):
        # With this matrix calculation, obtained spe_value represents the distance value from center to point location that is located on anomaly space.
        spe_value = calculateLengthOfVirtualPoint(np.dot(projection_matrix,y_value_matrix[indis].T))
        # If this value is greater than threshold value, this means that this value is an anomaly.
        if (spe_value > threshold_value):
            return_values.append(indis)

    return return_values

# This function calculates the distance value according the input coordinate values.

# INPUTS
# pointValues : This 1-D array represents the coordinate values of entered point.

# OUTPUT
# The output value represents the distance from the center to entered coordinate values.
def calculateLengthOfVirtualPoint(pointValues):
    value = 0
    for indis in range(len(pointValues)):
        value += pointValues[indis]**2
    return value**(1/2)

# This function find the indice value of entered eigen value in eigen value array.

# INPUT
# eigen_value_array : This 1-D array keeps the eigen values.
# eigen_value : This value is the indice value of eigen value that we want to know of its indice value.

# OUTPUT
# indis : The indice value of entered eigen value in eigen value array.
def findSequenceOfEigenValues(eigen_value_array,eigen_value):
    indis = 0
    # This loop find the indice value.
    while indis<len(eigen_value_array) and eigen_value_array[indis] != eigen_value :
        indis += 1
    return indis

# This function returns the selected data column indices in main and anomaly space

# INPUTS
# eigen_values : The calculated eigen values array.
# confidence_levek : The confidence level that cover at which percentage of data and it is read from user.

# OUTPUTS
# main_trend_indices : This 1-D array keeps that main space is consisted of which selected column indices in PCA calculation.
# anomaly_trend_indices : This 1-D array keeps that anomaly space is consisted of which selected column indices in PCA calculation.
def obtain_main_anomaly_trend(eigen_values, confidence_level):
    # indis: The value is for scanning the eigen_value array. (It points the end of array because of sorted eigen values
    #                           sorted from smaller value to bigger value)
    # eigen_sum_value : It keeps the sum of eigen values.
    # eigen_value_sequence : This keeps the indices values of sorted eigen value. (from smaller to bigger one)
    # main_trend_indices : This keeps the main space's consistence of PCA dimensions.
    # anomaly_trend_indices : This keeps the anomaly space's consistence of PCA dimensions.

    indis = len(eigen_values) - 1
    eigen_sum_value = 0.0
    eigen_total_sum_value = 0.0
    eigen_value_sequence = np.argsort(eigen_values)
    main_trend_indices = []
    anomaly_trend_indices = []

    # The sum of eigen values is calculated.
    for ind in range(len(eigen_values)):
        eigen_total_sum_value += eigen_values[ind]

    # While we obtain the confidence level, we add the biggest eigen value to eigen_sum_value.
    # Also the indice came from eigen_value_sequence is added to main_trend_indices.
    while indis >= 0 and eigen_sum_value < (confidence_level*eigen_total_sum_value):
        eigen_sum_value += eigen_values[eigen_value_sequence[indis]]
        main_trend_indices.append(eigen_value_sequence[indis])
        indis -= 1

    # Other values is added to anomaly_trend_indices.
    while indis >= 0 :
        anomaly_trend_indices.append(eigen_value_sequence[indis])
        indis -= 1

    return main_trend_indices,anomaly_trend_indices

# This function extracts the CSV output of PCA calculation.

# INPUTS
# dataFrame : The spark.DataFrame object that keeps normal data
# selected_column_names : The column names array that PCA calculation is implemented on.
# time_column_name : The column name that this column keeps the time value.
# anomalies : The indice values of anomaly points.
# projection_matrix :
# y_value_matrix : The array that is existed after PCA calculation.
# Threshold_value : The calculated threshold_value after PCA calculation

# OUTPUT
# Basically the csv file named general_anomaly.csv
def fileOutputPcaDimensions(dataFrame,selected_column_names,time_column_name,anomalies,projection_matrix,y_value_matrix,threshold_value):
    # anomaly_column : This array is for the anomaly or normal data representation with 0 and 1.
    # anomaly_indis : This is an indice value for scanning anomalies array.
    # index : This is an indice value for dataFrame.
    anomaly_column = []
    anomaly_indis = 0
    index = []
    # With this loop, we extract the anomaly_column and index arrays.
    for indis in range(dataFrame.count()):
        # If the indis value and the value of anomalies array is matched, the data that located on indis is an anomaly, otherwise normal data.
        print("indis : " + str(indis) + "\t anomaly_indis : " + str(anomaly_indis) + "\t len : " + str(len(anomalies)))
        print("anomalies data : " + str(anomalies[anomaly_indis]))
        if indis != anomalies[anomaly_indis] and anomaly_indis >= len(anomalies):
            anomaly_column.append('1')
            anomaly_indis += 1
        else:
            anomaly_column.append('0')
        index.append(str(indis))

    # columns is for extracting some data from dataFrame
    columns = list(selected_column_names)
    columns.append("container")

    # The data that is located on time column is extracted.
    time_values = dataFrame.select(time_column_name).rdd.flatMap(lambda x: x).collect()

    # In this loop, we transfrom the time value from string to time.
    for indis in range(len(time_values)):
        timestamp = datetime.datetime.fromtimestamp(time_values[indis]/1000)
        time_values[indis] = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    print(time_values)

    # Here, columns data is extracted.
    temporary = dataFrame.select(columns).rdd.flatMap(lambda x: x).collect()
    data = (np.reshape(np.array(temporary), (-1, len(selected_column_names) + 1)))
    print(data)
    print(len(data))

    # lists keeps the ouput csv content.
    lists = [[]]
    # In this loop, each row of lists added.
    for indis in range(dataFrame.count()):
        temp = []
        temp.append(index[indis])
        temp.append(time_values[indis])
        temp.append(calculateLengthOfVirtualPoint(np.dot(projection_matrix, y_value_matrix[indis].T)))
        temp.append(threshold_value)
        temp.append(anomaly_column[indis])
        for item in data[indis]:
            temp.append(item)
        lists.append(temp)

    # The first value of lists is dummy. So, it has to be removed.
    lists.remove(lists[0])

    # The header of csv is arranged.
    header_values = ["Index","Time","SPE","Threshold","outlier_status"]
    for item in selected_column_names:
        header_values.append(item)
    header_values.append("Container")

    # for output csv, the dataframe is created on pandas with lists and with to_csv method, we obtain the output csv.
    print_df = pd.DataFrame(lists)
    print_df.to_csv(path_or_buf="general_anomaly.csv", header=header_values, sep=',', index=False)


# This function returns the anomaly values to main.py

# INPUTS
# dataFrame : The spark.DataFrame object that keeps normal data
# selected_column_names : The column names array that PCA calculation is implemented on.
# confidence_level : Its represents the percentage that which data is not useful enough for PCA.
# time_column_name : The column name that this column keeps the time value.

# OUTPUT
# outlier_values : The indice values of anomaly potins in data.
def anomalyDetectionWithPCA(dataFrame, selected_column_names, confidence_level, time_column_name):

    # Normalized values are obtained.
    normalized_y_values = obtain_normalized_data(dataFrame,selected_column_names)


    print("selected colum names : ")
    print(selected_column_names)

    # Correlation matrix for PCA is obtained.
    correlation_matrix = getCorrelationMatrix(normalized_y_values, selected_column_names)
    print("\n\n")
    print("correlation matrix : ")
    print(correlation_matrix)

    correlation_matrix_with_numpy = np.cov(normalized_y_values.T)
    print(correlation_matrix_with_numpy)

    # The eigen values and eigen vectors is obtained with LA.eig funciton
    w, v = LA.eig(correlation_matrix_with_numpy)

    # The c_alpha represent the data inclusion rate.
    c_alpha = 1 - confidence_level
    print("\n\n")
    print("eigen value list : ")
    print(w)
    print("\n\n")
    print("eigen vector list : ")
    print(v)

    # Which input data column create main space and anomaly space is calculated.
    main_trend_indis,anomaly_trend_indis = obtain_main_anomaly_trend(w,c_alpha)
    print("\n\n")
    print("main trend indices : ")
    print(main_trend_indis)
    print("\n\n")
    print("anomaly trend indis : ")
    print(anomaly_trend_indis)

    # <-------------------------- THE Q-STATISTIC-BASED THRESHOLD VALUE CALCULATION -------------------------------->

    # eig_value_sum : The summary of eigen values.
    # eig_value_sum2 : The summary of square of eigen values.
    # eig_value_sum3 : The summary of cube eigen values.
    eig_value_sum = 0
    eig_value_sum2 = 0
    eig_value_sum3 = 0

    print("\n\n")
    print("len of anomaly_trend_indis : " + str(len(anomaly_trend_indis)))
    # The summary types of eigen values are calculated.
    for indis in range(len(anomaly_trend_indis)):
        eig_value_sum += w[anomaly_trend_indis[indis]]
        eig_value_sum2 += w[anomaly_trend_indis[indis]]**2
        eig_value_sum3 += w[anomaly_trend_indis[indis]]**3

    print("\n\n")
    print("eig_value_sum : " + str(eig_value_sum))
    print("eig_value_sum2 : " + str(eig_value_sum2))
    print("eig_value_sum3 : " + str(eig_value_sum3))

    # The h0 value in calculation formula is calculated.
    h0 = 1 - ((2*eig_value_sum*eig_value_sum3) / (3*(eig_value_sum2**2)))
    print("h0 : " + str(h0))

    # The threshold value is calculated according to q-statistic method.
    threshold_value = (eig_value_sum*(((c_alpha*(2*eig_value_sum2*h0**2)**(1/2))/eig_value_sum) + 1 + ((eig_value_sum2*h0*(h0-1))/(eig_value_sum**2)))**(1/h0))**(1/2)
    print("threshold value : " + str(threshold_value))

    # <-------------------------- THE Q-STATISTIC-BASED THRESHOLD VALUE CALCULATION -------------------------------->


    print("\n\n")
    print("eigen vectors : ")
    print(v)
    print("\n\n")
    print("eigen vectors for trend line : ")
    print(v[:,main_trend_indis])

    # The main space values is calculated. (The main space coordinated of values)
    transpoze_multiply = np.dot(v[:,main_trend_indis],v[:,main_trend_indis].T)
    print("\n\n")
    print("calculated transpoze matrix : ")
    print(transpoze_multiply)

    # The anomaly space values is calculated. (The anomaly space coordinate of values)
    projection_matrix = np.subtract(np.eye(len(selected_column_names)),transpoze_multiply)
    print("\n\n")
    print("projection matrix : ")
    print(projection_matrix)

    #suspected_values = [[]]
    y_values = obtainY_values(dataFrame,selected_column_names)

    print("\n\n")
    print("y_values length : " + str(len(y_values)))
    print("dataFrame count : " + str(dataFrame.count()))

    # The indice values of anomaly values are found.
    outlier_values = detectAnomalyValueIndices(normalized_y_values, threshold_value, projection_matrix)
    #print(outlier_values)
    print("\n\n")
    print("outlier number : "+ str(len(outlier_values)))
    print("\n\n")
    print("outlier values : ")
    print(outlier_values)

    # The CSV output is triggered.
    fileOutputPcaDimensions(dataFrame, selected_column_names, time_column_name, outlier_values, projection_matrix, normalized_y_values, threshold_value)

    return outlier_values