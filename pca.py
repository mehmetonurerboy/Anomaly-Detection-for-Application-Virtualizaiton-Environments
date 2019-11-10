from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from numpy import linalg as LA
import numpy as np
from statistics import mean as getMean

def obtain_normalized_data(dataFrame, selected_column_names):
    queried_values = dataFrame.select(selected_column_names).rdd.flatMap(lambda x: x).collect()

    # print(queried_values)
    queried_matrix = np.reshape(np.array(queried_values), (-1, len(selected_column_names)))
    #print(queried_matrix)
    print(queried_matrix[:,0])
    print(queried_matrix[0,:])

    column_means = []

    for column in range(len(selected_column_names)):
        column_means.append(np.mean(queried_matrix[:,column]))

    #print(column_means)

    #print(len(queried_matrix))
    #print(queried_matrix[0,0])
    for row in range(len(queried_matrix)):
        for column in range(len(selected_column_names)):
            queried_matrix[row,column] -= column_means[column]

    #print(queried_matrix)
    return queried_matrix

def getCorrelationMatrix(normalized_value_matrix, selected_column_names) :
    # old input : dataFrame, inputCols
    """
    # #Firstly we have to convert all columns to vector
    vector_col = "correlation_values"
    assembler = VectorAssembler(inputCols=inputCols, outputCol=vector_col)
    df_vector = assembler.transform(dataFrame).select(vector_col)
    df_vector.show()

    # We have vector values of all column. So, we can obtain correlation matrix
    matrix = Correlation.corr(df_vector, vector_col)

    values = matrix.collect()[0]["pearson({})".format(vector_col)].values
    matrix = np.reshape(np.array(values), (-1, 4))
    return matrix
    """

    covariance_matrix = np.zeros((len(selected_column_names),len(selected_column_names)))
    print(covariance_matrix)

    for row in range(len(normalized_value_matrix)):
        for ind1 in range(len(selected_column_names)):
             for ind2 in range(len(selected_column_names)):
                 covariance_matrix[ind1,ind2] += normalized_value_matrix[row,ind1]*normalized_value_matrix[row,ind2]

    for ind1 in range(len(selected_column_names)):
        for ind2 in range(len(selected_column_names)):
            covariance_matrix[ind1,ind2] /= len(normalized_value_matrix)-1

    print(covariance_matrix)


def obtainY_values(dataFrame, selected_column_names) :
    queried_values = dataFrame.select(selected_column_names).rdd.flatMap(lambda x : x).collect()

    #print(queried_values)
    queried_matrix = np.reshape(np.array(queried_values), (-1, len(selected_column_names)))
    #print(queried_matrix)

    #print(queried_matrix)

    exact_result = np.reshape(queried_matrix, (dataFrame.count(),len(selected_column_names)) , order=(0,3,2,1))

    #print(exact_result)

    return exact_result

def detectAnomalyValueIndices(dataFrame, y_value_matrix, threshold_value, projection_matrix):
    return_values = []

    for indis in range(len(y_value_matrix)):
        #print("y value matrix value : ")
        #print(y_value_matrix[indis])
        #print("length value : ")
        #print(np.dot(projection_matrix,y_value_matrix[indis]))
        spe_value = calculateLengthOfVirtualPoint(np.dot(projection_matrix,y_value_matrix[indis].T))
        #print('[' + str(indis) + '] : ' + str(spe_value) + '\t threshold : ' + str(threshold_value))
        if (spe_value > threshold_value):
            return_values.append(indis)


    return return_values

def calculateLengthOfVirtualPoint(pointValues):
    value = 0
    print(pointValues)
    for indis in range(len(pointValues)):
        value += pointValues[indis]**2
    return value**(1/2)

# Find the indice value of entered eigen value
def findSequenceOfEigenValues(eigen_value_array,eigen_value):
    indis = 0
    while indis<len(eigen_value_array) and eigen_value_array[indis] != eigen_value :
        indis += 1
    return indis


def indicesOfSortedList(list, pairs):
    indiceValues = []
    for indis in range(len(pairs)):
        temp_indis = 0
        while temp_indis < len(pairs) and list[indis] != pairs[indis][0]:
            temp_indis += 1
        if temp_indis != len(pairs):
            indiceValues.insert(indis,temp_indis)
    return indiceValues

def obtain_main_trend(eigen_values, confidence_level):
    indis = len(eigen_values) - 1
    eigen_sum_value = 0.0
    eigen_total_sum_value = 0.0
    eigen_value_sequence = np.argsort(eigen_values)
    main_trend_indices = []
    anomaly_trend_indices = []

    # The sum of eigen values is calculated.
    for ind in range(len(eigen_values)):
        eigen_total_sum_value += eigen_values[ind]

    # While we obtain the confidence level, we add to
    while indis >= 0 and eigen_sum_value < (confidence_level*eigen_total_sum_value):
        #print(indis)
        #print(eigen_values[eigen_value_sequence[indis]])
        eigen_sum_value += eigen_values[eigen_value_sequence[indis]]
        main_trend_indices.append(eigen_value_sequence[indis])
        indis -= 1

    while indis >= 0 :
        anomaly_trend_indices.append(eigen_value_sequence[indis])
        indis -= 1

    return main_trend_indices,anomaly_trend_indices

# return indice values of anomaly values.
def anomalyDetectionWithPCA(dataFrame, selected_column_names, confidence_level):

    normalized_y_values = obtain_normalized_data(dataFrame,selected_column_names)


    print("selected colum names : ")
    print(selected_column_names)
    correlation_matrix = getCorrelationMatrix(normalized_y_values, selected_column_names)
    print("\n\n")
    print("correlation matrix : ")
    print(correlation_matrix)
    #print(len(correlation_matrix))

    correlation_matrix_with_numpy = np.cov(normalized_y_values.T)
    print(correlation_matrix_with_numpy)


    w, v = LA.eig(correlation_matrix_with_numpy)
    c_alpha = 1 - confidence_level
    print("\n\n")
    print("eigen value list : ")
    print(w)
    print("\n\n")
    print("eigen vector list : ")
    print(v)

    main_trend_indis,anomaly_trend_indis = obtain_main_trend(w,c_alpha)
    print("\n\n")
    print("main trend indices : ")
    print(main_trend_indis)
    print("\n\n")
    print("anomaly trend indis : ")
    print(anomaly_trend_indis)


    #indice_value_pair_matrix = [[]]
    #for indis in range(len(w)):
    #    indice_value_pair_matrix.append([w[indis], indis])
    #indice_value_pair_matrix.remove(indice_value_pair_matrix[0])

    #print(indice_value_pair_matrix)


    #print(eigen_value_matrix)

    #sorted_indices = indicesOfSortedList(worked_eigen_values,eigen_value_matrix)
    #print(sorted_indices)


    eig_value_sum = 0
    eig_value_sum2 = 0
    eig_value_sum3 = 0

    print("\n\n")
    print("len of anomaly_trend_indis : " + str(len(anomaly_trend_indis)))
    for indis in range(len(anomaly_trend_indis)):
        eig_value_sum += w[anomaly_trend_indis[indis]]
        eig_value_sum2 += w[anomaly_trend_indis[indis]]**2
        eig_value_sum3 += w[anomaly_trend_indis[indis]]**3

    print("\n\n")
    print("eig_value_sum : " + str(eig_value_sum))
    print("eig_value_sum2 : " + str(eig_value_sum2))
    print("eig_value_sum3 : " + str(eig_value_sum3))

    h0 = 1 - ((2*eig_value_sum*eig_value_sum3) / (3*(eig_value_sum2**2)))
    print("h0 : " + str(h0))

    #c_alpha = 0.9

    threshold_value = (eig_value_sum*(((c_alpha*(2*eig_value_sum2*h0**2)**(1/2))/eig_value_sum) + 1 + ((eig_value_sum2*h0*(h0-1))/(eig_value_sum**2)))**(1/h0))**(1/2)
    print("threshold value : " + str(threshold_value))



    
    #print(w)
    #print(np.eye(4))
    #print('\n\n')
    #print(v)
    #print('\n\n')
    #print(v.T)
    #print('\n\n')

    print("\n\n")
    print("eigen vectors : ")
    print(v)
    print("\n\n")
    print("eigen vectors for trend line : ")
    print(v[:,main_trend_indis])
    transpoze_multiply = np.dot(v[:,main_trend_indis],v[:,main_trend_indis].T)
    print("\n\n")
    print("calculated transpoze matrix : ")
    print(transpoze_multiply)
    print(np.subtract(np.eye(len(v[0])),np.dot(v[:,main_trend_indis],v[:,main_trend_indis].T)))
    projection_matrix = np.subtract(np.eye(len(selected_column_names)),transpoze_multiply)
    #projection_matrix = 1 - np.dot(v[:main_trend_indis],v[:main_trend_indis].T)
    print("\n\n")
    print("projection matrix : ")
    print(projection_matrix)

    #suspected_values = [[]]
    y_values = obtainY_values(dataFrame,selected_column_names)

    print("\n\n")
    print("y_values length : " + str(len(y_values)))
    print("dataFrame count : " + str(dataFrame.count()))

    outlier_values = detectAnomalyValueIndices(dataFrame, normalized_y_values, threshold_value, projection_matrix)
    #print(outlier_values)
    print("\n\n")
    print("outlier number : "+ str(len(outlier_values)))
    print("\n\n")
    print("outlier values : ")
    #print(outlier_values)

    return outlier_values



