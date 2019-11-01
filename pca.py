from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from numpy import linalg as LA
import numpy as np




def getCorrelationMatrix(dataFrame, inputCols) :

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

def obtainY_values(dataFrame, selected_column_names) :
    queried_values = dataFrame.select(selected_column_names).rdd.flatMap(lambda x : x).collect()

    #print(queried_values)
    queried_matrix = np.reshape(np.array(queried_values), (-1, 4))
    #print(queried_matrix)

    print(queried_matrix)

    exact_result = np.reshape(queried_matrix, (dataFrame.count(),len(selected_column_names)) , order=(0,3,2,1))

    print(exact_result)

    return exact_result

def detectAnomalyValueIndices(dataFrame, y_value_matrix, threshold_value, projection_matrix):
    return_values = []

    for indis in range(len(y_value_matrix)):
        print("y value matrix value : ")
        print(y_value_matrix[indis])
        print("length value : ")
        print(np.dot(projection_matrix,y_value_matrix[indis]))
        spe_value = calculateLengthOfVirtualPoint(np.dot(projection_matrix,y_value_matrix[indis].T))
        print('[' + str(indis) + '] : ' + str(spe_value) + '\t threshold : ' + str(threshold_value))
        if (spe_value > threshold_value):
            return_values.append(indis)


    return return_values

def calculateLengthOfVirtualPoint(pointValues):
    value = 0
    for indis in range(len(pointValues)):
        value += pointValues[indis]**2
    return value

# Find the indice value of entered eigen value
def findSequenceOfEigenValues(eigen_value_array,eigen_value):
    indis = 0
    while indis<len(eigen_value_array) and eigen_value_array[indis] != eigen_value :
        indis += 1
    return indis

# This function creates a 2D array that contains <Eigen value,indice> pairs
def createEigenValueIndiceMatrix(eigen_value_array):
    return_matrix = [[]]
    for indis in range(len(eigen_value_array)):
        return_matrix.append([eigen_value_array[indis],indis])
    return_matrix.remove(return_matrix[0])
    return return_matrix

def indicesOfSortedList(list, pairs):
    indiceValues = []
    for indis in range(len(pairs)):
        temp_indis = 0
        while temp_indis < len(pairs) and list[indis] != pairs[indis][0]:
            temp_indis += 1
        if temp_indis != len(pairs):
            indiceValues.insert(indis,temp_indis)
    return indiceValues

def anomalyDetectionWithPCA(dataFrame, selected_column_names, confidence_level):
    print(selected_column_names)
    correlation_matrix = getCorrelationMatrix(dataFrame, selected_column_names)
    print(correlation_matrix)
    #print(len(correlation_matrix))

    w, v = LA.eig(correlation_matrix)
    c_alpha = 1 - confidence_level
    np_version_w = np.array(w)

    print(w)
    #print(type(w))
    print(v)

    eigen_value_matrix = createEigenValueIndiceMatrix(np_version_w)
    worked_eigen_values = np.sort(w)

    for indis in range(int(len(worked_eigen_values)/2)):
        temp = worked_eigen_values[indis]
        worked_eigen_values[indis] = worked_eigen_values[len(worked_eigen_values)-1-indis]
        worked_eigen_values[len(worked_eigen_values) - 1 - indis] = temp

    #print(worked_eigen_values)

    indice_value_pair_matrix = [[]]
    for indis in range(len(w)):
        indice_value_pair_matrix.append([w[indis], indis])
    indice_value_pair_matrix.remove(indice_value_pair_matrix[0])

    #print(indice_value_pair_matrix)


    #print(eigen_value_matrix)

    sorted_indices = indicesOfSortedList(worked_eigen_values,eigen_value_matrix)
    #print(sorted_indices)


    eig_value_sum = w[1]
    eig_value_sum2 = w[1]**2
    eig_value_sum3 = w[1]**3


    #for indis in range(len(selected_column_names)):
     #   eig_value_sum += w[indis]
      #  eig_value_sum2 += w[indis]**2
       # eig_value_sum3 += w[indis]**3
    
    #print(eig_value_sum)
    #print(eig_value_sum2)
    #print(eig_value_sum3)

    h0 = 1 - ((2*eig_value_sum*eig_value_sum3) / (3*(eig_value_sum2**2)))
    #print(h0)

    #c_alpha = 0.9

    threshold_value = (eig_value_sum*(((c_alpha*(2*eig_value_sum2*h0**2)**(1/2))/eig_value_sum) + 1 + ((eig_value_sum2*h0*(h0-1))/(eig_value_sum**2)))**(1/h0))**(1/2)
    #print(threshold_value)

    """
    print(w)
    print(np.eye(4))
    print('\n\n')
    print(v)
    print('\n\n')
    print(v.T)
    print('\n\n')
    """
    transpoze_multiply = np.dot(v[:,[0, 3, 2]],v[:,[0, 3, 2]].T)
    #print(transpoze_multiply)
    #print(np.subtract(np.eye(len(v[0])),np.dot(v[:,[0, 3, 2]],v[:,[0, 3, 2]].T)))
    projection_matrix = np.subtract(np.eye(4),transpoze_multiply)
    #projection_matrix = 1 - np.dot(v[:[0,3,2]],v[:[0,3,2]].T)
    #print(projection_matrix)

    #suspected_values = [[]]
    y_values = obtainY_values(dataFrame,selected_column_names)

    print(len(y_values))
    print(dataFrame.count())

    outlier_values = detectAnomalyValueIndices(dataFrame, y_values, threshold_value, projection_matrix)
    print(outlier_values)
    print(len(outlier_values))


