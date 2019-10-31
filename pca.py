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

    print(queried_values)
    queried_matrix = np.reshape(np.array(queried_values), (-1, 4))
    print(queried_matrix)

    return queried_matrix

def detectAnomalyValueIndices(dataFrame, y_value_matrix, threshold_value, projection_matrix):
    return_values = []

    for indis in range(len(y_value_matrix)):
        spe_value = calculateLengthOfVirtualPoint(np.dot(projection_matrix,y_value_matrix[indis].T))
        print('[' + str(indis) + '] : ' + str(spe_value) + '\t threshold : ' + str(threshold_value))
        if (spe_value > threshold_value):
            return_values.append(y_value_matrix[indis])

    return return_values

def calculateLengthOfVirtualPoint(pointValues):
    value = 0
    for indis in range(len(pointValues)):
        value += pointValues[indis]**2
    return value

def findSequenceOfEigenValues(eigen_value_array):
    indices = []
    for indis in range(len(eigen_value_array)):
        maxValue = eigen_value_array[indis]
        maxIndis = indis
        for indis2 in range(len(eigen_value_array)-1):
            if eigen_value_array[indis2] > eigen_value_array[indis]:
                maxIndis = indis2
                maxValue = eigen_value_array[indis2]
        indices.append(maxIndis)

def anomalyDetectionWithPCA(dataFrame, selected_column_names):
    print(selected_column_names)
    correlation_matrix = getCorrelationMatrix(dataFrame, selected_column_names)
    print(correlation_matrix)
    print(len(correlation_matrix))

    w, v = LA.eig(correlation_matrix)

    print(w)
    print(v)

    eig_value_sum = 0
    eig_value_sum2 = 0
    eig_value_sum3 = 0

    for indis in range(len(selected_column_names)):
        eig_value_sum += w[indis]
        eig_value_sum2 += w[indis]**2
        eig_value_sum3 += w[indis]**3

    print(eig_value_sum)
    print(eig_value_sum2)
    print(eig_value_sum3)

    h0 = 1 - ((2*eig_value_sum*eig_value_sum3) / (3*(eig_value_sum2**2)))
    print(h0)

    c_alpha = 0.9

    threshold_value = (eig_value_sum*(((c_alpha*(2*eig_value_sum2*h0**2)**(1/2))/eig_value_sum) + 1 + ((eig_value_sum2*h0*(h0-1))/(eig_value_sum**2)))**(1/h0))**(1/2)
    print(threshold_value)

    print(w)
    print(np.eye(4))
    print('\n\n')
    print(v)
    print('\n\n')
    print(v.T)
    print('\n\n')
    print(np.dot(v,v.T))
    projection_matrix = np.subtract(np.eye(4),(np.dot(v,v.T)))
    print(projection_matrix)

    #suspected_values = [[]]

    y_values = obtainY_values(dataFrame,selected_column_names)

    print(len(y_values))
    print(dataFrame.count())

    outlier_values = detectAnomalyValueIndices(dataFrame, y_values, threshold_value, projection_matrix)
    print(outlier_values)
    print(len(outlier_values))

