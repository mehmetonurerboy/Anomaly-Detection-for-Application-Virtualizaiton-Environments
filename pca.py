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
def anomalyDetectionWithPCA(dataFrame, selected_column_names):
    print(selected_column_names)
    correlation_matrix = getCorrelationMatrix(dataFrame, selected_column_names)
    print(correlation_matrix)
    print(len(correlation_matrix))

    w, v = LA.eig(correlation_matrix)

    print(w)
    print(v)