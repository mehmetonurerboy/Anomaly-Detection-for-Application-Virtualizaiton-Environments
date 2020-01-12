from pyspark.sql import functions as func
import pandas as pd


# This function returns the mean values and standard deviation values of entered column.
def getStatisticsOfDataFrame(dataFrame, selectedColumns):
    meanValues = []
    stdValues = []

    for column in selectedColumns:
        df_stats = dataFrame.select(func.mean(func.col(column)).alias('mean'),
                                    func.stddev(func.col(column)).alias('std'))

        mean = df_stats.select('mean').toPandas().values.tolist()
        std = df_stats.select('std').toPandas().values.tolist()

        meanValues.append(mean[0][0])
        stdValues.append(std[0][0])

    return meanValues,stdValues
