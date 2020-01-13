from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

def calculateLogisticRegression(train_dataframe,test_dataframe):

    #selectedCols = ['label', 'features'] + pyspark_train_df.columns


    #print(train_dataframe.columns)

    assembler = VectorAssembler(inputCols=test_dataframe.columns, outputCol="features")
    pyspark_train_df = assembler.transform(train_dataframe)
    pyspark_test_df = assembler.transform(test_dataframe)

    #print(pyspark_train_df.columns)
    #print(pyspark_test_df.columns)

    #train, test = df.randomSplit([0.7, 0.3], seed = 2018)
    #print("Training Dataset Count: " + str(train.count()))
    #print("Test Dataset Count: " + str(test.count()))
    #print(type(train))


    #lr = LogisticRegression(featuresCol= "features", maxIter=10)
    lr = LogisticRegression(featuresCol= 'features', labelCol='outlier_status', maxIter=10)
    lrModel = lr.fit(pyspark_train_df)

    """
    print("success")

    import numpy as np
    beta = np.sort(lrModel.coefficients)

    print("beta")
    print(beta)
    print("\n\n\n")

    trainingSummary = lrModel.summary
    roc = trainingSummary.roc.toPandas()

    #print("roc")
    #print(roc)
    print("\n\n\n")

    pr = trainingSummary.pr.toPandas()

    #print("pr")
    #print(pr)
    print("\n\n\n")
    """

    predictions = lrModel.transform(pyspark_test_df)
    #predictions.show()

    #print(type(predictions))
    #cross_validation_test(lr,pyspark_train_df,pyspark_test_df)

    print(predictions.count())
    return predictions.toPandas()

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def cross_validation_test(lr,train_dataframe,test_dataframe):
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol ="Outcome")

    paramGrid = ParamGridBuilder()\
        .addGrid(lr.aggregationDepth, [2,5,10])\
        .addGrid(lr.elasticNetParam, [0.0, .05, 1.0])\
        .addGrid(lr.fitIntercept, [False,True])\
        .addGrid(lr.maxIter, [10, 100, 1000])\
        .addGrid(lr.regParam, [0.01, 0.5, 2.0])\
        .build()

    # Create 10-fold CrossValidator
    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)

    # Run cross validator
    cvModel = cv.fit(train_dataframe)
    print("cvModel")
    print(cvModel)

    # Make predictions on test documents. cvModel uses the best model found (lrModel).
    prediction = cvModel.transform(test_dataframe)
    print("\n\npredictions")
    print(prediction)
    columns = []
    columns.extend(test_dataframe.columns)
    columns.append("probability")
    columns.append("predictions")
    selected = prediction.select(test_dataframe.columns,)
    for row in selected.collect():
        print(row)

def logisticRegressionTest(pandas_df,test_df):
    #print(pandas_df)
    #print(pandas_df.columns)

    confidence_statistics = [0,0,0,0]
    for indis in range(len(pandas_df)):
        if test_df.iloc[indis]["outlier_status"] == 0 and pandas_df.iloc[indis]["prediction"] == 0.0:
            confidence_statistics[0] += 1
        elif test_df.iloc[indis]["outlier_status"] == 0 and pandas_df.iloc[indis]["prediction"] == 1.0:
            confidence_statistics[1] += 1
        elif test_df.iloc[indis]["outlier_status"] == 1 and pandas_df.iloc[indis]["prediction"] == 0.0:
            confidence_statistics[2] += 1
        elif test_df.iloc[indis]["outlier_status"] == 1 and pandas_df.iloc[indis]["prediction"] == 1.0:
            confidence_statistics[3] += 1

    return confidence_statistics

