from pyspark.sql import SparkSession
import pyspark


spark = SparkSession.builder.appName("PCA").getOrCreate()

dataFilePath = "E:\\PERSONAL ITEMS\\LESSON ITEMS\\SEVENTH TERM\\BİTİRME\\Data\\AnomalyDetection\\dataset"

#data = spark.read.option("header","true").option("inferSchema","true").format("csv")
df = spark.read.option("header","true").option("inferSchema","true").csv(dataFilePath + "\\1.csv")

from pyspark.ml.feature import PCA,StandardScaler,VectorAssembler



column_names = ["ram_usage","container","cpu_percent","ram_limit","io_usage","io_limit","network_limit","node","time","network_usage","pids"]

assembler = VectorAssembler(
    #inputCols=["ram_usage","cpu_percent","io_usage","network_usage"],
    inputCols=["cpu_percent","ram_usage"],
    outputCol="features"
)

output = assembler.transform(df)

scaler = StandardScaler(inputCol="features",
                        outputCol="scaledFeatures",
                        withStd=True,
                        withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(output)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(output)
scaledData.show()

pca = PCA(k=2, inputCol="features", outputCol="pca_features").fit(scaledData)

pcaDf = pca.transform(scaledData)
results = pcaDf.select("pca_features")
results.show()

split_col = pyspark.sql.functions.split(results['pca_features'].values.str, ',')
results = results.withColumn('pca_feature_1', split_col.getItem(0))
results = results.withColumn('pca_feature_2', split_col.getItem(1))
results.show()

#results.toPandas().to_excel('1-1-cpu-ram-2.xlsx')

