
# PURPOSE 1: Predict the rating

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from datetime import datetime
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import confusion_matrix
from pyspark.ml.feature import Normalizer



# CREATE SPARK CONTEXT
now = datetime.now()
unique_key = now.strftime("%m-%d-%Y-%H-%M-%S")

spark = SparkSession.builder.appName('google-play-store').getOrCreate()
print("STARTING SPARK SESSION. UNIQUE KEY: " + unique_key)

## STEP 1 - READ DATA FROM CSV FILE
print("READING FROM CSV FILE")

# MAIN DATA FILE
df = spark.read.csv('Data/googleplaystore.csv',inferSchema=True,header=True)
df.printSchema()
df.cache()

# REVIEWS FILE
df2 = spark.read.csv('Data/googleplaystore_user_reviews.csv',inferSchema=True,header=True)
df2.printSchema()

df2 = df2.filter(df2["Sentiment"] != "nan")
df2 = df2.withColumn("Potential Score", (0.5 + 0.5 * df2["Sentiment_Polarity"] * (1.5 - df2["Sentiment_Subjectivity"])))
df2 = df2.groupBy("App").agg({"Potential Score" : "avg"}).withColumnRenamed("avg(Potential Score)", "avg score")
df = df.join(df2, df.App == df2.App).drop("App")


# CLEANING DATA
print("Start cleaning data")


# SIZE feature
# Size data comes as: xM or 'Varies with device'
# If in xM format then it should be parsed to x, default unit of measure should be M
# If in Varies with device format then it should be parsed to the avg of their Category

print("Cleaning Size feature")

df_temp = df
df_temp = df_temp.withColumn("Size uom", F.expr("substring(Size, length(Size), 1)"))

df_temp = df_temp.withColumn("Size parsed", F.when(df_temp["Size uom"] == "k", \
    F.expr("substring(Size, 1, length(Size) - 1)").cast('double') / 1024))

df_temp = df_temp.withColumn("Size parsed", F.when(df_temp["Size uom"] == "M", \
    F.expr("substring(Size, 1, length(Size) - 1)").cast('double')).otherwise(df_temp['Size parsed']))

df_temp_avg_per_cat = df_temp.filter(df_temp["Size uom"] != "e").groupBy("Category").agg({"Size parsed" : "avg"}) \
                             .withColumnRenamed("avg(Size parsed)", "avg_size") \
                             .withColumnRenamed("Category", "Category for avg")

df_temp = df_temp.join(df_temp_avg_per_cat, df_temp_avg_per_cat["Category for avg"] == df_temp["Category"])

df_temp = df_temp.withColumn("Size parsed", F.when(df_temp["Size uom"] == "e", \
    df_temp["avg_size"].cast('double')).otherwise(df_temp['Size parsed']))

df_temp = df_temp.withColumn("Size parsed", F.round(df_temp["Size parsed"], 2))

df = df_temp.drop("Size", "Size uom", "Category for avg", "avg_size")


# Price feature might have $ sign at the beggining for not null values
df = df.withColumn("Price", F.when(df["Price"] != "0", F.expr("substring(Price, 2, length(Price) - 1)").cast('double')) \
        .otherwise(df_temp['Price']).cast('double'))


# Rating feature - fill NaN ratings from Rating with 2.5 avg val
df = df.withColumn("Rating", F.when(df["Rating"] == "NaN", F.lit("2.5")).otherwise(df['Rating']))



# Installs cols which we want to classify by has many close values
# So I will convert values over 10 to 10+ and also add an upper roof of 100,000,000+
df = df.withColumn("Installs", F.when(df["Installs"] == "0", F.lit("10+")).otherwise(df["Installs"]))
df = df.withColumn("Installs", F.when(F.expr('substring(Installs, 1, length(Installs) - 1)').cast('double') < 100, F.lit("10+")).otherwise(df["Installs"]))
df = df.withColumn("Installs", F.when(F.expr('substring(Installs, 1, length(Installs) - 1)').cast('double') > 100000000, F.lit("100000000+")).otherwise(df["Installs"]))
df = df.withColumn("Installs", F.expr('substring(Installs, 1, length(Installs) - 1)'))


# CAST used numeric features to double values
print("Casting numeric features to double values")

features_to_cast_to_double = [ \
    'Rating', \
    'Reviews', \
    'Price', \
    'avg score'
]
for feature in features_to_cast_to_double:
    print("Casting " + feature + " to double")
    df = df.withColumn(feature, df[feature].cast('double'))

# INDEXING categorical features
print("Indexing categorical features")

features_to_index = [
    'Category', \
    'Type', \
    'Content Rating', \
    'Genres', \
    'Android Ver', \
    'Installs'
]

df_indexed = df
df_indexed.cache()
for feature in features_to_index:
    print("Indexing: " + feature)
    indexer =  StringIndexer(inputCol=feature, outputCol=(feature + " indexed"))
    df_indexed = indexer.fit(df_indexed).transform(df_indexed)
    df_indexed = df_indexed.drop(feature)

# DROP unused columns
print("Dropping unused columns")

featres_to_drop = [
    'App', \
    'Last Updated', \
    'Current Ver',
    'Android Ver indexed'
]

for feature in featres_to_drop:
    print("Dropping: " + feature)
    df_indexed = df_indexed.drop(feature)

# Splitting data into train test data and streaming data
train_test_data, streaming_data = df_indexed.randomSplit([0.95, 0.05])

# SAVING STREAMING DATA
streaming_data.write.save("OutputGStore\\" + unique_key + "-streaming-data.csv", format="csv", header="true")
del streaming_data

## STEP 2: Prepare, train and validate the data
print("STEP 2: Train and validate the model")

feature_cols = train_test_data.columns
feature_cols.remove('Installs indexed')

assembler = VectorAssembler(inputCols = feature_cols, outputCol = "features", handleInvalid = "error")
pipeline = Pipeline(stages=[assembler])
outputModel = pipeline.fit(train_test_data)
output = outputModel.transform(train_test_data)
final_data = output.select("features", "Installs indexed")

train_data, test_data = final_data.randomSplit([0.7, 0.3])


# CLASIFICATION CODE
# Random forest classifier
rf = RandomForestClassifier(labelCol="Installs indexed", featuresCol="features", numTrees=32, maxBins=120)
model = rf.fit(train_data)

predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="Installs indexed", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Random forest test Error = %g" % (1.0 - accuracy))
randomForestError = (1.0 - accuracy)

# Save just the random forest model
print("Saving the model")
strErr = str(round(randomForestError, 2))
model.save("OutputGStore\\" + str(unique_key) + "-" + strErr + "-randomForestModel")

# MultilayerPerceptronClassifier
layers = [len(feature_cols), 10]
trainer = MultilayerPerceptronClassifier(layers=layers, labelCol="Installs indexed", featuresCol="features", blockSize=500, seed=10003, maxIter=1000)
model = trainer.fit(train_data)

result = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(
    labelCol="Installs indexed", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(result)
print("Perceptron test Error = %g" % (1.0 - accuracy))
deepLearningError = (1.0 - accuracy)


temp = result.select("Installs indexed", "prediction")
actual = [int(row['Installs indexed']) for row in temp.collect()]
predicted = [int(row['prediction']) for row in temp.collect()]
conf = confusion_matrix(actual, predicted)


# Let's try to predict the Rating based on comments

feature_cols = ['avg score', 'Category indexed', 'Genres indexed', 'Installs indexed', \
     'Size parsed', 'Price', 'Type indexed', 'Reviews']

assembler = VectorAssembler(inputCols = feature_cols, outputCol = "features", handleInvalid = "error")
pipeline = Pipeline(stages=[assembler])
outputModel = pipeline.fit(train_test_data)
output = outputModel.transform(train_test_data)
final_data = output.select("features", "Rating")

train_data, test_data = final_data.randomSplit([0.7, 0.3])

lr = LinearRegression(featuresCol = "features", labelCol='Rating', regParam=0.1)
lrModel = lr.fit(train_data)
test_results = lrModel.evaluate(test_data)

print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))
linearRegressionR2 = test_results.r2

print("Script results: ")
print("Random forest classifier error", randomForestError)
print("Neural networks classifier error", deepLearningError)
print("Linear regression R2", linearRegressionR2)