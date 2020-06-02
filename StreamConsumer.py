import sys
import os
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
import json

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml import Pipeline

CURRENT_KEY = "06-01-2020-21-12-29-0.4"
MODEL_PATH = "OutputGStore/" + CURRENT_KEY + "-randomForestModel"


def main():
    # spark = SparkSession.builder.appName('google-play-store-streamer').getOrCreate()
    sc = SparkContext(appName="PysparkStreaming").getOrCreate()
    ssc = StreamingContext(sc, 3)  

    # Load Model
    model = RandomForestClassificationModel.load(MODEL_PATH)


    def parseStream(rdd):
        if not rdd.isEmpty():
            df = sc.read.json(rdd)
            df.show()
            # Vectorize data
            feature_cols = df.columns
            feature_cols.remove('Installs indexed')
            assembler = VectorAssembler(inputCols = feature_cols, outputCol = "features", handleInvalid = "error")
            pipeline = Pipeline(stages=[assembler])
            outputModel = pipeline.fit(df)
            output = outputModel.transform(df)
            final_data = output.select("features", "Installs indexed")
            # Predict
            predictions = model.transform(final_data)
            evaluator = MulticlassClassificationEvaluator(
                labelCol="Installs indexed", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            print("Random forest test Error = %g" % (1.0 - accuracy))
            randomForestError = (1.0 - accuracy)
            print(randomForestError)

    stream_data = ssc.textFileStream('StreamData/')
    stream_data.foreachRDD( lambda rdd: parseStream(rdd) )

    ssc.start()
    ssc.awaitTermination()


if __name__ == "__main__":
    main()