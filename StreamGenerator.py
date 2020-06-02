from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import json
import time

CURRENT_KEY = "06-01-2020-21-12-29"
STREAMING_DATA_SOURCE = CURRENT_KEY + "-streaming-data.csv"

spark = SparkSession.builder.appName('google-play-store-streamer').getOrCreate()

df = spark.read.load("OutputGStore/" + STREAMING_DATA_SOURCE, \
      format="csv", inferSchema="true", header="true")

data_array = df.collect()
a = 1
for row in data_array:
    row_dict = row.asDict()
    # row_str = json.dumps(row_dict)
    with open('StreamData/data{}.txt'.format(a), 'w') as writefile:
        json.dump(row_dict, writefile)
        print('creating file log{}.txt'.format(a))
        a += 1
        time.sleep(5)

spark.stop()