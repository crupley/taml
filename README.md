# taml

> [TransmogrifAI](https://github.com/salesforce/TransmogrifAI)-based Auto-ML

The goal of this project is to make a simple application that will automatically train a "good" predictive model given only a CSV dataset and the name of the column containing the response variable. It basically adds some I/O to the main example in the [project readme](https://github.com/salesforce/TransmogrifAI/blob/master/README.md).

## Environment Requirements

* [Maven](https://maven.apache.org/)
* [Spark v2.3.2](https://spark.apache.org/)

## Usage

```bash
mvn install

spark-submit \
  --class com.crupley.taml.TamlApp \
  --master local[*] \
  target/taml-0.0.1-uber.jar \
  <path-to-csv> \
  <response-column>
```