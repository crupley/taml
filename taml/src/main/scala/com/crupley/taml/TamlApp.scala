package com.crupley.taml

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.{Feature, FeatureBuilder}
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object TamlApp {
  def main(args: Array[String]): Unit = {
    val path = args(0)

    // initialize spark
    val conf = new SparkConf()
    conf.setAppName("taml")
    implicit val spark: SparkSession = SparkSession.builder.config(conf).getOrCreate()

    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)

    // Extract response and predictor features
    val (responseRaw: Feature[Integral], predictors: Array[Feature[_]]) =
      FeatureBuilder.fromDataFrame[Integral](data, response = "Survived")
    val response = responseRaw.map(_.toDouble.toRealNN(0.0))

    // Automated feature engineering
    val featureVector = predictors.transmogrify()

    // Automated feature validation and selection
    val checkedFeatures = response.sanityCheck(featureVector, removeBadFeatures = true)

    // Automated model selection
    val pred = BinaryClassificationModelSelector().setInput(response, checkedFeatures).getOutput()

    // Setting up a TransmogrifAI workflow and training the model
    val model = new OpWorkflow().setInputDataset(data).setResultFeatures(pred).train()

    println("Model summary:\n" + model.summaryPretty())

  }
}
