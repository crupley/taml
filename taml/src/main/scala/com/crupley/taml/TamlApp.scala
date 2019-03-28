package com.crupley.taml

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types.RealNN
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object TamlApp {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setAppName("taml")
    implicit val spark = SparkSession.builder.config(conf).getOrCreate()

    val data = spark.createDataFrame(Seq.empty)

    // Extract response and predictor features
    val (survived, predictors) = FeatureBuilder.fromDataFrame[RealNN](data, response = "survived")

    // Automated feature engineering
    val featureVector = predictors.transmogrify()

    // Automated feature validation and selection
    val checkedFeatures = survived.sanityCheck(featureVector, removeBadFeatures = true)

    // Automated model selection
    val pred = BinaryClassificationModelSelector().setInput(survived, checkedFeatures).getOutput()

    // Setting up a TransmogrifAI workflow and training the model
    val model = new OpWorkflow().setInputDataset(data).setResultFeatures(pred).train()

    println("Model summary:\n" + model.summaryPretty())

  }
}
