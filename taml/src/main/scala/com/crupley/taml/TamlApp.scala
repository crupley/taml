package com.crupley.taml

import java.io.{File, PrintWriter}

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.evaluator.LogLoss.binaryLogLoss
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
 * TransmogrifAI Auto-ML App
 *  - Uses TransmogrifAI's Auto-ML defaults to train a model from a csv file
 *
 *  @param args(0) - path to csv data file
 *  @param args(1) - name of response variable column
 */
object TamlApp {
  def main(args: Array[String]): Unit = {
    // Parse args
    val Array(path, responseField) = args

    // Initialize spark
    val conf = new SparkConf()
    conf.setAppName("taml")
    implicit val spark: SparkSession = SparkSession.builder.config(conf).getOrCreate()

    // Load data
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)

    // Extract response and predictor features
    val (responseRaw, predictors) = FeatureBuilder.fromDataFrame[Integral](data, response = responseField)
    val responseFeature = responseRaw.map(_.toDouble.toRealNN(0.0))

    // Automated feature engineering
    val featureVector = predictors.transmogrify()

    // Automated feature validation and selection
    val checkedFeatures = responseFeature.sanityCheck(featureVector, removeBadFeatures = true)

    // Automated model selection
    val pred = BinaryClassificationModelSelector.withTrainValidationSplit(trainTestEvaluators = Seq(binaryLogLoss))
      .setInput(responseFeature, checkedFeatures)
      .getOutput()

    // Setting up a TransmogrifAI workflow and training the model
    val model = new OpWorkflow().setInputDataset(data).setResultFeatures(pred).train()

    // Print and write results
    println("Model summary:\n" + model.summaryPretty())

    val writer = new PrintWriter(new File(s"$responseField-${math.random}.json"))
    writer.write(model.summary())
    writer.close()
  }
}
