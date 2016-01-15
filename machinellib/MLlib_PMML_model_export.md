# MLlib - PMML model export
* MLlib supported models
* Examples

## MLlib supported models
MLlib supports model export to Predictive Model Markup Language (PMML).

MLlib支持导出模型为预测模型标记语言

The table below outlines the MLlib models that can be exported to PMML and their equivalent PMML model.

下面的表格中概括了MLlib模型能够导出为PMML和等价的PMML模型

<table class="table">  <thead>    <tr><th>MLlib model</th><th>PMML model</th></tr>  </thead>  <tbody>    <tr>      <td>KMeansModel</td><td>ClusteringModel</td>    </tr>        <tr>      <td>LinearRegressionModel</td><td>RegressionModel (functionName="regression")</td>    </tr>    <tr>      <td>RidgeRegressionModel</td><td>RegressionModel (functionName="regression")</td>    </tr>    <tr>      <td>LassoModel</td><td>RegressionModel (functionName="regression")</td>    </tr>    <tr>      <td>SVMModel</td><td>RegressionModel (functionName="classification" normalizationMethod="none")</td>    </tr>    <tr>      <td>Binary LogisticRegressionModel</td><td>RegressionModel (functionName="classification" normalizationMethod="logit")</td>    </tr>  </tbody></table>

## Examples

### Scala

To export a supported model (see table above) to PMML, simply call model.toPMML.
为了导出所支持的模型为PMML，只需要调用model.toPMML命令就可以了。

Here a complete example of building a KMeansModel and print it out in PMML format:

下面是一个完整的例子来构建一个KMeansModel模型，然后按照PMML的格式输出。

	import org.apache.spark.mllib.clustering.KMeans
	import org.apache.spark.mllib.linalg.Vectors
	
	// Load and parse the data
	val data = sc.textFile("data/mllib/kmeans_data.txt")
	val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
	
	// Cluster the data into two classes using KMeans
	val numClusters = 2
	val numIterations = 20
	val clusters = KMeans.train(parsedData, numClusters, numIterations)
	
	// Export to PMML
	println("PMML Model:\n" + clusters.toPMML)

As well as exporting the PMML model to a String (model.toPMML as in the example above), you can export the PMML model to other formats:

为了能够导出PMML模型为一个字符串，你可以导出PMML模型为其它的格式：

	// Export the model to a String in PMML format
	clusters.toPMML
	
	// Export the model to a local file in PMML format
	clusters.toPMML("/tmp/kmeans.xml")
	
	// Export the model to a directory on a distributed file system in PMML format
	clusters.toPMML(sc,"/tmp/kmeans")
	
	// Export the model to the OutputStream in PMML format
	clusters.toPMML(System.out)

For unsupported models, either you will not find a .toPMML method or an IllegalArgumentException will be thrown.
对于哪些不支持的模型，你或许不会找到一个.toPMML方法或者IllegalArgumentException将会被弃用。


