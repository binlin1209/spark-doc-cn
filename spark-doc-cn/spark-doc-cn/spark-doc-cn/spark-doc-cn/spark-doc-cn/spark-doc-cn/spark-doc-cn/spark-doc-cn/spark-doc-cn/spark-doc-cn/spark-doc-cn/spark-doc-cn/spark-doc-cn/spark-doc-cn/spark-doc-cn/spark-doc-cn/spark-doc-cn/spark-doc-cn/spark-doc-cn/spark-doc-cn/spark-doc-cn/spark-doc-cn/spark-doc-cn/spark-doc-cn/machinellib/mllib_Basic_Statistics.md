# MLlib - Basic Statistics

* Summary statistics
* Correlations
* Stratified sampling
* Hypothesis testing
* Random data generation
* Kernel density estimation
* 概要统计
* 相关性
* 分层抽样
* 假设检验
* 随机数生成
* 核密度估计


## Summary statistics
We provide column summary statistics for RDD[Vector] through the function colStats available in Statistics.

我们提供了RDD的列的概要统计，通过统计学中可用的colStats函数。

### python

colStats() returns an instance of MultivariateStatisticalSummary, which contains the column-wise max, min, mean, variance, and number of nonzeros, as well as the total count.

colStats()返回一个多重统计概要的事件，这包含了列的最大值，最小值，平均值，变量，非零数，和数的和。

	from pyspark.mllib.stat import Statistics
	
	sc = ... # SparkContext
	
	mat = ... # an RDD of Vectors
	
	# Compute column summary statistics.
	summary = Statistics.colStats(mat)
	print(summary.mean())
	print(summary.variance())
	print(summary.numNonzeros())
	
## Correlations

Calculating the correlation between two series of data is a common operation in Statistics. In MLlib we provide the flexibility to calculate pairwise correlations among many series. The supported correlation methods are currently Pearson’s and Spearman’s correlation.

计算两个系列数值的相关性在统计学中是一个很常见的操作。在MLlib中，我们提供了灵活的方法来计算两列值的相关性。所支持的相关性方法是最流行的Pearson’s and Spearman’s相关性方法。

### python

Statistics provides methods to calculate correlations between series. Depending on the type of input, two RDD[Double]s or an RDD[Vector], the output will be a Double or the correlation Matrix respectively.

统计学提供方法来计算多系列值之间的相关性。依靠输入类型，两个RDD或者一个RDD，输出类型将会是一个double型或者相关矩阵。

	from pyspark.mllib.stat import Statistics
	
	sc = ... # SparkContext
	
	seriesX = ... # a series
	seriesY = ... # must have the same number of partitions and cardinality as seriesX
	
	# Compute the correlation using Pearson's method. Enter "spearman" for Spearman's method. If a 
	# method is not specified, Pearson's method will be used by default. 
	print(Statistics.corr(seriesX, seriesY, method="pearson"))
	
	data = ... # an RDD of Vectors
	# calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method.
	# If a method is not specified, Pearson's method will be used by default. 
	print(Statistics.corr(data, method="pearson"))
	
## Stratified sampling
Unlike the other statistics functions, which reside in MLlib, stratified sampling methods, sampleByKey and sampleByKeyExact, can be performed on RDD’s of key-value pairs. For stratified sampling, the keys can be thought of as a label and the value as a specific attribute. For example the key can be man or woman, or document ids, and the respective values can be the list of ages of the people in the population or the list of words in the documents. The sampleByKey method will flip a coin to decide whether an observation will be sampled or not, therefore requires one pass over the data, and provides an expected sample size. sampleByKeyExact requires significant more resources than the per-stratum simple random sampling used in sampleByKey, but will provide the exact sampling size with 99.99% confidence. sampleByKeyExact is currently not supported in python.

和其它的统计函数不同的是，这是属于MLlib，能够在RDD的key-value中执行。对于分层抽样方法，keys可以认为是一个标签，value值可以认为是一个特殊的属性。例如key值可以是男人，女人，文件的id。在文件中，不同的value值可以代表不同人的年龄。这种sampleByKey方法将像抛硬币一样来决定这种观察是否会被取样。因此需要一个通过数据，也提供一个预期的抽样大小。sampleByKeyExact需要更多重要的资源，而不是每个阶层简单的随机抽样。但是这将提供更加准确的抽样大小，准确率有99.99%。sampleByKeyExact目前还不支持python。

### python

sampleByKey() allows users to sample approximately ⌈fk⋅nk⌉∀k∈K items, where fk is the desired fraction for key k, nk is the number of key-value pairs for key k, and K is the set of keys.

sampleByKey()允许用户来抽样近似的⌈fk⋅nk⌉∀k∈K项目。fk是所期望的k部分。nk是key-value值对数目中的key。k是keys的集合。  

_Note_: sampleByKeyExact() is currently not supported in Python.

	sc = ... # SparkContext
	
	data = ... # an RDD of any key value pairs
	fractions = ... # specify the exact fraction desired from each key as a dictionary
	
	approxSample = data.sampleByKey(False, fractions);
	
## Hypothesis testing

Hypothesis testing is a powerful tool in statistics to determine whether a result is statistically significant, whether this result occurred by chance or not. MLlib currently supports Pearson’s chi-squared ( χ2) tests for goodness of fit and independence. The input data types determine whether the goodness of fit or the independence test is conducted. The goodness of fit test requires an input type of Vector, whereas the independence test requires a Matrix as input.

MLlib also supports the input type RDD[LabeledPoint] to enable feature selection via chi-squared independence tests.

假设测试在统计学中是一种很强大的工具，来决定一个结果在统计上是否重要，这种结果是否是偶然发生的。MLlib目前支持Pearson’s chi-squared测试，具有更好的拟合优度和独立性。输入的数据类型决定了这种拟合优度或者独立性测试是否被执行。拟合优度测试需要一个输入类型的vector对象。然而独立性测试需要一个matrix作为输入值。

MLlib也支持输入类型的RDD值来确保特性选择通过chi-squared独立性测试。

### python

Statistics provides methods to run Pearson’s chi-squared tests. The following example demonstrates how to run and interpret hypothesis tests.

统计学提供的方法来运行Pearson’s chi-squared测试。下面的例子表明怎样运行和解释假设测试。

	from pyspark import SparkContext
	from pyspark.mllib.linalg import Vectors, Matrices
	from pyspark.mllib.regresssion import LabeledPoint
	from pyspark.mllib.stat import Statistics
	
	sc = SparkContext()
	
	vec = Vectors.dense(...) # a vector composed of the frequencies of events
	
	# compute the goodness of fit. If a second vector to test against is not supplied as a parameter,
	# the test runs against a uniform distribution.
	goodnessOfFitTestResult = Statistics.chiSqTest(vec)
	print(goodnessOfFitTestResult) # summary of the test including the p-value, degrees of freedom,
	                               # test statistic, the method used, and the null hypothesis.
	
	mat = Matrices.dense(...) # a contingency matrix
	
	# conduct Pearson's independence test on the input contingency matrix
	independenceTestResult = Statistics.chiSqTest(mat)
	print(independenceTestResult)  # summary of the test including the p-value, degrees of freedom...
	
	obs = sc.parallelize(...)  # LabeledPoint(feature, label) .
	
	# The contingency table is constructed from an RDD of LabeledPoint and used to conduct
	# the independence test. Returns an array containing the ChiSquaredTestResult for every feature
	# against the label.
	featureTestResults = Statistics.chiSqTest(obs)
	
	for i, result in enumerate(featureTestResults):
	    print("Column $d:" % (i + 1))
	    print(result)

Additionally, MLlib provides a 1-sample, 2-sided implementation of the Kolmogorov-Smirnov (KS) test for equality of probability distributions. By providing the name of a theoretical distribution (currently solely supported for the normal distribution) and its parameters, or a function to calculate the cumulative distribution according to a given theoretical distribution, the user can test the null hypothesis that their sample is drawn from that distribution. In the case that the user tests against the normal distribution (distName="norm"), but does not provide distribution parameters, the test initializes to the standard normal distribution and logs an appropriate message.

然而，MLlib提供了一个KS的 1-sample, 2-sided的实现，为了更平等的概率分布。通过提供理论分布的名字和它的参数，或者一个函数来计算累计分布，根据一个给定的理论分布，用户能够测试一个空的假设，它们的样本是从它们的分布中提取的。在这种情况下，用户测试相对与正常的分布。但是并不提供分布参数，测试初始化标准的正常分布，记录相关信息。

### python

Statistics provides methods to run a 1-sample, 2-sided Kolmogorov-Smirnov test. The following example demonstrates how to run and interpret the hypothesis tests.

统计学提供的方法来运行Pearson’s chi-squared测试。下面的例子表明怎样运行和解释假设测试。

	from pyspark.mllib.stat import Statistics
	
	parallelData = sc.parallelize([1.0, 2.0, ... ])
	
	# run a KS test for the sample versus a standard normal distribution
	testResult = Statistics.kolmogorovSmirnovTest(parallelData, "norm", 0, 1)
	print(testResult) # summary of the test including the p-value, test statistic,
	                  # and null hypothesis
	                  # if our p-value indicates significance, we can reject the null hypothesis
	# Note that the Scala functionality of calling Statistics.kolmogorovSmirnovTest with
	# a lambda to calculate the CDF is not made available in the Python API
	
## Random data generation

Random data generation is useful for randomized algorithms, prototyping, and performance testing. MLlib supports generating random RDDs with i.i.d. values drawn from a given distribution: uniform, standard normal, or Poisson.

随机数生成对于随机算法，样机研究，性能测试，是非常有用的。MLlib支持生成随机的 i.i.d.值的RDD。这些值是从给定的uniform, standard normal, or Poisson提取的。

### python
RandomRDDs provides factory methods to generate random double RDDs or vector RDDs. The following example generates a random double RDD, whose values follows the standard normal distribution N(0, 1), and then map it to N(1, 4).

RandomRDDs提供了工厂模式来生成随机的double或者向量RDDS。下面的例子生成了一个随机的double型RDD。它们的值是沿着标准正常的分布式方向（1，0）然后map它到N（1，4）。

	from pyspark.mllib.random import RandomRDDs
	
	sc = ... # SparkContext
	
	# Generate a random double RDD that contains 1 million i.i.d. values drawn from the
	# standard normal distribution `N(0, 1)`, evenly distributed in 10 partitions.
	u = RandomRDDs.uniformRDD(sc, 1000000L, 10)
	# Apply a transform to get a random double RDD following `N(1, 4)`.
	v = u.map(lambda x: 1.0 + 2.0 * x)

## Kernel density estimation

Kernel density estimation is a technique useful for visualizing empirical probability distributions without requiring assumptions about the particular distribution that the observed samples are drawn from. It computes an estimate of the probability density function of a random variables, evaluated at a given set of points. It achieves this estimate by expressing the PDF of the empirical distribution at a particular point as the the mean of PDFs of normal distributions centered around each of the samples.

内核密度估计对于可视化经验性概率分布是非常有效的，并且不需要关于特定分布的假设。所观察到的样本就是从其中提取的。它计算了一个随机变量的概率密度函数的估计值，在一个给定的点集进行评估。通过表达的经验性分布的概率分布函数,在特殊的点，作为正常分布的概率分布函数的方法，集中在每个样本中。

### python

KernelDensity provides methods to compute kernel density estimates from an RDD of samples. The following example demonstrates how to do so.

核密度提供了计算核密度估计的RDD样本的方法。下面的例子演示了该怎样运行。

	from pyspark.mllib.stat import KernelDensity
	
	data = ... # an RDD of sample data
	
	# Construct the density estimator with the sample data and a standard deviation for the Gaussian
	# kernels
	kd = KernelDensity()
	kd.setSample(data)
	kd.setBandwidth(3.0)

	# Find density estimates for the given values
	densities = kd.estimate([-1.0, 2.0, 5.0])