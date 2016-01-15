# Machine Learning Library (MLlib) Guide

MLlib is Spark’s machine learning (ML) library. Its goal is to make practical machine learning scalable and easy. It consists of common learning algorithms and utilities, including classification, regression, clustering, collaborative filtering, dimensionality reduction, as well as lower-level optimization primitives and higher-level pipeline APIs.

MLlib是spark的机器学习包。它的目的是为了使实践性的机器学习更加可行和容易。包括了通用的学习算法和实用性，包括分类，回归，聚类，协同过滤，降维，和低级优化原语，高级管道API。

It divides into two packages:

* spark.mllib contains the original API built on top of RDDs.
* spark.ml provides higher-level API built on top of DataFrames for constructing ML pipelines.

主要分为两个包：

* spark.mllib包含在RDD顶层的原始API。
* spark.ml提供了更高级的建立于DataFrames上的API。


Using spark.ml is recommended because with DataFrames the API is more versatile and flexible. But we will keep supporting spark.mllib along with the development of spark.ml. Users should be comfortable using spark.mllib features and expect more features coming. Developers should contribute new algorithms to spark.ml if they fit the ML pipeline concept well, e.g., feature extractors and transformers.

使用spark.ml是被推荐的，因为在DataFrames下，API功能更强大且更加灵活。但是我们依然支持spark.mllib。使用spark.mllib的特性，用户应该更加舒服，期待更多的特性的到来。开发者应该贡献新的算法到spark.ml，如果它们很好的适合ML管道，比如特征提取器和转换器。

We list major functionality from both below, with links to detailed guides.

我们从下面列表列出了主要功能和详细指南的链接。

## spark.mllib: data types, algorithms, and utilities

* Data types
* Basic statistics
 - summary statistics
 - correlations
 - stratified sampling
 - hypothesis testing
 - random data generation
* Classification and regression
 - linear models (SVMs, logistic regression, linear regression)
 - naive Bayes
 - decision trees
 - ensembles of trees (Random Forests and Gradient-Boosted Trees)
 - isotonic regression
* Collaborative filtering
 - alternating least squares (ALS)
* Clustering
 - k-means
 - Gaussian mixture
 - power iteration clustering (PIC)
 - latent Dirichlet allocation (LDA)
 - streaming k-means
* Dimensionality reduction
 - singular value decomposition (SVD)
 - principal component analysis (PCA)
* Feature extraction and transformation
* Frequent pattern mining
 - FP-growth
 - association rules
 - PrefixSpan
* Evaluation metrics
* PMML model export
* Optimization (developer)
 - stochastic gradient descent
 - limited-memory BFGS (L-BFGS)

## spark.ml: high-level APIs for ML pipelines

spark.ml programming guide provides an overview of the Pipelines API and major concepts. It also contains sections on using algorithms within the Pipelines API, for example:

* Feature extraction, transformation, and selection
* Decision trees for classification and regression
* Ensembles
* Linear methods with elastic net regularization
* Multilayer perceptron classifier

## Dependencies
MLlib uses the linear algebra package Breeze, which depends on netlib-java for optimised numerical processing. If natives libraries1 are not available at runtime, you will see a warning message and a pure JVM implementation will be used instead.

Due to licensing issues with runtime proprietary binaries, we do not include netlib-java’s native proxies by default. To configure netlib-java / Breeze to use system optimised binaries, include com.github.fommil.netlib:all:1.1.2 (or build Spark with -Pnetlib-lgpl) as a dependency of your project and read the netlib-java documentation for your platform’s additional installation instructions.

To use MLlib in Python, you will need NumPy version 1.4 or newer.

## Migration guide

MLlib is under active development. The APIs marked Experimental/DeveloperApi may change in future releases, and the migration guide below will explain all changes between releases.

## From 1.4 to 1.5
In the spark.mllib package, there are no break API changes but several behavior changes:

* SPARK-9005: RegressionMetrics.explainedVariance returns the average regression sum of squares.
* SPARK-8600: NaiveBayesModel.labels become sorted.
* SPARK-3382: GradientDescent has a default convergence tolerance 1e-3, and hence iterations might end earlier than 1.4.
In the spark.ml package, there exists one break API change and one behavior change:

* SPARK-9268: Java’s varargs support is removed from Params.setDefault due to a Scala compiler bug.
* SPARK-10097: Evaluator.isLargerBetter is added to indicate metric ordering. Metrics like RMSE no longer flip signs as in 1.4.

## Previous Spark versions
Earlier migration guides are archived on this page.
---

1. To learn more about the benefits and background of system optimised natives, you may wish to watch Sam Halliday’s ScalaX talk on High Performance Linear Algebra in Scala. ↩





