# MLlib - Collaborative Filtering
* Collaborative filtering
  - Explicit vs. implicit feedback
  - Scaling of the regularization parameter
* Examples
* Tutorial

* 协同过滤
   - 明确的反馈VS含蓄的反馈
   - 规则化参数的扩展
* 例子
* 指南

## Collaborative filtering

Collaborative filtering is commonly used for recommender systems. These techniques aim to fill in the missing entries of a user-item association matrix. MLlib currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries. MLlib uses the alternating least squares (ALS) algorithm to learn these latent factors. The implementation in MLlib has the following parameters:

协同过滤通常被用作推荐系统。这些技术主要用来填充用户-项目相关矩阵的入口。MLlib目前支持以模型为基础的协同过滤。用户和产物被描述为一个很小的潜在因素，且可以用来预测缺失的入口。MLlib使用ALS算法来学习这些潜在因素。MLlib的实现具有下面的参数：

* numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
* rank is the number of latent factors in the model.
* iterations is the number of iterations to run.
* lambda specifies the regularization parameter in ALS.
* implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
* alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations.

* numBlocks是blocks的数目，用来并行计算的
* rank是model中潜在因素的数目、
* iterations是用来运行的迭代数目
* lambda指明了ALS中的规则化参数。
* implicitPrefs指明了是否使用明确的反馈ALS变量或者一个适应与含蓄的反馈数据
* alpaha是一个能用于含蓄反馈变量的参数。在优先观察中，控制着基线的信心。

## Explicit vs. implicit feedback

The standard approach to matrix factorization based collaborative filtering treats the entries in the user-item matrix as explicit preferences given by the user to the item.

根据协同过滤，矩阵因子分解的标准方法对待用户-项目的入口为明确的偏向，通过给定的用户项目。

It is common in many real-world use cases to only have access to implicit feedback (e.g. views, clicks, purchases, likes, shares etc.). The approach used in MLlib to deal with such data is taken from Collaborative Filtering for Implicit Feedback Datasets. Essentially instead of trying to model the matrix of ratings directly, this approach treats the data as a combination of binary preferences and confidence values. The ratings are then related to the level of confidence in observed user preferences, rather than explicit ratings given to items. The model then tries to find latent factors that can be used to predict the expected preference of a user for an item.

在真实使用的事件中，非常普遍的使用明确的反馈。使用在MLlib中的方法来处理数据是从含蓄反馈数据集协同过滤中取出的。本质上，代替直接使用矩阵的评级，这种方法把数据处理为二进制优先和信心值的结合。相关的评级是与观察的用户首选项相关的，而不是给定的项目评级。模型然后尝试发现潜在的因素，可以用于预测预期的用户首选项。

## Scaling of the regularization parameter

Since v1.1, we scale the regularization parameter lambda in solving each least squares problem by the number of ratings the user generated in updating user factors, or the number of ratings the product received in updating product factors. This approach is named “ALS-WR” and discussed in the paper “Large-Scale Parallel Collaborative Filtering for the Netflix Prize”. It makes lambda less dependent on the scale of the dataset. So we can apply the best parameter learned from a sampled subset to the full dataset and expect similar performance.

自从1.1版本，我们扩展了规则化参数lambda来解决每个小的问题，通过评级的数目，用户通过更新用户用户因素来产生的，或者评级的数目，通过更新产物因子来接受的。这种方法被命名为ALS-WR，在文献““Large-Scale Parallel Collaborative Filtering for the Netflix Prize”中被讨论了。它使得lambda在数据集扩展上依赖更少

## Examples

### python

In the following example we load rating data. Each row consists of a user, a product and a rating. We use the default ALS.train() method which assumes ratings are explicit. We evaluate the recommendation by measuring the Mean Squared Error of rating prediction.

在下面的例子中，我们加载评级数据。每一行包括一个用户，一个产物，一个评价。我们使用默认的ALS.train()方法，假设评级是显著的。

	from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
	
	# Load and parse the data
	data = sc.textFile("data/mllib/als/test.data")
	ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
	
	# Build the recommendation model using Alternating Least Squares
	rank = 10
	numIterations = 10
	model = ALS.train(ratings, rank, numIterations)
	
	# Evaluate the model on training data
	testdata = ratings.map(lambda p: (p[0], p[1]))
	predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
	ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
	print("Mean Squared Error = " + str(MSE))
	
	# Save and load model
	model.save(sc, "myModelPath")
	sameModel = MatrixFactorizationModel.load(sc, "myModelPath")
	
If the rating matrix is derived from other source of information (i.e., it is inferred from other signals), you can use the trainImplicit method to get better results.

如果评级矩阵是从其它的信息源衍生来的，你可以使用trainImplicit方法来获得更好的结果。

	# Build the recommendation model using Alternating Least Squares based on implicit ratings
	model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.01)
	
In order to run the above application, follow the instructions provided in the Self-Contained Applications section of the Spark Quick Start guide. Be sure to also include spark-mllib to your build file as a dependency.

为了能够运行上面的应用，请参考Spark Quick Start guide中的 Self-Contained Applications部分。请确保spark-mllib在你的安装文件中。

## Tutorial

The training exercises from the Spark Summit 2014 include a hands-on tutorial for personalized movie recommendation with MLlib.

