# MLlib - Clustering

Clustering is an unsupervised learning problem whereby we aim to group subsets of entities with one another based on some notion of similarity. Clustering is often used for exploratory analysis and/or as a component of a hierarchical supervised learning pipeline (in which distinct classifiers or regression models are trained for each cluster).

聚类分析是一种非监督学习方法，然而我们的目标就是基于一些相似的概念收集子集实体。聚类问题被用作探究分析，或者作为一个监督式学习管道的成分。在那里，显著的分类或者回归模型被每个团簇模型训练所得到。

MLlib supports the following models:

* K-means
* Gaussian mixture
* Power iteration clustering (PIC)
* Latent Dirichlet allocation (LDA)
* Streaming k-means

## K-means

k-means is one of the most commonly used clustering algorithms that clusters the data points into a predefined number of clusters. The MLlib implementation includes a parallelized variant of the k-means++ method called kmeans||. The implementation in MLlib has the following parameters:

* k is the number of desired clusters.
* maxIterations is the maximum number of iterations to run.
* initializationMode specifies either random initialization or initialization via k-means||.
* runs is the number of times to run the k-means algorithm (k-means is not guaranteed to find a globally optimal solution, and when run multiple times on a given dataset, the algorithm returns the best clustering result).
* initializationSteps determines the number of steps in the k-means|| algorithm.
* epsilon determines the distance threshold within which we consider k-means to have converged.
* initialModel is an optional set of cluster centers used for initialization. If this parameter is supplied, only one run is performed.

K-means是一种最普遍聚类算法，把点聚集到预先定义的团簇数目中。MLlib的实现包括一个 k-means++ 方法的平行变量，称为kmeans||。MLlib的实现需要下面的参数：
* K是想要的团簇数目
* maxIterations 是运行的最大迭代数目
* initializationMode 详细说明了随机初始化或者通过k-means||初始化。
* runs是运行k-means算法的次数
* initializationSteps 决定了k-means||算法的步数
* epsilon决定了极限距离，我们认为k-means会收敛。
* initialModel是可选择的团簇中心来初始化的。如果这个参数已经提供了，那么只可以运行一次

__Examples__

### python

The following examples can be tested in the PySpark shell.

In the following example after loading and parsing data, we use the KMeans object to cluster the data into two clusters. The number of desired clusters is passed to the algorithm. We then compute Within Set Sum of Squared Error (WSSSE). You can reduce this error measure by increasing k. In fact the optimal k is usually one where there is an “elbow” in the WSSSE graph.

下面的例子可以在PySpark shell中测试。

在下面的例子中，通过加载和分析数据，我们通过KMeans方法来把数据聚集为两个团簇。所需数量的团簇是通过算法来传递的。我们然后计算WSSSE。你能够减少错误方法通过提高K的值。实际上最佳的k通常是位于“WSSSE“图的肘部的。

	from pyspark.mllib.clustering import KMeans, KMeansModel
	from numpy import array
	from math import sqrt
	
	# Load and parse the data
	data = sc.textFile("data/mllib/kmeans_data.txt")
	parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))
	
	# Build the model (cluster the data)
	clusters = KMeans.train(parsedData, 2, maxIterations=10,
	        runs=10, initializationMode="random")
	
	# Evaluate clustering by computing Within Set Sum of Squared Errors
	def error(point):
	    center = clusters.centers[clusters.predict(point)]
	    return sqrt(sum([x**2 for x in (point - center)]))
	
	WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
	print("Within Set Sum of Squared Error = " + str(WSSSE))
	
	# Save and load model
	clusters.save(sc, "myModelPath")
	sameModel = KMeansModel.load(sc, "myModelPath")
	
## Gaussian mixture

A Gaussian Mixture Model represents a composite distribution whereby points are drawn from one of k Gaussian sub-distributions, each with its own probability. The MLlib implementation uses the expectation-maximization algorithm to induce the maximum-likelihood model given a set of samples. The implementation has the following parameters:

* k is the number of desired clusters.
* convergenceTol is the maximum change in log-likelihood at which we consider convergence achieved.
* maxIterations is the maximum number of iterations to perform without reaching convergence.
* initialModel is an optional starting point from which to start the EM algorithm. If this parameter is omitted, a random starting point will be constructed from the data.

一个高斯混合模型代表一个复合的分布模型。然而点是从k Gaussian子分布提取的，每一个都有它们自己的概率。MLlib的实现使用预期-最大化算法来推导出最大可能性的模型，通过一系列给定的模型。上述方法的实现有下面的参数：
* K是所期望的团簇数量
* 收敛的TOl是最大的改变，在所有加载的可能性中，我们都认为已经收敛了。
* maxIterations 是最大的迭代数目来执行，而没有达到收敛
* initialModel是最优化的起点，从EM算法开始。如果这个参数被忽略，一个随机的起点会被创建。

__Examples__

### python

In the following example after loading and parsing data, we use a GaussianMixture object to cluster the data into two clusters. The number of desired clusters is passed to the algorithm. We then output the parameters of the mixture model.

在下面的例子中，在加载和分析数据之后，我们使用 GaussianMixture来聚集数据，从而形成两个团簇。预期的团簇数目是通过算法来传递的。我们然后输出混合模型的参数。

	from pyspark.mllib.clustering import GaussianMixture
	from numpy import array
	
	# Load and parse the data
	data = sc.textFile("data/mllib/gmm_data.txt")
	parsedData = data.map(lambda line: array([float(x) for x in line.strip().split(' ')]))
	
	# Build the model (cluster the data)
	gmm = GaussianMixture.train(parsedData, 2)
	
	# output parameters of model
	for i in range(2):
	    print ("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
	        "sigma = ", gmm.gaussians[i].sigma.toArray())
	        
## Power iteration clustering (PIC)

Power iteration clustering (PIC) is a scalable and efficient algorithm for clustering vertices of a graph given pairwise similarties as edge properties, described in Lin and Cohen, Power Iteration Clustering. It computes a pseudo-eigenvector of the normalized affinity matrix of the graph via power iteration and uses it to cluster vertices. MLlib includes an implementation of PIC using GraphX as its backend. It takes an RDD of (srcId, dstId, similarity) tuples and outputs a model with the clustering assignments. The similarities must be nonnegative. PIC assumes that the similarity measure is symmetric. A pair (srcId, dstId) regardless of the ordering should appear at most once in the input data. If a pair is missing from input, their similarity is treated as zero. MLlib’s PIC implementation takes the following (hyper-)parameters:

* k: number of clusters
* maxIterations: maximum number of power iterations
* initializationMode: initialization model. This can be either “random”, which is the default, to use a random vector as vertex properties, or “degree” to use normalized sum similarities.

PIC是一个可称量和有效率的算法，对于一个图表的团簇顶点，考虑到边缘效应的相似性。它计算了特征向量的赝势，图表中正常的密切相关矩阵，通过能量迭代和使用它称为团簇顶点。MLlib包括一个PIC的实现使用GraphX作为它的后端。它采用了RDD的一个元组，然后输出一个团簇任务模型。相似性必须非负。PIC假设相似性测量是对称的。不管顺序，pair (srcId, dstId)应该在输入的数据中，出现一次。如果一对数据从输入值中消失，它们的相似性就被认为是0。MLlib的PIC实现采用了下面的参数。

* k：团簇的数目
* maxIterations：力量迭代的最大数目
* initializationMode：初始化模型。这或许可以是随机的，这是默认值，来使用一个随机向量作为顶点性能，或者使用正常值和的相似性。

__Examples__

In the following, we show code snippets to demonstrate how to use PIC in MLlib.

在下面例子中，我们通过代码的片段来演示在MLlib中怎样使用PIC。

### python

PowerIterationClustering implements the PIC algorithm. It takes an RDD of (srcId: Long, dstId: Long, similarity: Double) tuples representing the affinity matrix. Calling PowerIterationClustering.run returns a PowerIterationClusteringModel, which contains the computed clustering assignments.

PowerIterationClustering 实现PIC算法。它采用了RDD中的元祖来代表矩阵的密切关系。调用PowerIterationClustering.run 返回一个PowerIterationClusteringModel，这包括了所计算的团簇任务。

	from __future__ import print_function
	from pyspark.mllib.clustering import PowerIterationClustering, PowerIterationClusteringModel
	
	# Load and parse the data
	data = sc.textFile("data/mllib/pic_data.txt")
	similarities = data.map(lambda line: tuple([float(x) for x in line.split(' ')]))
	
	# Cluster the data into two classes using PowerIterationClustering
	model = PowerIterationClustering.train(similarities, 2, 10)
	
	model.assignments().foreach(lambda x: print(str(x.id) + " -> " + str(x.cluster)))
	
	# Save and load model
	model.save(sc, "myModelPath")
	sameModel = PowerIterationClusteringModel.load(sc, "myModelPath")
	
## Latent Dirichlet allocation (LDA)

Latent Dirichlet allocation (LDA) is a topic model which infers topics from a collection of text documents. LDA can be thought of as a clustering algorithm as follows:

* Topics correspond to cluster centers, and documents correspond to examples (rows) in a dataset.
* Topics and documents both exist in a feature space, where feature vectors are vectors of word counts (bag of words).
* Rather than estimating a clustering using a traditional distance, LDA uses a function based on a statistical model of how text documents are generated.

LDA是一个主题模型，能够从一系列文本文件的集合中推断出主题。LDA可以认为是一个团簇算法，如下：

* 主题对应于团簇中央。文件对应于数据集中的行。
* 主题和文件都存在于特定的空间，特征向量是单词统计的向量。
* 不是使用一个传统的距离来评估一个团簇，根据统计模型LDA使用一个函数来判断怎样生成一个文件。

LDA supports different inference algorithms via setOptimizer function. EMLDAOptimizer learns clustering using expectation-maximization on the likelihood function and yields comprehensive results, while OnlineLDAOptimizer uses iterative mini-batch sampling for online variational inference and is generally memory friendly.

通过setOptimizer函数，LDA支持不同的推断算法。使用预期-最大化函数， EMLDAOptimizer学习团簇，然后生成一个全面的结果，然而OnlineLDAOptimizer使用可迭代的最小的批次样本，对于在线的变量推断，而且通常而言是记忆友好的。

LDA takes in a collection of documents as vectors of word counts and the following parameters (set using the builder pattern):

* k: Number of topics (i.e., cluster centers)
* optimizer: Optimizer to use for learning the LDA model, either EMLDAOptimizer or OnlineLDAOptimizer
* docConcentration: Dirichlet parameter for prior over documents’ distributions over topics. Larger values encourage smoother inferred distributions.
* topicConcentration: Dirichlet parameter for prior over topics’ distributions over terms (words). Larger values encourage smoother inferred distributions.
* maxIterations: Limit on the number of iterations.
* checkpointInterval: If using checkpointing (set in the Spark configuration), this parameter specifies the frequency with which checkpoints will be created. If maxIterations is large, using checkpointing can help reduce shuffle file sizes on disk and help with failure recovery.

LDA发生于文件的收集作为一个单词统计的向量，下面的参数：
＊　k:主题的个数
＊　optimizer：optimizer被用作学习LDA模型，或者是EMLDAOptimizer或者是OnlineLDAOptimizer
＊　docConcentration:边界条件参数对于优先的主题文件分布。更大的值使得有更加平滑的推断分布。
＊　topicConcentration：边界条件参数对于优先的主题分布。更大的值使得有更加平滑的推断分布。
＊　maxIterations：限制了迭代的数量
＊　checkpointInterval：如果使用checkpointing，这个参数指明了将要创建的checkpoints的频率。如果maxIterations很大，使用checkpointing能够帮助减少

All of MLlib’s LDA models support:

* describeTopics: Returns topics as arrays of most important terms and term weights
* topicsMatrix: Returns a vocabSize by k matrix where each column is a topic

所有的MLlib的LDA模型支持：
＊　describeTopics：返回主题作为数组的最重要条目和术语权重
＊　topicsMatrix：通过k矩阵返回一个vocabSize。每一列都是一个主题。

_Note_: LDA is still an experimental feature under active development. As a result, certain features are only available in one of the two optimizers / models generated by the optimizer. Currently, a distributed model can be converted into a local model, but not vice-versa.

_注意_：尽管发展非常活跃，LDA仍然是一个经验性的特征。作为结果，确定的特征仅仅是可用的，在二选一的优化结构/模型中。目前为止，一个分布式的模型能够转化为一个本地模型，而不是vice-versa。

The following discussion will describe each optimizer/model pair separately.

下面的讨论将会分别描述每个优化模型/模型对。

__Expectation Maximization__

Implemented in EMLDAOptimizer and DistributedLDAModel.

For the parameters provided to LDA:

* docConcentration: Only symmetric priors are supported, so all values in the provided k-dimensional vector must be identical. All values must also be >1.0. Providing Vector(-1) results in default behavior (uniform k dimensional vector with value (50/k)+1
* topicConcentration: Only symmetric priors supported. Values must be >1.0. Providing -1 results in defaulting to a value of 0.1+1.
* maxIterations: The maximum number of EM iterations.

在 EMLDAOptimizer 和 DistributedLDAModel 中实施
对于在LDA中提供的参数：

* docConcentration：只有对称优先是被支持的。所以所有的值在提供的k维向量中是完全相同的。所有的值都必须大于1.0。所提供的向量导致默认的行为+1。
* topicConcentration：只有对称优先是被支持的。value必须大于1.0。提供的-1导致默认值变成了0.1+1。
* maxIterations: EM迭代的最大值。


EMLDAOptimizer produces a DistributedLDAModel, which stores not only the inferred topics but also the full training corpus and topic distributions for each document in the training corpus. A DistributedLDAModel supports:

* topTopicsPerDocument: The top topics and their weights for each document in the training corpus
* topDocumentsPerTopic: The top documents for each topic and the corresponding weight of the topic in the documents.
* logPrior: log probability of the estimated topics and document-topic distributions given the hyperparameters docConcentration and topicConcentration
* logLikelihood: log likelihood of the training corpus, given the inferred topics and document-topic distributions

EMLDAOptimizer生成了一个DistributedLDAModel，这存储了不仅仅是可推断的主题也包括完全训练的语料库，包括每个文件的主题分布。一个 DistributedLDAModel支持：

* topTopicsPerDocument：在训练的集合中每个文件的最顶层的话题和他们的权重。
* topDocumentsPerTopic：每个主题的最顶层文件和文件中相应的主题权重。
* logPrior：加载估计话题的可能性和文件话题的分布，通过给定的超参docConcentration和topicConcentration
* logLikelihood：记载训练语料的可能性，考虑到隐式主题和文件主题分布

__Online Variational Bayes__

Implemented in OnlineLDAOptimizer and LocalLDAModel.

在OnlineLDAOptimizer 和 LocalLDAModel模式下实施

For the parameters provided to LDA:

* docConcentration: Asymmetric priors can be used by passing in a vector with values equal to the Dirichlet parameter in each of the k dimensions. Values should be >=0. Providing Vector(-1) results in default behavior (uniform k dimensional vector with value (1.0/k))
* topicConcentration: Only symmetric priors supported. Values must be >=0. Providing -1 results in defaulting to a value of (1.0/k).
* maxIterations: Maximum number of minibatches to submit.

LDA中提供了下面的参数：
* docConcentration：通过传递向量值等于边界条件参数，对于k维中的每一维，非对称优先能够被使用。values应该大于等于0。所提供的矢量导致了默认的行为
* topicConcentration:只有对称优先是支持的。Values值必须大于等于0。提供的-1值导致了一个默认的值1.0/k
* maxIterations：最小的一批中的最大值用来提交

In addition, OnlineLDAOptimizer accepts the following parameters:

* miniBatchFraction: Fraction of corpus sampled and used at each iteration
* optimizeDocConcentration: If set to true, performs maximum-likelihood estimation of the hyperparameter docConcentration (aka alpha) after each minibatch and sets the optimized docConcentration in the returned LocalLDAModel
* tau0 and kappa: Used for learning-rate decay, which is computed by (τ0+iter)−κ where iter is the current number of iterations.

而且，OnlineLDAOptimizer 接受下面的参数：
* miniBatchFraction:在每次迭代中，语料库的片断都会被采用和使用
* optimizeDocConcentration：如果这个值设置为true，执行超参数的最大可能性估计，在每一个最小批之后，在返回的LocalLDAModel之后，设置最优化的docConcentration。
* tau0 and kappa：用于学习速率衰减，这是通过 (τ0+iter)?κ计算得到的，iter是目前迭代的数目。

OnlineLDAOptimizer produces a LocalLDAModel, which only stores the inferred topics. A LocalLDAModel supports:

* logLikelihood(documents): Calculates a lower bound on the provided documents given the inferred topics.
* logPerplexity(documents): Calculates an upper bound on the perplexity of the provided documents given the inferred topics.

OnlineLDAOptimizer 产生了一个 LocalLDAModel模型，这仅仅储存已经推断的主题。一个LocalLDAModel支持如下参数：
* logLikelihood(documents): 考虑到可推断的主题，在给定的文件中，计算一个更低的范围。
* logPerplexity(documents): 考虑到可推断的主题，在给定的混乱文件中，计算一个更高的范围。


__Examples__

In the following example, we load word count vectors representing a corpus of documents. We then use LDA to infer three topics from the documents. The number of desired clusters is passed to the algorithm. We then output the topics, represented as probability distributions over words.

在下面的例子中，我们加载一个字数统计向量来代表一个文件语料库。我们然后使用LDA来推断文件的三个主题。所想要的团簇数目是通过算法传递的。我们然后输出主题，代表着单词的概率分布。

### python

from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors

	# Load and parse the data
	data = sc.textFile("data/mllib/sample_lda_data.txt")
	parsedData = data.map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
	# Index documents with unique IDs
	corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()
	
	# Cluster the documents into three topics using LDA
	ldaModel = LDA.train(corpus, k=3)
	
	# Output topics. Each is a distribution over words (matching word count vectors)
	print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):")
	topics = ldaModel.topicsMatrix()
	for topic in range(3):
	    print("Topic " + str(topic) + ":")
	    for word in range(0, ldaModel.vocabSize()):
	        print(" " + str(topics[word][topic]))
			
	# Save and load model
	model.save(sc, "myModelPath")
	sameModel = LDAModel.load(sc, "myModelPath")
	
## Streaming k-means

When data arrive in a stream, we may want to estimate clusters dynamically, updating them as new data arrive. MLlib provides support for streaming k-means clustering, with parameters to control the decay (or “forgetfulness”) of the estimates. The algorithm uses a generalization of the mini-batch k-means update rule. For each batch of data, we assign all points to their nearest cluster, compute new cluster centers, then update each cluster using:


当数据到达流中的时候，我们想动态的评估团簇，当新数据到达的时候，更新他们。MLlib提供了支持流k-means团簇的方法，用参数来控制评估的衰减。这种算法使用了泛化的mini-batch k-means更新规则。对于每一批数据来说，我们指派所有的点到他们附近的集群，计算新的集群中心，然后更新它们的团簇：

ct+1=ctntα+xtmtntα+mt(1)
nt+1=nt+mt(2)

Where ct is the previous center for the cluster, nt is the number of points assigned to the cluster thus far, xt is the new cluster center from the current batch, and mt is the number of points added to the cluster in the current batch. The decay factor α can be used to ignore the past: with α=1 all data will be used from the beginning; with α=0 only the most recent data will be used. This is analogous to an exponentially-weighted moving average.

ct是前面的团簇中心，nt是点的数量，分配到团簇中，xt是当前批次的新的团簇中心，mt是点的数目，在当前批次中添加进团簇。衰退因子α能够被用作忽略过去：当α=1时，所有的数据从头开始被使用。当α=0 时，只有最近被使用的数据将会被使用。这类似于一个exponentially-weighted移动平均线。

The decay can be specified using a halfLife parameter, which determines the correct decay factor a such that, for data acquired at time t, its contribution by time t + halfLife will have dropped to 0.5. The unit of time can be specified either as batches or points and the update rule will be adjusted accordingly.

使用半生命参数，这种衰退因子能够被特殊化，这决定了正确的衰退参数，当数据在时间t获取时，时间的贡献量将从 t + halfLife下降为0.5。时间单元将会被特殊化为批次或者点，更新的规则也相应的被调整。

__Examples__

This example shows how to estimate clusters on streaming data.

这个例子表明了在流数据中，怎样估算团簇。

### python

First we import the neccessary classes.

	from pyspark.mllib.linalg import Vectors
	from pyspark.mllib.regression import LabeledPoint
	from pyspark.mllib.clustering import StreamingKMeans
Then we make an input stream of vectors for training, as well as a stream of labeled data points for testing. We assume a StreamingContext ssc has been created, see Spark Streaming Programming Guide for more info.

然后我们生成一个向量的输入流用来训练。也可以用一个已经标记的数据点流来进行测试。我们假设一个StreamingContext ssc已经被创建，参考Spark Streaming Programming Guide来获取更多的内容。

	def parse(lp):
	    label = float(lp[lp.find('(') + 1: lp.find(',')])
	    vec = Vectors.dense(lp[lp.find('[') + 1: lp.find(']')].split(','))
	    return LabeledPoint(label, vec)
	
	trainingData = ssc.textFileStream("/training/data/dir").map(Vectors.parse)
	testData = ssc.textFileStream("/testing/data/dir").map(parse)

We create a model with random clusters and specify the number of clusters to find

我们创建了一个拥有随机团簇的模型，指定团簇的数量来发现

	model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(3, 1.0, 0)
Now register the streams for training and testing and start the job, printing the predicted cluster assignments on new data points as they arrive.

现在注册一个流用来训练和测试，开始这个工作，在新的数据点中打印预期的任务团簇。

	model.trainOn(trainingData)
	print(model.predictOnValues(testData.map(lambda lp: (lp.label, lp.features))))
	
	ssc.start()
	ssc.awaitTermination()

As you add new text files with data the cluster centers will update. Each training point should be formatted as [x1, x2, x3], and each test data point should be formatted as (y, [x1, x2, x3]), where y is some useful label or identifier (e.g. a true category assignment). Anytime a text file is placed in /training/data/dir the model will update. Anytime a text file is placed in /testing/data/dir you will see predictions. With new data, the cluster centers will change!

当你添加新的数据文件时，团簇中心将会被更新。每一个训练的点将会被格式化为[x1, x2, x3]。每一个测试的数据点将会被格式化为(y, [x1, x2, x3])。y是一个有用的标签或者标识符。在任何时候，一个text文件，将会被放置在/training/data/dir中，模型将会被更新。在任何时候，一个text文件将会被放置在 /testing/data/dir，你将会看到预期的结果。当有新的数据时，团簇中心将会改变。