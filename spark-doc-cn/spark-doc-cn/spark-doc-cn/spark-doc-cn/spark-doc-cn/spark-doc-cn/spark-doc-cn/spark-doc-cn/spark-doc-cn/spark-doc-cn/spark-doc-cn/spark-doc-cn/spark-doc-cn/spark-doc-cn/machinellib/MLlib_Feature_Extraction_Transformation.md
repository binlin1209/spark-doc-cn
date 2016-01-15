# MLlib - Feature Extraction and Transformation(特征提取和转化)
* TF-IDF
* Word2Vec
  - Model
  - Example
* StandardScaler
  - Model Fitting
  - Example
* Normalizer
  - Example
* Feature selection
  - ChiSqSelector
     * Model Fitting
     * Example
* ElementwiseProduct
  - Example
* PCA
  - Example
  
## TF-IDF

Term frequency-inverse document frequency (TF-IDF) is a feature vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus. Denote a term by t, a document by d, and the corpus by D. Term frequency TF(t,d) is the number of times that term t appears in document d, while document frequency DF(t,D) is the number of documents that contains term t. If we only use term frequency to measure the importance, it is very easy to over-emphasize terms that appear very often but carry little information about the document, e.g., “a”, “the”, and “of”. If a term appears very often across the corpus, it means it doesn’t carry special information about a particular document. Inverse document frequency is a numerical measure of how much information a term provides:

TF-IDF是一个特征向量化方法广泛应用与文本挖掘来反映一个术语的重要性语料库中的一个文档。用t来表示一个术语，用d来表示一个文档，用D来表示一个语料库。术语频率TF(t,d)是术语t在文档d中出现的次数，然后文档频率DF(t,D)是指包含术语t的文档数目。如果我们仅仅使用术语频率来测量重要性，这是非常容易过分强调术语，就是哪些经常出现，但表示很少的信息的，比如 “a”, “the”, 和 “of”。如果一个术语在语料库中经常出现，这意味着它不能代表特殊的信息，对于一个特别的文档。相反的文档频率是一个关于术语提供了多少信息的数量测量。

IDF(t,D)=log|D|+1DF(t,D)+1,

where |D| is the total number of documents in the corpus. Since logarithm is used, if a term appears in all documents, its IDF value becomes 0. Note that a smoothing term is applied to avoid dividing by zero for terms outside the corpus. The TF-IDF measure is simply the product of TF and IDF

在语料库中|D|代表了所以文档的数量。自从对数被使用后，如果一个术语在所有的文档中都出现，它的IDF值变成0。注意一个“润滑”的术语被用来避免被0分解，对于在语料库之外的术语。这个TF-IDF测量是一个简单的TF和IDF的产物。

TFIDF(t,d,D)=TF(t,d)⋅IDF(t,D).

There are several variants on the definition of term frequency and document frequency. In MLlib, we separate TF and IDF to make them flexible.

这有若干不同的变量在定义术语频率和文档频率中。在MLlib中，我们把TF和IDF分来，使它们更灵活。

Our implementation of term frequency utilizes the hashing trick. A raw feature is mapped into an index (term) by applying a hash function. Then term frequencies are calculated based on the mapped indices. This approach avoids the need to compute a global term-to-index map, which can be expensive for a large corpus, but it suffers from potential hash collisions, where different raw features may become the same term after hashing. To reduce the chance of collision, we can increase the target feature dimension, i.e., the number of buckets of the hash table. The default feature dimension is 220=1,048,576.

我们对于术语频率的实现是通过使用哈希方法来实现的。通过运用一个哈希函数，一个原生的特征被映射到一个术语索引中。根据映射的索引值，术语频率就可以被计算得出。这种方法避免了需要计算全局术语索引映射，对于一个大的语料库而言，但是会遭受潜在的哈希冲突，在那里不同的原生特征可能会变成一个相同的术语，在哈希之后。为了减少冲突的机会，我们可以提高目标特征维数，哈希表的桶数目。这个默认的特征维数是2**20=1,048,576。

**Note**: MLlib doesn’t provide tools for text segmentation. We refer users to the Stanford NLP Group and scalanlp/chalk.

**注意**： MLlib没有提供文档分割的工具。

### python

TF and IDF are implemented in HashingTF and IDF. HashingTF takes an RDD of list as the input. Each record could be an iterable of strings or other types.

TF和IDF在哈希TF和IDF中被实施。哈希TF采用了一个RDD的列表作为一个输入。每个记录能够作为一个字符串的迭代或者其它类型。

	from pyspark import SparkContext
	from pyspark.mllib.feature import HashingTF
	
	sc = SparkContext()
	
	# Load documents (one per line).
	documents = sc.textFile("...").map(lambda line: line.split(" "))
	
	hashingTF = HashingTF()
	tf = hashingTF.transform(documents)

While applying HashingTF only needs a single pass to the data, applying IDF needs two passes: first to compute the IDF vector and second to scale the term frequencies by IDF.

尽管运用哈希TF仅仅需要一个数据通过，运用IDF需要两个通过：首先需要计算IDF向量，其次通过IDF测量这个术语的频率。

	from pyspark.mllib.feature import IDF
	
	# ... continue from the previous example
	tf.cache()
	idf = IDF().fit(tf)
	tfidf = idf.transform(tf)

MLLib’s IDF implementation provides an option for ignoring terms which occur in less than a minimum number of documents. In such cases, the IDF for these terms is set to 0. This feature can be used by passing the minDocFreq value to the IDF constructor.

MLlib的IDF的实现提供了一个忽视术语的选项，这发生在一个小于一个最小的文档数目。在这种情况下，这些的术语的IDF被设置为0。通过传递minDocFreq值到IDF构造器，这个特征能够被使用。

	# ... continue from the previous example
	tf.cache()
	idf = IDF(minDocFreq=2).fit(tf)
	tfidf = idf.transform(tf)
	
## Word2Vec

Word2Vec computes distributed vector representation of words. The main advantage of the distributed representations is that similar words are close in the vector space, which makes generalization to novel patterns easier and model estimation more robust. Distributed vector representation is showed to be useful in many natural language processing applications such as named entity recognition, disambiguation, parsing, tagging and machine translation.

Word2Vec计算分布式向量代表单词。分布式方法的主要优势就是相似的单词放在相似的向量空间中。这使得新模式的概括更加容易，模型估计更加强健。分布式向量表示被认为是很有用的，在许多自然语言的执行应用中，例如被命名的整体识别，消歧、解析、标签和机器翻译。

### Model

In our implementation of Word2Vec, we used skip-gram model. The training objective of skip-gram is to learn word vector representations that are good at predicting its context in the same sentence. Mathematically, given a sequence of training words w1,w2,…,wT, the objective of the skip-gram model is to maximize the average log-likelihood 

在我们执行Word2Vec的过程中，我们使用skip-gram模式。skip-gram的训练目标就是学习单词向量表示，这个在相同的句子中，具有非常好的预测效果。在数学上，假设有一个训练单词的顺序表，skip-gram模型的目标就是最大化对数函数的可能性。

1T∑t=1T∑j=−kj=klogp(wt+j|wt)

where k is the size of the training window.

k是训练窗口的大小。

In the skip-gram model, every word w is associated with two vectors uw and vw which are vector representations of w as word and context respectively. The probability of correctly predicting word wi given word wj is determined by the softmax model, which is

在skip-gram模型中，每个词w适合两个向量uw和vw相联系的，分别代表了单词和内容。正确预测单词的概率比如wj是由softmax模型决定的， 

p(wi|wj)=exp(u⊤wivwj)∑Vl=1exp(u⊤lvwj)

where V is the vocabulary size.

V是词汇表的大小。

The skip-gram model with softmax is expensive because the cost of computing logp(wi|wj) is proportional to V, which can be easily in order of millions. To speed up training of Word2Vec, we used hierarchical softmax, which reduced the complexity of computing of logp(wi|wj) to O(log(V))

这个skip-gram模型的softmax是昂贵的因为计算logp的成本对于V而言是成比例的。就算数量是百万计的也是很容易的。为了加快Word2Vec的训练，我们使用分层的softmax，这减少了计算logp的复杂性。

## Example

The example below demonstrates how to load a text file, parse it as an RDD of Seq[String], construct a Word2Vec instance and then fit a Word2VecModel with the input data. Finally, we display the top 40 synonyms of the specified word. To run the example, first download the text8 data and extract it to your preferred directory. Here we assume the extracted file is text8 and in same directory as you run the spark shell.

下面的例子演示了怎样加载一个text文件，把它解析为一个RDD，构建一个Word2Vec的事件，适应一个Word2VecModel作为一个输入数据。最后，我们展示了最多了特定单词的同义词。为了运行这个例子，首先加载text数据和提取它到你预先设置的文件夹中。在这里我们假设这个提取的文件就是text8，你可以运行这个spark shell在相同的文件夹中。

### python

	from pyspark import SparkContext
	from pyspark.mllib.feature import Word2Vec
	
	sc = SparkContext(appName='Word2Vec')
	inp = sc.textFile("text8_lines").map(lambda row: row.split(" "))
	
	word2vec = Word2Vec()
	model = word2vec.fit(inp)
	
	synonyms = model.findSynonyms('china', 40)
	
	for word, cosine_distance in synonyms:
	    print("{}: {}".format(word, cosine_distance))
	    
## StandardScaler

Standardizes features by scaling to unit variance and/or removing the mean using column summary statistics on the samples in the training set. This is a very common pre-processing step.

在样本训练事件中，通过衡量单元变化或者通过列概要统计移除平均值，来标准化特征。在预先处理的步骤中，这是一个非常常见的步骤。

For example, RBF kernel of Support Vector Machines or the L1 and L2 regularized linear models typically work better when all features have unit variance and/or zero mean.

例如，对于支持向量机的RBF核函数或者规则化线性模型的L1和L2都运行的很好，当所有的特征有单位变量或者0平均值时。

Standardization can improve the convergence rate during the optimization process, and also prevents against features with very large variances exerting an overly large influence during model training.

在优化过程中，标准化能够提高收敛速度，在模型训练中，也防止对特性有很大变化从而施加过于巨大的影响力。

### Model Fitting

StandardScaler has the following parameters in the constructor:

* withMean False by default. Centers the data with mean before scaling. It will build a dense output, so this does not work on sparse input and will raise an exception.
* withStd True by default. Scales the data to unit standard deviation.
We provide a fit method in StandardScaler which can take an input of RDD[Vector], learn the summary statistics, and then return a model which can transform the input dataset into unit standard deviation and/or zero mean features depending how we configure the StandardScaler.

在构造体中，标准衡量具有下列的参数：
* withMean参数默认的是False.在衡量之前，把数据的平均值居中。这将会建立一个密集的输出值，所以对于稀疏的输入值是没有作用的，也会引起一个异常。
* withStd默认值是True.计算这个数据的单位标准差。在StandardScaler中，我们提供了一个适应的方法，该方法能够提取RDD的输入值，学习概要统计，然后返回一个模型，这个模型能偶转化一个输入数据集为单位标准差，或者0平均特性，这些特性依赖于我们怎样配置StandardScaler。

This model implements VectorTransformer which can apply the standardization on a Vector to produce a transformed Vector or on an RDD[Vector] to produce a transformed RDD[Vector].

这个模型运行在那些能够运用向量标准化来产生一个转化的向量或者一个RDD来生成一个转化的RDD。

Note that if the variance of a feature is zero, it will return default 0.0 value in the Vector for that feature.

注意：如果一个特征值设为0，这将返回一个默认值0.0在这个特征向量中。

### Example
The example below demonstrates how to load a dataset in libsvm format, and standardize the features so that the new features have unit standard deviation and/or zero mean.

下面的例子演示了怎样加载一个libsvm格式加载一个数据集，然后标准化这个特征，是为了这个新特征拥有单位标准差或者0平均值。

### python

	from pyspark.mllib.util import MLUtils
	from pyspark.mllib.linalg import Vectors
	from pyspark.mllib.feature import StandardScaler
	
	data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	label = data.map(lambda x: x.label)
	features = data.map(lambda x: x.features)
	
	scaler1 = StandardScaler().fit(features)
	scaler2 = StandardScaler(withMean=True, withStd=True).fit(features)
	# scaler3 is an identical model to scaler2, and will produce identical transformations
	scaler3 = StandardScalerModel(scaler2.std, scaler2.mean)
	
	
	# data1 will be unit variance.
	data1 = label.zip(scaler1.transform(features))
	
	# Without converting the features into dense vectors, transformation with zero mean will raise
	# exception on sparse vector.
	# data2 will be unit variance and zero mean.
	data2 = label.zip(scaler1.transform(features.map(lambda x: Vectors.dense(x.toArray()))))
	
## Normalizer
Normalizer scales individual samples to have unit Lp norm. This is a common operation for text classification or clustering. For example, the dot product of two L2 normalized TF-IDF vectors is the cosine similarity of the vectors.

标准化衡量特殊化的例子来得到单元Lp 标准。对于文本分类或者聚集，这是一个通用的操作。例如，对于点产物的两个L2标准化的TF-IDF向量是向量的余弦近似的。

Normalizer has the following parameter in the constructor:

在构造体中，标准化具有下面的参数：

* p Normalization in Lp space, p=2 by default.

* p标准化在Lp空间中，默认值p = 2

Normalizer implements VectorTransformer which can apply the normalization on a Vector to produce a transformed Vector or on an RDD[Vector] to produce a transformed RDD[Vector].

标准化实施VectorTransformer方法，这个方法能够运用在向量标准化中，能够生成一个转化的向量或者在RDD中生成一个转化的RDD。

Note that if the norm of the input is zero, it will return the input vector.

注意：如果标准输入设置为0，这将返回一个输入向量。

## Example
The example below demonstrates how to load a dataset in libsvm format, and normalizes the features with L2 norm, and L∞ norm.

下面的例子演示了怎样在一个libsvm格式中加载一个数据集，然后用L2准则和L∞准则来标准化特征。

### python

	from pyspark.mllib.util import MLUtils
	from pyspark.mllib.linalg import Vectors
	from pyspark.mllib.feature import Normalizer
	
	data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	labels = data.map(lambda x: x.label)
	features = data.map(lambda x: x.features)
	
	normalizer1 = Normalizer()
	normalizer2 = Normalizer(p=float("inf"))
	
	# Each sample in data1 will be normalized using $L^2$ norm.
	data1 = labels.zip(normalizer1.transform(features))
	
	# Each sample in data2 will be normalized using $L^\infty$ norm.
	data2 = labels.zip(normalizer2.transform(features))
	
## Feature selectionzero mean
Feature selection allows selecting the most relevant features for use in model construction. Feature selection reduces the size of the vector space and, in turn, the complexity of any subsequent operation with vectors. The number of features to select can be tuned using a held-out validation set.

特征选择允许选择最近的相关特征，在模型构建中使用。特征选择减少了向量空间的大小，反过来，增加了随后的向量操作的复杂性。所选择的特征数目可以通过验证设置来调整。

## ChiSqSelector
ChiSqSelector stands for Chi-Squared feature selection. It operates on labeled data with categorical features. ChiSqSelector orders features based on a Chi-Squared test of independence from the class, and then filters (selects) the top features which the class label depends on the most. This is akin to yielding the features with the most predictive power.

ChiSqSelector代表了Chi-Squared特征选择。它作用于带安全标签的数据分类特性中。根据类中独立性Chi-Squared测试，ChiSqSelector整理了相关的特征，然后过滤了最明显的特征，这些特征主要是依靠类标签来决定的。这类似于最佳的预测能力的特性。

### Model Fitting

ChiSqSelector has the following parameters in the constructor:

在构造体中，ChiSqSelector具有下面的参数：

* numTopFeatures number of top features that the selector will select (filter).

* 选择器会选择最明显特征的numTopFeatures数目

We provide a fit method in ChiSqSelector which can take an input of RDD[LabeledPoint] with categorical features, learn the summary statistics, and then return a ChiSqSelectorModel which can transform an input dataset into the reduced feature space.

在ChiSqSelector中，我们提供了一个合适的方法，这个方法能够根据分类特征来提取RDD中的输入值，学习汇总统计，返回一个ChiSqSelectorModel模型，这个模型能够转化一个输入数据集为一个减少的特征空间。

This model implements VectorTransformer which can apply the Chi-Squared feature selection on a Vector to produce a reduced Vector or on an RDD[Vector] to produce a reduced RDD[Vector].

这个模型运行在那些能够运用Chi-Squared特征来产生一个减少的向量或者一个RDD来生成一个减少的RDD。

Note that the user can also construct a ChiSqSelectorModel by hand by providing an array of selected feature indices (which must be sorted in ascending order).

注意：通过手动的提供一个选择好的特征索引，用户能够构造一个ChiSqSelectorModel。

### Example

The following example shows the basic use of ChiSqSelector. The data set used has a feature matrix consisting of greyscale values that vary from 0 to 255 for each feature.

下面的例子表明了ChiSqSelector基本的用法。哪些被用的数据集有一个特征矩阵，包含一个 greyscale值，从0到255各对应一个特征。

### Scala

	import org.apache.spark.SparkContext._
	import org.apache.spark.mllib.linalg.Vectors
	import org.apache.spark.mllib.regression.LabeledPoint
	import org.apache.spark.mllib.util.MLUtils
	import org.apache.spark.mllib.feature.ChiSqSelector
	
	// Load some data in libsvm format
	val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	// Discretize data in 16 equal bins since ChiSqSelector requires categorical features
	// Even though features are doubles, the ChiSqSelector treats each unique value as a category
	val discretizedData = data.map { lp =>
	  LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map { x => (x / 16).floor } ) )
	}
	// Create ChiSqSelector that will select top 50 of 692 features
	val selector = new ChiSqSelector(50)
	// Create ChiSqSelector model (selecting features)
	val transformer = selector.fit(discretizedData)
	// Filter the top 50 features from each feature vector
	val filteredData = discretizedData.map { lp => 
	  LabeledPoint(lp.label, transformer.transform(lp.features)) 
	}
	
## ElementwiseProduct

ElementwiseProduct multiplies each input vector by a provided “weight” vector, using element-wise multiplication. In other words, it scales each column of the dataset by a scalar multiplier. This represents the Hadamard product between the input vector, v and transforming vector, w, to yield a result vector.

ElementwiseProduct 乘以每个输入向量，通过提供一个权值，使用element-wise相乘。换句话说，通过一个scalar乘数，它衡量了每一个数据集的列。这代表了输入向量v和转化向量w之间的Hadamard 产物，来产生一个结果向量。

⎛⎝⎜⎜⎜v1⋮vN⎞⎠⎟⎟⎟∘⎛⎝⎜⎜⎜w1⋮wN⎞⎠⎟⎟⎟=⎛⎝⎜⎜⎜v1w1⋮vNwN⎞⎠⎟⎟⎟

ElementwiseProduct has the following parameter in the constructor:

在构造体共ElementwiseProduct有下面的参数。

* w: the transforming vector.

* w: 转化的向量

ElementwiseProduct implements VectorTransformer which can apply the weighting on a Vector to produce a transformed Vector or on an RDD[Vector] to produce a transformed RDD[Vector].

ElementwiseProduct 实施VectorTransformer，这能够在向量中运用权值，来产生一个转化的向量或者在RDD中来产生一个转化的RDD。

## Example

This example below demonstrates how to transform vectors using a transforming vector value.

使用一个转化向量值，下面的例子演示了怎样转化向量。

### python

	from pyspark import SparkContext
	from pyspark.mllib.linalg import Vectors
	from pyspark.mllib.feature import ElementwiseProduct
	
	# Load and parse the data
	sc = SparkContext()
	data = sc.textFile("data/mllib/kmeans_data.txt")
	parsedData = data.map(lambda x: [float(t) for t in x.split(" ")])
	
	# Create weight vector.
	transformingVector = Vectors.dense([0.0, 1.0, 2.0])
	transformer = ElementwiseProduct(transformingVector)
	
	# Batch transform
	transformedData = transformer.transform(parsedData)
	# Single-row transform
	transformedData2 = transformer.transform(parsedData.first())
	
## PCA
A feature transformer that projects vectors to a low-dimensional space using PCA. Details you can read at dimensionality reduction.

使用PCA能够把一个一个向量映射到一个低维的空间中，来得到它的特征。详情可在维数降低中阅读。

## Example
The following code demonstrates how to compute principal components on a Vector and use them to project the vectors into a low-dimensional space while keeping associated labels for calculation a Linear Regression

下面的代码演示了在向量中怎样计算主成份分析，使用它们映射向量到一个低维的空间中，与此同时依然保持着线性回归的相关性标签的计算。

### Scala

	import org.apache.spark.mllib.regression.LinearRegressionWithSGD
	import org.apache.spark.mllib.regression.LabeledPoint
	import org.apache.spark.mllib.linalg.Vectors
	import org.apache.spark.mllib.feature.PCA
	
	val data = sc.textFile("data/mllib/ridge-data/lpsa.data").map { line =>
	  val parts = line.split(',')
	  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
	}.cache()
	
	val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
	val training = splits(0).cache()
	val test = splits(1)
	
	val pca = new PCA(training.first().features.size/2).fit(data.map(_.features))
	val training_pca = training.map(p => p.copy(features = pca.transform(p.features)))
	val test_pca = test.map(p => p.copy(features = pca.transform(p.features)))
	
	val numIterations = 100
	val model = LinearRegressionWithSGD.train(training, numIterations)
	val model_pca = LinearRegressionWithSGD.train(training_pca, numIterations)
	
	val valuesAndPreds = test.map { point =>
	  val score = model.predict(point.features)
	  (score, point.label)
	}
	
	val valuesAndPreds_pca = test_pca.map { point =>
	  val score = model_pca.predict(point.features)
	  (score, point.label)
	}
	
	val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
	val MSE_pca = valuesAndPreds_pca.map{case(v, p) => math.pow((v - p), 2)}.mean()
	
	println("Mean Squared Error = " + MSE)
	println("PCA Mean Squared Error = " + MSE_pca)
