# MLlib - Frequent Pattern Mining(频率挖掘模式)

Mining frequent items, itemsets, subsequences, or other substructures is usually among the first steps to analyze a large-scale dataset, which has been an active research topic in data mining for years. We refer users to Wikipedia’s association rule learning for more information. MLlib provides a parallel implementation of FP-growth, a popular algorithm to mining frequent itemsets.

挖掘频繁项目,项目集,子序列或其他子结构之间通常是第一步分析大规模数据集，多年来在数据挖掘领域一直是一个非常活跃的课题。用户可以在维基上搜索相关的规则学习来获得更多的消息。MLlib对于FP-growth提供了一个并行运算,在频率挖掘项目集这是一个非常流行的算法。

## FP-growth

The FP-growth algorithm is described in the paper Han et al., Mining frequent patterns without candidate generation, where “FP” stands for frequent pattern. Given a dataset of transactions, the first step of FP-growth is to calculate item frequencies and identify frequent items. Different from Apriori-like algorithms designed for the same purpose, the second step of FP-growth uses a suffix tree (FP-tree) structure to encode transactions without generating candidate sets explicitly, which are usually expensive to generate. After the second step, the frequent itemsets can be extracted from the FP-tree. In MLlib, we implemented a parallel version of FP-growth called PFP, as described in Li et al., PFP: Parallel FP-growth for query recommendation. PFP distributes the work of growing FP-trees based on the suffices of transactions, and hence more scalable than a single-machine implementation. We refer users to the papers for more details.

FP-growth算法是有Han等人发表在论文上的，频率挖掘模式而不需要等待，'FP'代表了频率模式。假设有一个交易数据集，FP-growth的第一步就是计算项目频率和识别频率项目。不同于Apriori-like，算法的设计也是相同的目的，第二步就是使用FP-tree结构来编码交易数据，而不用显示的生成候选集，如果生成这些数据集，代价是很昂贵的。第二步过后，频率数据集能够从FP-tree中提取出来。在MLlib中，我们实施了一个并行的版本叫做PFP，PFP：并行FP-growth查询推荐。根据足够的交易量，PFP分发FP-trees增长的工作。因此性能更加好，比起单机运行。用户可以参考文献来获得更加详细的信息。

MLlib’s FP-growth implementation takes the following (hyper-)parameters:

* minSupport: the minimum support for an itemset to be identified as frequent. For example, if an item appears 3 out of 5 transactions, it has a support of 3/5=0.6.
* numPartitions: the number of partitions used to distribute the work.

FP-growth的实施需要下面的参数：
* minSupport:数据集被识别为频率的最小支持度
* numPartitions：运行分布式工作需要的分区数目

### Examples
### python

FPGrowth implements the FP-growth algorithm. It take an RDD of transactions, where each transaction is an List of items of a generic type. Calling FPGrowth.train with transactions returns an FPGrowthModel that stores the frequent itemsets with their frequencies.

FPGrowth实施了FP-growth算法。它采用了一个RDD数据交易集，每一个交易数据都是一个一个泛型类型的项目列表。调用交易数据的FPGrowth.train方法返回一个FPGrowthModel模型，这个模型存储了项目集。

	from pyspark.mllib.fpm import FPGrowth
	
	data = sc.textFile("data/mllib/sample_fpgrowth.txt")
	
	transactions = data.map(lambda line: line.strip().split(' '))
	
	model = FPGrowth.train(transactions, minSupport=0.2, numPartitions=10)
	
	result = model.freqItemsets().collect()
	for fi in result:
	    print(fi)
	    
## Association Rules

### Scala

AssociationRules implements a parallel rule generation algorithm for constructing rules that have a single item as the consequent.

关联规则实施了一个并行规则生成算法来构建一个规则，然后会生成一个项目。

	import org.apache.spark.rdd.RDD
	import org.apache.spark.mllib.fpm.AssociationRules
	import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset
	
	val freqItemsets = sc.parallelize(Seq(
	  new FreqItemset(Array("a"), 15L),
	  new FreqItemset(Array("b"), 35L),
	  new FreqItemset(Array("a", "b"), 12L)
	));
	
	val ar = new AssociationRules()
	  .setMinConfidence(0.8)
	val results = ar.run(freqItemsets)
	
	results.collect().foreach { rule =>
	  println("[" + rule.antecedent.mkString(",")
	    + "=>"
	    + rule.consequent.mkString(",") + "]," + rule.confidence)
	}
	
## PrefixSpan

PrefixSpan is a sequential pattern mining algorithm described in Pei et al., Mining Sequential Patterns by Pattern-Growth: The PrefixSpan Approach. We refer the reader to the referenced paper for formalizing the sequential pattern mining problem.

PrefixSpan是一个连续的模式挖掘算法。Pattern-Growth的连续挖掘模型：PrefixSpan方法。用户可以参考相关的文献来解决连续模型挖掘的问题。

MLlib’s PrefixSpan implementation takes the following parameters:

* minSupport: the minimum support required to be considered a frequent sequential pattern.
* maxPatternLength: the maximum length of a frequent sequential pattern. Any frequent pattern exceeding this length will not be included in the results.
* maxLocalProjDBSize: the maximum number of items allowed in a prefix-projected database before local iterative processing of the projected databse begins. This parameter should be tuned with respect to the size of your executors.

PrefixSpan算法的实施需要下面的参数：
* minsupport: 所需的最低支持被认为是频繁序列模式
* maxPatternLength:频繁序列模式的最大长度。任何频繁模式超过这个长度都不会包括进这个结果里面。
* maxLocalProjDBSize:所允许的最大项目数，在一个前缀映射数据库中和本地迭代处理投影数据库之前。考虑到执行节点的大小，这个参数将会做出一个相应的调整。

### Examples

The following example illustrates PrefixSpan running on the sequences (using same notation as Pei et al):

下面的例子演示了运行在序列中的 PrefixSpan算法。

	  <(12)3>
	  <1(32)(12)>
	  <(12)5>
	  <6>
	  
### Scala

PrefixSpan implements the PrefixSpan algorithm. Calling PrefixSpan.run returns a PrefixSpanModel that stores the frequent sequences with their frequencies.

PrefixSpan实施了PrefixSpan算法。调用PrefixSpan.run返回一个PrefixSpanModel模型，这存储了他们的频繁序列。

	import org.apache.spark.mllib.fpm.PrefixSpan
	
	val sequences = sc.parallelize(Seq(
	    Array(Array(1, 2), Array(3)),
	    Array(Array(1), Array(3, 2), Array(1, 2)),
	    Array(Array(1, 2), Array(5)),
	    Array(Array(6))
	  ), 2).cache()
	val prefixSpan = new PrefixSpan()
	  .setMinSupport(0.5)
	  .setMaxPatternLength(5)
	val model = prefixSpan.run(sequences)
	model.freqSequences.collect().foreach { freqSequence =>
	println(
	  freqSequence.sequence.map(_.mkString("[", ", ", "]")).mkString("[", ", ", "]") + ", " + freqSequence.freq)
	}