# Quick Start

* Interactive Analysis with the Spark Shell
 - Basics
 - More on RDD Operations
 - Caching
* Self-Contained Applications
* Where to Go from Here

This tutorial provides a quick introduction to using Spark. We will first introduce the API through Spark’s interactive shell (in Python or Scala), then show how to write applications in Java, Scala, and Python. See the programming guide for a more complete reference.

这篇指南对于使用spark提供了一个快速的介绍。首先我们会通过spark的交互式shell(python 或者 Scala)来介绍API。然后展示怎样在java 和 scala中写一个应用程序。阅读程序指南可以得到一个更全面的参考。

To follow along with this guide, first download a packaged release of Spark from the Spark website. Since we won’t be using HDFS, you can download a package for any version of Hadoop.

为了能够更好的学习这篇指南，首先得从spark的官网下载spark的安装包。因为我们不会使用hdfs，你可以下载Hadoop包的任何版本。

## Interactive Analysis with the Spark Shell
### Basics
Spark’s shell provides a simple way to learn the API, as well as a powerful tool to analyze data interactively. It is available in either Scala (which runs on the Java VM and is thus a good way to use existing Java libraries) or Python. Start it by running the following in the Spark directory:

Spark shell提供了一种很简单的方式来学习API，和一种非常强大的交互式工具来分析数据。可以在Scala或者python中使用。通过运行下面的在spark文件夹下的程序来启动它们。

__python__

    ./bin/pyspark
Spark’s primary abstraction is a distributed collection of items called a Resilient Distributed Dataset (RDD). RDDs can be created from Hadoop InputFormats (such as HDFS files) or by transforming other RDDs. Let’s make a new RDD from the text of the README file in the Spark source directory:

spark首要的抽象就是事件的分布式集合，称为弹性分布式数据集（RDD）。RDDs可以从Hadoop 输入格式中创建，如HDFS文件，或者通过其它RDDS来转化。在spark 源代码的文件夹下，让我们来从README文件来生成一个新的RDD。

    >>> textFile = sc.textFile("README.md")
RDDs have actions, which return values, and transformations, which return pointers to new RDDs. Let’s start with a few actions:

RDDS有两个行为，一个为行动，会返回一个值，另一个为转化，会生成一个指针，指向一个新的RDDs。让我们开始一个新的行动吧。

	>>> textFile.count() # Number of items in this RDD
	126
	
	>>> textFile.first() # First item in this RDD
	u'# Apache Spark'
Now let’s use a transformation. We will use the filter transformation to return a new RDD with a subset of the items in the file.

现在让我们来使用转化操作。我们将会使用一个过滤转化操作来生成一个新的RDD，在这个文件中，会生成一系列的分组事件。

    >>> linesWithSpark = textFile.filter(lambda line: "Spark" in line)
We can chain together transformations and actions:

我们可以把转化和行动结合在一起执行。

	    >>> textFile.filter(lambda line: "Spark" in line).count() # How many lines contain "Spark"?
	15

## More on RDD Operations
RDD actions and transformations can be used for more complex computations. Let’s say we want to find the line with the most words:

RDD行动和转化操作可以在更加复杂的计算中使用。比如说，我们想要找到一个单词量最多的一行。

__python__

	>>> textFile.map(lambda line: len(line.split())).reduce(lambda a, b: a if (a > b) else b)
	15
    
This first maps a line to an integer value, creating a new RDD. reduce is called on that RDD to find the largest line count. The arguments to map and reduce are Python anonymous functions (lambdas), but we can also pass any top-level Python function we want. For example, we’ll define a max function to make this code easier to understand:

首先把各行与一个整数值进行匹配，创建一个新的RDD。reduce操作是用来发现单词量最多的一行。用来map和reduce的参数都是隐式的函数表达式，比如lambdas，但是我们可以通过任何的top-level python函数来生成我们想要的函数。例如，我们可以定义一个找最大值的函数，这样可以让我们更容易理解。

	>>> def max(a, b):
	...     if a > b:
	...         return a
	...     else:
	...         return b
	...
	
	>>> textFile.map(lambda line: len(line.split())).reduce(max)
	15
	
One common data flow pattern is MapReduce, as popularized by Hadoop. Spark can implement MapReduce flows easily:

一个通用的数据流形式就是mapreduce，和Hadoop一样流行。spark可以在mapreduce下轻易的运行。

	>>> wordCounts = textFile.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)
	
Here, we combined the flatMap, map, and reduceByKey transformations to compute the per-word counts in the file as an RDD of (string, int) pairs. To collect the word counts in our shell, we can use the collect action:

现在我们来结合flatmap, map和reduceByKey转化操作来计算一个文件中的单词统计数。为了统计单词数，我们可以使用collect行动。

	>>> wordCounts.collect()
	[(u'and', 9), (u'A', 1), (u'webpage', 1), (u'README', 1), (u'Note', 1), (u'"local"', 1), (u'variable', 1), ...]

## Caching
Spark also supports pulling data sets into a cluster-wide in-memory cache. This is very useful when data is accessed repeatedly, such as when querying a small “hot” dataset or when running an iterative algorithm like PageRank. As a simple example, let’s mark our linesWithSpark dataset to be cached:

spark也支持把数据集pull到多节点的计算缓存中去。当数据需要反复使用时，这是非常有效的。例如，当我们查询一个比较小且经常使用的数据集时，或者运行一个可迭代的算法比如pagerank时。下面是一个简单的例子，让我们来标记我们的 lineswithspark 数据集到缓存中吧。

__python__

	>>> linesWithSpark.cache()
	
	>>> linesWithSpark.count()
	19
	
	>>> linesWithSpark.count()
	19

It may seem silly to use Spark to explore and cache a 100-line text file. The interesting part is that these same functions can be used on very large data sets, even when they are striped across tens or hundreds of nodes. You can also do this interactively by connecting bin/pyspark to a cluster, as described in the programming guide.

当我们使用spark来探索或者缓存100行的text文件时，这看上去是一个比较愚蠢的行为。比较有趣的是我们可以使用相同的方法来运行一个非常大的数据集，甚至当它们被分配到数十个甚至上百个节点中时。你也可以通过交互式的`bin/pyspark`连接到集群，来运行它们，正如在programming guide中描述的一样。


## Self-Contained Applications

Suppose we wish to write a self-contained application using the Spark API. We will walk through a simple application in Scala (with sbt), Java (with Maven), and Python.

假设我们想使用spark API来写一个自我包含的应用，我们可以walk through 一个简单的应用在scala, java 或者 python 上。

__python__

Now we will show how to write an application using the Python API (PySpark).

现在我们可以展示怎样通过python API 来写一个应用了。

As an example, we’ll create a simple Spark application, SimpleApp.py:

作为一个例子，我们会创建一个简单的spark应用。文件名是SimpleApp.py。

	"""SimpleApp.py"""
	from pyspark import SparkContext
	
	logFile = "YOUR_SPARK_HOME/README.md"  # Should be some file on your system
	sc = SparkContext("local", "Simple App")
	logData = sc.textFile(logFile).cache()
	
	numAs = logData.filter(lambda s: 'a' in s).count()
	numBs = logData.filter(lambda s: 'b' in s).count()
	
	print("Lines with a: %i, lines with b: %i" % (numAs, numBs))
	
This program just counts the number of lines containing ‘a’ and the number containing ‘b’ in a text file. Note that you’ll need to replace YOUR_SPARK_HOME with the location where Spark is installed. As with the Scala and Java examples, we use a SparkContext to create RDDs. We can pass Python functions to Spark, which are automatically serialized along with any variables that they reference. For applications that use custom classes or third-party libraries, we can also add code dependencies to spark-submit through its --py-files argument by packaging them into a .zip file (see spark-submit --help for details). SimpleApp is simple enough that we do not need to specify any code dependencies.

这个程序只是统计一个文件中包含“a”的行的数目和包含“b”的行的数目。注意，当运行spark时，你需要用自己的路径来替代YOUR_SPARK_HOME。同样的在Scala和java中，我们使用SparkContext来创建一个RDDs。我们可以把python函数传递到spark中去。这将自动把各种变量连接起来。对于那些自己设置类或者运用第三方库的应用来说，我们可以添加代码到spark-submit 中，通过打包为.zip的文件。simpleapp是足够简单的以至于我们不需要指定任何代码依赖。

We can run this application using the bin/spark-submit script:

我们用`bin/spark-submit`脚本来运行这个程序。


	# Use spark-submit to run your application
	$ YOUR_SPARK_HOME/bin/spark-submit \
	  --master local[4] \
	  SimpleApp.py
	...
	Lines with a: 46, Lines with b: 23
	
## Where to Go from Here
Congratulations on running your first Spark application!

* For an in-depth overview of the API, start with the Spark programming guide, or see “Programming Guides” menu for other components.
* For running applications on a cluster, head to the deployment overview.
* Finally, Spark includes several samples in the examples directory (Scala, Java, Python, R). You can run them as follows:

恭喜你运行了第一个spark应用！

* 对于一个有深度的关于API的综述，开始spark 程序指南，或者点击 “Programming Guides”来看其它的部分。
* 如果想在集群上运行，可以看deployment overview部分。
* 最后，spark包含了若干个例子在examples文件夹中。你可以用如下的方式，运行它们。

		# For Scala and Java, use run-example:
		./bin/run-example SparkPi
		
		# For Python examples, use spark-submit directly:
		./bin/spark-submit examples/src/main/python/pi.py
		
		# For R examples, use spark-submit directly:
		./bin/spark-submit examples/src/main/r/dataframe.R



