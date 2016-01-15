 
# Spark Programming Guide

## Overview

At a high level, every Spark application consists of a driver program that runs the user’s main function and executes various parallel operations on a cluster. The main abstraction Spark provides is a resilient distributed dataset (RDD), which is a collection of elements partitioned across the nodes of the cluster that can be operated on in parallel. RDDs are created by starting with a file in the Hadoop file system (or any other Hadoop-supported file system), or an existing Scala collection in the driver program, and transforming it. Users may also ask Spark to persist an RDD in memory, allowing it to be reused efficiently across parallel operations. Finally, RDDs automatically recover from node failures.

在高的阶段，每个spark应用包含一个驱动程序，它在一个集群上执行用户的主程序和执行许多并行操作。Spark提出的主要抽象概念是一个弹性分布式数据集（RDD），它是一个元素的集合，这些元素可以被并行操作，且来自于集群的各个节点。RDD由Hadoop文件系统的一个文件或者驱动程序中存在的Scala集合来启动，并且转移它。用户可以要求Spark保留RDD在内存中，允许它们在并行操作中被有效的再利用。

A second abstraction in Spark is shared variables that can be used in parallel operations. By default, when Spark runs a function in parallel as a set of tasks on different nodes, it ships a copy of each variable used in the function to each task. Sometimes, a variable needs to be shared across tasks, or between tasks and the driver program. Spark supports two types of shared variables: broadcast variables, which can be used to cache a value in memory on all nodes, and accumulators, which are variables that are only “added” to, such as counters and sums.Spark的第二个抽象是共享变量，它能被用在并行操作中。默认的，当Spark执行一个函数，该函数是在不同节点上的一系列任务，它传输函数中用到的所有变量的一个副本给每个任务。有时，一个变量需要在任务之间进行共享，或者在任务和驱动程序之间。Spark支持两类共享变量：传播变量，它可以用来推送一个内存中的值到所有节点；累计变量，它是增加的变量，比如计数器和累加器。

This guide shows each of these features in each of Spark’s supported languages. It is easiest to follow along with if you launch Spark’s interactive shell – either bin/spark-shell for the Scala shell or bin/pyspark for the Python one.这个指南展示了每个Spark的特性，它非常容易学会，如果你下载了Spark的互动shell。

## Linking with Spark

###  python

Spark 1.5.2 works with Python 2.6+ or Python 3.4+. It can use the standard CPython interpreter, so C libraries like NumPy can be used. It also works with PyPy 2.3+.

Spark1.5.2 可以运行在python 2.6及以后的版本或者python 3.4及以后的版本。它也可以用标准CPython编译器来执行，所以像用C语言写的库，比如numpy，也可以使用。spark 1.5.2 也可以运行在PyPy 2.3及以后的版本中。

To run Spark applications in Python, use the `bin/spark-submit` script located in the Spark directory. This script will load Spark’s Java/Scala libraries and allow you to submit applications to a cluster. You can also use bin/pyspark to launch an interactive Python shell.

为了用Python执行Spark应用，可以用Spark目录里面的`bin/spark-submit`脚本。这个脚本会加载Spark的Java库并且允许你向集群提交一个应用。你也可以用bin/pyspark来使用一个交互的.

If you wish to access HDFS data, you need to use a build of PySpark linking to your version of HDFS. Some common HDFS version tags are listed on the third party distributions page. Prebuilt packages are also available on the Spark homepage for common HDFS versions.

如果你希望链接到HDFS数据，你需要使用一个编译好的Spark包对应着HDFS版本。一些普通的HDFS版本列在第三部分。预编译的包也提供在Spark的主页上。

Finally, you need to import some Spark classes into your program. Add the following line:
最后，你需要在程序里引用一些Spark类，插入下面的内容： 

    from pyspark import SparkContext, SparkConf
    
PySpark requires the same minor version of Python in both driver and workers. It uses the default python version in PATH, you can specify which version of Python you want to use by PYSPARK_PYTHON, for example:
    
不管是在驱动节点还是工作节点中，pyspark需要相同的python版本。它们在path路径下使用默认的python版本， 你也可以指定你想用的版本，通过PYSPARK_PYTHON，例如：

    $ PYSPARK_PYTHON=python3.4 bin/pyspark
    $ PYSPARK_PYTHON=/opt/pypy-2.5/bin/pypy bin/spark-submit examples/src/main/python/pi.py## Initializing Spark
###  python
The first thing a Spark program must do is to create a SparkContext object, which tells Spark how to access a cluster. To create a SparkContext you first need to build a SparkConf object that contains information about your application.

Spark程序必须做的最重要的事情就是创建一个SparkContext类，它告诉Spark如何连接一个集群。为了创建一个SaprkContext，首先需要创建一个SparkConf类，它包含应用程序的信息。
    conf = SparkConf().setAppName(appName).setMaster(master)
    sc = SparkContext(conf=conf)
    
The appName parameter is a name for your application to show on the cluster UI. master is a Spark, Mesos or YARN cluster URL, or a special “local” string to run in local mode. In practice, when running on a cluster, you will not want to hardcode master in the program, but rather launch the application with spark-submit and receive it there. However, for local testing and unit tests, you can pass “local” to run Spark in-process.
 
appName参数是你的应用程序显示的名字。Master是Spark集群的地址或者本地模式下的local字符串。在实践中，当在集群中运行时，我们不想在程序中硬编码master，而是用spark-submit来发起程序，并且在那里接收它。然而，为了本地测试和单元测试，你可以利用local字符串来执行线性Spark

## Using the Shell
### python
In the PySpark shell, a special interpreter-aware SparkContext is already created for you, in the variable called sc. Making your own SparkContext will not work. You can set which master the context connects to using the --master argument, and you can add Python .zip, .egg or .py files to the runtime path by passing a comma-separated list to --py-files. You can also add dependencies (e.g. Spark Packages) to your shell session by supplying a comma-separated list of maven coordinates to the --packages argument. Any additional repositories where dependencies might exist (e.g. SonaType) can be passed to the --repositories argument. Any python dependencies a Spark Package has (listed in the requirements.txt of that package) must be manually installed using pip when necessary. For example, to run bin/pyspark on exactly four cores, use:

在PySpark脚本中，一个特殊的解释感知SparkContext已经创建了，在一个叫sc的包里面。使你自己的SparkContext不能工作。你可以设置哪个主机是context需要连接的，利用master参数；你可以增加Python文件在执行路径中，利用一个逗号分隔的列表。你还可以增加依赖关系给脚本，通过提供一个逗号分隔的列表，通过packages参数。所有附加存储库都可以通过repositories参数进行传递。依赖于一个Spark包的任何python都被人工的安装了，利用pip。例如，用四个核运行bin/pyspark，使用：

    $ ./bin/pyspark --master local[4]
    
Or, to also add code.py to the search path (in order to later be able to import code), use:

或者添加code.py来寻找路径，使用：

    $ ./bin/pyspark --master local[4] --py-files code.py
    
For a complete list of options, run pyspark --help. Behind the scenes, pyspark invokes the more general spark-submit script.

要想知道一个完整的选择列表，运行pyspark --help命令。在后台，pyspark调用更一般的spark-submit脚本。

It is also possible to launch the PySpark shell in IPython, the enhanced Python interpreter. PySpark works with IPython 1.0.0 and later. To use IPython, set the PYSPARK_DRIVER_PYTHON variable to ipython when running bin/pyspark:

也可以把 pyspark shell 加载在 ipython 中，这是一个提高版的python解释器。pyspark 运行在 IPython 1.0.0 或者更高的版本。使用 Ipython，运行 running bin/pyspark时，环境变量PYSPARK_DRIVER_PYTHON指向ipython。

    $ PYSPARK_DRIVER_PYTHON=ipython ./bin/pyspark
    
You can customize the ipython command by setting PYSPARK_DRIVER_PYTHON_OPTS. For example, to launch the IPython Notebook with PyLab plot support:

你也可以自己定义ipython的路径。例如，为了加载 IPython Notebook，可以设置：

    $ PYSPARK_DRIVER_PYTHON=ipython PYSPARK_DRIVER_PYTHON_OPTS="notebook" ./bin/pyspark
    
After the IPython Notebook server is launched, you can create a new “Python 2” notebook from the “Files” tab. Inside the notebook, you can input the command %pylab inline as part of your notebook before you start to try Spark from the IPython notebook.

在IPython记事本服务发起之后，你可以发起一个新的Python2记事本。在记事本里面，你可以输入命令pylab inline 就像记事本的一部分一样，在你开始尝试Spark之前。

## Resilient Distributed Datasets (RDDs)

Spark revolves around the concept of a resilient distributed dataset (RDD), which is a fault-tolerant collection of elements that can be operated on in parallel. There are two ways to create RDDs: parallelizing an existing collection in your driver program, or referencing a dataset in an external storage system, such as a shared filesystem, HDFS, HBase, or any data source offering a Hadoop InputFormat.

 Spark提出弹性分布式数据集的概念，它是一个可以容忍错误的集合，而且可以被并行操作。有两种方法来创建RDDs，并行化一个驱动程序里已经存在的数据集合，或者引用一个外部存储系统（比如共享文件，HDFS，HBase或其他Hadoop输入来的数据）的数据集。

## Parallelized Collections
### python
Parallelized collections are created by calling SparkContext’s parallelize method on an existing iterable or collection in your driver program. The elements of the collection are copied to form a distributed dataset that can be operated on in parallel. For example, here is how to create a parallelized collection holding the numbers 1 to 5:

并行化集合由呼叫SparkContext的parallelize方法创建的，该方法执行与驱动程序中的存在的迭代和集合。并行化集合由呼叫SparkContext的parallelize方法创建的，该方法执行与驱动程序中的存在的迭代和集合。

    data = [1, 2, 3, 4, 5]
    distData = sc.parallelize(data)Once created, the distributed dataset (distData) can be operated on in parallel. For example, we can call distData.reduce(lambda a, b: a + b) to add up the elements of the list. We describe operations on distributed datasets later on.

只要创建成功，分布式数据集distData可以被并行操作。比如，我们可以调用distData.reduce(lambda a,b:a+b)来对列表中的元素求和。One important parameter for parallel collections is the number of partitions to cut the dataset into. Spark will run one task for each partition of the cluster. Typically you want 2-4 partitions for each CPU in your cluster. Normally, Spark tries to set the number of partitions automatically based on your cluster. However, you can also set it manually by passing it as a second parameter to parallelize (e.g. sc.parallelize(data, 10)). Note: some places in the code use the term slices (a synonym for partitions) to maintain backward compatibility.

并行集合的一个重要的参数就是分区个数。Spark会对集群上的每个分区执行一个任务。你想一个CPU执行4个数据块。正常情况下，Spark尝试基于集群来自动设置分区个数。 然而，你可以可以手动设置它。提示：代码的一些地方用片的形式来保持向后兼容性。

## External Datasets
### python
PySpark can create distributed datasets from any storage source supported by Hadoop, including your local file system, HDFS, Cassandra, HBase, Amazon S3, etc. Spark supports text files, SequenceFiles, and any other Hadoop InputFormat.

PySpark可以从Hadoop支持的任何存储资源创建分布式数据集，包括本地文件系统，HDFS，Cassandra，HBase和亚马逊S3.Spark支持文本文件，二进制文件和其他Hadoop输入格式的文件.

Text file RDDs can be created using SparkContext’s textFile method. This method takes an URI for the file (either a local path on the machine, or a hdfs://, s3n://, etc URI) and reads it as a collection of lines. Here is an example invocation:

RDDs文件可以用SparkContext的textFile方法来创建。这个方法用URI当做文件，并且将它当做行的集合来读取。这里有一个调用的例子。

    >>> distFile = sc.textFile("data.txt")
    
Once created, distFile can be acted on by dataset operations. For example, we can add up the sizes of all the lines using the map and reduce operations as follows: distFile.map(lambda s: len(s)).reduce(lambda a, b: a + b).

一旦创建完成distFile就在数据集操作中发挥作用。比如，我们可以增加求和所有的行，利用map和reduce操作如下：distfile.map(lambda s: len(s)).reduce(lambda a,b :a+b).

Some notes on reading files with Spark:

* If using a path on the local filesystem, the file must also be accessible at the same path on worker nodes. Either copy the file to all workers or use a network-mounted shared file system.
* All of Spark’s file-based input methods, including textFile, support running on directories, compressed files, and wildcards as well. For example, you can use textFile("/my/directory"), textFile("/my/directory/\*.txt"), and textFile("/my/directory/*.gz").
* The textFile method also takes an optional second argument for controlling the number of partitions of the file. By default, Spark creates one partition for each block of the file (blocks being 64MB by default in HDFS), but you can also ask for a higher number of partitions by passing a larger value. Note that you cannot have fewer partitions than blocks.

一些阅读Spark文件的提示:

* 如果使用本地文件系统的一个路径，这个文件的相同路径必须在工作节点能连接到。可以拷贝文件到所有节点，或者使用网络共享的文件系统。
* 所有的Spark的基于文件输入的方法，包括textFile，都支持目录，压缩文件和通配符。例如，你可以用： textFile("/my/directory"), textFile("/my/directory/\*.txt"), and textFile("/my/directory/\*.gz")
* textFile方法还提供一个控制文件块数量的非强制的第二参数。默认的，Spark创建一个分块对应一个文件，但是也可以要求更多数量的分区，通过传递一个更大的值。注意，你不能得到比数据块个数更少的分区。Apart from text files, Spark’s Python API also supports several other data formats:

* SparkContext.wholeTextFiles lets you read a directory containing multiple small text files, and returns each of them as (filename, content) pairs. This is in contrast with textFile, which would return one record per line in each file.
* RDD.saveAsPickleFile and SparkContext.pickleFile support saving an RDD in a simple format consisting of pickled Python objects. Batching is used on pickle serialization, with default batch size 10.
* SequenceFile and Hadoop Input/Output Formats

除了文本文件，Spark的Python API还支持其它若干种数据格式:

* SparkContext.wholeTextFiles让你读取一个包含众多小文件的目录，并且返回（文件名，内容）对。这个对应于textFile，它可以返回一个文件的每行。
* RDD.saveAsPickleFile和SparkContext.pickleFile支持存储RDD进入一个由Python的基本类型组成的简单格式中。
* 二进制文件和Hadoop的输入输出文件

__Note__ this feature is currently marked Experimental and is intended for advanced users. It may be replaced in future with read/write support based on Spark SQL, in which case Spark SQL is the preferred approach.**注意** 这个特性也许会被基于spark SQL的读写支持在未来取代，在那时spark SQL 是优先考虑的方法。

**Writable Support**

PySpark SequenceFile support loads an RDD of key-value pairs within Java, converts Writables to base Java types, and pickles the resulting Java objects using Pyrolite. When saving an RDD of key-value pairs to SequenceFile, PySpark does the reverse. It unpickles Python objects into Java objects and then converts them to Writables. The following Writables are automatically converted:

PySpark SequenceFile 支持用Java加载RDD键值对，转换Writeables类型去Java基本类型，然后用Pyrolite来转换Java结果。当保存一个RDD键值对到二进制文件时，PySpark负责转换。它转换Python类型为Java类型，并且转换他们为Writables类型。以下可写的类型将自动转换：

<table class="table"><tbody><tr><th>Writable Type</th><th>Python Type</th></tr><tr><td>Text</td><td>unicode str</td></tr><tr><td>IntWritable</td><td>int</td></tr><tr><td>FloatWritable</td><td>float</td></tr><tr><td>DoubleWritable</td><td>float</td></tr><tr><td>BooleanWritable</td><td>bool</td></tr><tr><td>BytesWritable</td><td>bytearray</td></tr><tr><td>NullWritable</td><td>None</td></tr><tr><td>MapWritable</td><td>dict</td></tr></tbody></table>

Arrays are not handled out-of-the-box. Users need to specify custom ArrayWritable subtypes when reading or writing. When writing, users also need to specify custom converters that convert arrays to custom ArrayWritable subtypes. When reading, the default converter will convert custom ArrayWritable subtypes to Java Object[], which then get pickled to Python tuples. To get Python array.array for arrays of primitive types, users need to specify custom converters.

数组并没有自动转换。用户需要指定自定义的ArrayWritable亚型，在读和写时。当写的时候，用户需要指定自定义转换，它转换数组为自定义的ArrayWritable亚型。当读的时候，默认转换器会将自定义ArrayWritable亚型转换为Java数组，它又被转换为Python的元组。为了获得Python的数组的原始类型，用户需要指定自定义转换器。__Saving and Loading SequenceFiles__

__保存和加载二进制文件__

Similarly to text files, SequenceFiles can be saved and loaded by specifying the path. The key and value classes can be specified, but for standard Writables this is not required.

和文本文件相似，二进制文件可以靠指定路径来保存和加载。键和值可以指定，但是在标准Writabes中，这些是不必须的。

    >>> rdd = sc.parallelize(range(1, 4)).map(lambda x: (x, "a" * x ))
    >>> rdd.saveAsSequenceFile("path/to/file")
    >>> sorted(sc.sequenceFile("path/to/file").collect())
    [(1, u'a'), (2, u'aa'), (3, u'aaa')]
    
__Saving and Loading Other Hadoop Input/Output Formats__

__保存和加载其它Hadoop 输入输出__

PySpark can also read any Hadoop InputFormat or write any Hadoop OutputFormat, for both ‘new’ and ‘old’ Hadoop MapReduce APIs. If required, a Hadoop configuration can be passed in as a Python dict. Here is an example using the Elasticsearch ESInputFormat:

 PySpark 可以读任何Hadoop输入格式或写任何Hadoop输出格式，无论新旧Hadoop的API。如果需要，一个Hadoop的配置文件可以作为Python的dict传输。
 
     $ SPARK_CLASSPATH=/path/to/elasticsearch-hadoop.jar ./bin/pyspark
    >>> conf = {"es.resource" : "index/type"}   # assume Elasticsearch is running on localhost defaults
    >>> rdd = sc.newAPIHadoopRDD("org.elasticsearch.hadoop.mr.EsInputFormat",\
    "org.apache.hadoop.io.NullWritable", "org.elasticsearch.hadoop.mr.LinkedMapWritable", conf=conf)
    >>> rdd.first()         # the result is a MapWritable that is converted to a Python dict
    (u'Elasticsearch ID',
     {u'field1': True,
      u'field2': u'Some Text',
      u'field3': 12345})
      
Note that, if the InputFormat simply depends on a Hadoop configuration and/or input path, and the key and value classes can easily be converted according to the above table, then this approach should work well for such cases.

注意，如果输入格式简单的依赖于Hadoop配置文件和输入文件路径，并且键和值类能简单的转换，那么这些程序应该工作顺利。

If you have custom serialized binary data (such as loading data from Cassandra / HBase), then you will first need to transform that data on the Scala/Java side to something which can be handled by Pyrolite’s pickler. A Converter trait is provided for this. Simply extend this trait and implement your transformation code in the convert method. Remember to ensure that this class, along with any dependencies required to access your InputFormat, are packaged into your Spark job jar and included on the PySpark classpath.

如果你已经序列化二进制文件，那么你首先需要转化Java数据到Pyrolite可以处理的类型。一个转换器特征提供此类功能。简单的扩展这个特性，并且在转换方法中实现转换代码。记住确保这些类被打包进了你的Spark工作包，并且包含PySpark类的路径。

See the Python examples and the Converter examples for examples of using Cassandra / HBase InputFormat and OutputFormat with custom converters.

可以参考Python examples包作为例子。


## RDD Operations

RDDs support two types of operations: transformations, which create a new dataset from an existing one, and actions, which return a value to the driver program after running a computation on the dataset. For example, map is a transformation that passes each dataset element through a function and returns a new RDD representing the results. On the other hand, reduce is an action that aggregates all the elements of the RDD using some function and returns the final result to the driver program (although there is also a parallel reduceByKey that returns a distributed dataset).

RDD支持两类操作：转换，它可以从存在的数据集中产生新的数据集，还有行动，它可以在习性一个数据集之后返回给驱动程序一个值。比如，map是一个转换，它传递每个数据集的元素到一个函数，并且返回一个新的RDD代表的值。另一方面，reduce是一个行动，它用函数聚合了RDD的所有元素并且返回最终结果给驱动程序（还有一个并行的行动叫reduceByKey，它返回一个分布式的数据集）。

All transformations in Spark are lazy, in that they do not compute their results right away. Instead, they just remember the transformations applied to some base dataset (e.g. a file). The transformations are only computed when an action requires a result to be returned to the driver program. This design enables Spark to run more efficiently – for example, we can realize that a dataset created through map will be used in a reduce and return only the result of the reduce to the driver, rather than the larger mapped dataset.
所有的转换都是懒的，他们不会立刻计算结果。它们只是记忆一些申请基本数据的转换。转换只会在一个行动需要一个结果被输出到驱动程序中时。这个设计让Spark可以执行的更有效率，举个例子，我们能意识到一个由map创造的数据集会被用在一个reduce中而且只会返回reduce的结果，而不用返回更大的map过的数据集。By default, each transformed RDD may be recomputed each time you run an action on it. However, you may also persist an RDD in memory using the persist (or cache) method, in which case Spark will keep the elements around on the cluster for much faster access the next time you query it. There is also support for persisting RDDs on disk, or replicated across multiple nodes.
默认的，每个转换RDD可能重新计算，只要你在其上执行一个行动。然而，你可以利用persist方法来坚持让RDD在内存中，使用这个方法，Spark可以保证下次需要该元素时，可以快速的获取到。坚持RDD也支持磁盘，或者多节点复制。

## Basics
### python
To illustrate RDD basics, consider the simple program below:

    lines = sc.textFile("data.txt")
    lineLengths = lines.map(lambda s: len(s))
    totalLength = lineLengths.reduce(lambda a, b: a + b)
The first line defines a base RDD from an external file. This dataset is not loaded in memory or otherwise acted on: lines is merely a pointer to the file. The second line defines lineLengths as the result of a map transformation. Again, lineLengths is not immediately computed, due to laziness. Finally, we run reduce, which is an action. At this point Spark breaks the computation into tasks to run on separate machines, and each machine runs both its part of the map and a local reduction, returning only its answer to the driver program.

第一行从外部文件定义了一个RDD。这个数据集没有加载到内存，或者可以这样说：行仅仅是文件的一个指针。第二行代码定义了了lineLenths这个变量作为map转化的一个结果。 重申一下，lineLengths没有立刻计算。最后，我们执行reduce，它是一个动作。在这点上，Spark分解计算任务，让他们在不同的机器上执行，并且每个机器执行一部分map和一个本地reducetion，返回reduce的结果给驱动程序。If we also wanted to use lineLengths again later, we could add:
    
    lineLengths.persist()
before the reduce, which would cause lineLengths to be saved in memory after the first time it is computed.

如果我们想以后再用lineLengths，我们可以在reduce之前增加一句代码，它可以让lineLengths被长久的存储起来。

## Passing Functions to Spark
### python
Spark’s API relies heavily on passing functions in the driver program to run on the cluster. There are three recommended ways to do this:

* Lambda expressions, for simple functions that can be written as an expression. (Lambdas do not support multi-statement functions or statements that do not return a value.)
* Local defs inside the function calling into Spark, for longer code.
* Top-level functions in a module.

Spark的接口依赖于传递驱动程序中的函数去集群中执行。

* Lambda表达式，就是简单的函数可以被写成一个表达式。（Lambdas不支持多语句的函数，不支持没有返回值语句）
* 本地的在函数中的def可以调用长代码。
* 模块的顶层函数。
    
For example, to pass a longer function than can be supported using a lambda, consider the code below:

举个例子，为了传递一个不能用lambda表示的长函数，考虑下面的代码。

	"""MyScript.py"""
	if __name__ == "__main__":
	    def myFunc(s):
	        words = s.split(" ")
	        return len(words)
	
	    sc = SparkContext(...)
	    sc.textFile("file.txt").map(myFunc)
	    
Note that while it is also possible to pass a reference to a method in a class instance (as opposed to a singleton object), this requires sending the object that contains that class along with the method. 

注意到我们可以在类实例中传递一个引用到一个方法中（而不是一个单独的对象），这个需要发送对象，它包含那个类。

For example, consider:

	class MyClass(object):
	    def func(self, s):
	        return s
	    def doStuff(self, rdd):
	        return rdd.map(self.func)
Here, if we create a new MyClass and call doStuff on it, the map inside there references the func method of that MyClass instance, so the whole object needs to be sent to the cluster.

在这里，如果我们创建一个新的MyClass类，并且调用doStuff在它上，里面的map引用了MyClass实例的func方法。

In a similar way, accessing fields of the outer object will reference the whole object:

在这里，如果我们创建一个新的MyClass类，并且调用doStuff在它上，里面的map引用了MyClass实例的func方法。

	class MyClass(object):
	    def __init__(self):
	        self.field = "Hello"
	    def doStuff(self, rdd):
	        return rdd.map(lambda s: self.field + s)
	        
To avoid this issue, the simplest way is to copy field into a local variable instead of accessing it externally:

为了避免这个问题，最简单的办法是拷贝属性到一个本地变量，而不是从外部访问它。

	def doStuff(self, rdd):
	    field = self.field
	    return rdd.map(lambda s: field + s)
	    
## Understanding closures
One of the harder things about Spark is understanding the scope and life cycle of variables and methods when executing code across a cluster. RDD operations that modify variables outside of their scope can be a frequent source of confusion. In the example below we’ll look at code that uses foreach() to increment a counter, but similar issues can occur for other operations as well.

Spark的一个难点是理解变量和方法的范围和生命周期，当在集群中执行代码时。在变量范围之外修改它的值的RDD操作时频繁存在的混乱。在下面例子中，我们可以看到代码用foreach()来新增加一个计数器，相同的问题发生在其他的操作中。

### Example
Consider the naive RDD element sum below, which behaves completely differently depending on whether execution is happening within the same JVM. A common example of this is when running Spark in local mode (--master = local[n]) versus deploying a Spark application to a cluster (e.g. via spark-submit to YARN):

考虑到简单的RDD元素相加，它表现的完全不同在相同的JVM中，根据操作是否发生。一个普通的例子是在本地模式执行Spark，与在集群上部署Spark应用。

### python

	counter = 0
	rdd = sc.parallelize(data)
	
	# Wrong: Don't do this!!
	rdd.foreach(lambda x: counter += x)
	
	print("Counter value: " + counter)
	
### Local vs. cluster modes

The primary challenge is that the behavior of the above code is undefined. In local mode with a single JVM, the above code will sum the values within the RDD and store it in counter. This is because both the RDD and the variable counter are in the same memory space on the driver node.

主要的挑战是上面代码的执行方式没有定义。在本地模式下的单独JVM中，上面的代码会求和RDD的值，并且存储在计数器中。这是因为RDD和变量计数器都在同一个驱动节点的内存空间上。

However, in cluster mode, what happens is more complicated, and the above may not work as intended. To execute jobs, Spark breaks up the processing of RDD operations into tasks - each of which is operated on by an executor. Prior to execution, Spark computes the closure. The closure is those variables and methods which must be visible for the executor to perform its computations on the RDD (in this case foreach()). This closure is serialized and sent to each executor. In local mode, there is only the one executors so everything shares the same closure. In other modes however, this is not the case and the executors running on seperate worker nodes each have their own copy of the closure.

然而，在集群模式下，执行情况更加复杂，可能不会和预期一样。为了执行作业，Spark分解RDD操作的进程为多个任务，每个都被一个执行单元执行。执行之前，Spark 计算边界。边界是那么变量和方法必须有效的对执行器在RDD上进行计算。这个边界被序列化，并且送到每个执行器。在本地模式下，之类只有一个执行器，所以所有变量共享一个边界。在其它模式下，每个执行器运行在不同的节点上，每个都有自己的边界备份。

What is happening here is that the variables within the closure sent to each executor are now copies and thus, when counter is referenced within the foreach function, it’s no longer the counter on the driver node. There is still a counter in the memory of the driver node but this is no longer visible to the executors! The executors only sees the copy from the serialized closure. Thus, the final value of counter will still be zero since all operations on counter were referencing the value within the serialized closure.

这里发生的事情是这样的：边界范围内的变量被发送给执行器，从而当计数器在函数范围内被引用是，它就不再是驱动节点上的计数器了。驱动节点中仍然有一个计数器，但是它对执行器不可见。执行器智能看到序列化的边界的拷贝。因此，计数器的最终值依然会是零，自从计数器上的所有操作被引用在序列化边界之内。To ensure well-defined behavior in these sorts of scenarios one should use an Accumulator. Accumulators in Spark are used specifically to provide a mechanism for safely updating a variable when execution is split up across worker nodes in a cluster. The Accumulators section of this guide discusses these in more detail.

为了确保方案中的完全定义的行为，他应该利用累计器Spark的累加器被用来给变量提供安全更新机制，当执行被分割到多个节点时。累计器章节会详细介绍。

In general, closures - constructs like loops or locally defined methods, should not be used to mutate some global state. Spark does not define or guarantee the behavior of mutations to objects referenced from outside of closures. Some code that does this may work in local mode, but that’s just by accident and such code will not behave as expected in distributed mode. Use an Accumulator instead if some global aggregation is needed.

大体上，边界-循环或者本地定义方法的约束，不应该用来改变全局变量的状态。Spark不定义不保证从外界引用的对象的修改行为。一些代码这样做在本地模式可能有效，但是那只是偶然并且这个代码不会有效的使用在分布式模式下。使用累计器代替，如果一些全局聚合需要。

#### Printing elements of an RDD
Another common idiom is attempting to print out the elements of an RDD using rdd.foreach(println) or rdd.map(println). On a single machine, this will generate the expected output and print all the RDD’s elements. However, in cluster mode, the output to stdout being called by the executors is now writing to the executor’s stdout instead, not the one on the driver, so stdout on the driver won’t show these! To print all elements on the driver, one can use the collect() method to first bring the RDD to the driver node thus: rdd.collect().foreach(println). This can cause the driver to run out of memory, though, because collect() fetches the entire RDD to a single machine; if you only need to print a few elements of the RDD, a safer approach is to use the take(): rdd.take(100).foreach(println).

另一个常用的语法是尝试打印RDD的元素，利用rdd.foreach(println) 和 rdd.map(println)。在单独机器上，这样做会成功。然而，在集群模式下， 被执行器叫做stdout的输出地址是执行器的节点的stdout，而不是驱动程序所在节点，所以驱动程序的stdout不会显示结果。为了在驱动程序节点上打印元素，应该用collect()函数首先把RDD带到驱动程序节点，就像这样：rdd.collect().foreach(println)。这个可能造成驱动程序耗尽内存，因为collect()函数把整个RDD搬进一个机器；如果你只需要打印RDD的少数元素，一个安全的获取方式是用 take():rdd.take(100).foreach(println)函数。
## Working with Key-Value Pairs
### python

While most Spark operations work on RDDs containing any type of objects, a few special operations are only available on RDDs of key-value pairs. The most common ones are distributed “shuffle” operations, such as grouping or aggregating the elements by a key.

尽管大多数Spark操作包含了各种对象类型，少数几个特殊的操作只能用RDD的kv对。最寻常的一个就是在分布式的混洗操作，就像利用key来分组或者聚合元素。

In Python, these operations work on RDDs containing built-in Python tuples such as (1, 2). Simply create such tuples and then call your desired operation.在Python中，这些操作利用Python数组。简单的创造这样的数组，然后调用需要的操作。

For example, the following code uses the reduceByKey operation on key-value pairs to count how many times each line of text occurs in a file:

举个例子，下面的代码利用reduceByKey操作在kv对上来统计每行出现在一个文件中的次数。

	lines = sc.textFile("data.txt")
	pairs = lines.map(lambda s: (s, 1))
	counts = pairs.reduceByKey(lambda a, b: a + b)
	
We could also use counts.sortByKey(), for example, to sort the pairs alphabetically, and finally counts.collect() to bring them back to the driver program as a list of objects.

我们应该用counts.sortByKey()，来计算kd对按照字母表的顺序，并且最后counts.collect()来讲它们带回到驱动程序。

## Transformations

The following table lists some of the common transformations supported by Spark. Refer to the RDD API doc (Scala, Java, Python, R) and pair RDD functions doc (Scala, Java) for details.
下面的表格列出了一些普通的转化，选择RDD 接口来查看详情。
<table class="table">
<tbody><tr><th style="width:25%">Transformation</th><th>Meaning</th></tr>
<tr>
  <td> <b>map</b>(<i>func</i>) </td>
  <td> Return a new distributed dataset formed by passing each element of the source through a function <i>func</i>. </td>
</tr>
<tr>
  <td> <b>filter</b>(<i>func</i>) </td>
  <td> Return a new dataset formed by selecting those elements of the source on which <i>func</i> returns true. </td>
</tr>
<tr>
  <td> <b>flatMap</b>(<i>func</i>) </td>
  <td> Similar to map, but each input item can be mapped to 0 or more output items (so <i>func</i> should return a Seq rather than a single item). </td>
</tr>
<tr>
  <td> <b>mapPartitions</b>(<i>func</i>) <a name="MapPartLink"></a> </td>
  <td> Similar to map, but runs separately on each partition (block) of the RDD, so <i>func</i> must be of type
    Iterator&lt;T&gt; =&gt; Iterator&lt;U&gt; when running on an RDD of type T. </td>
</tr>
<tr>
  <td> <b>mapPartitionsWithIndex</b>(<i>func</i>) </td>
  <td> Similar to mapPartitions, but also provides <i>func</i> with an integer value representing the index of
  the partition, so <i>func</i> must be of type (Int, Iterator&lt;T&gt;) =&gt; Iterator&lt;U&gt; when running on an RDD of type T.
  </td>
</tr>
<tr>
  <td> <b>sample</b>(<i>withReplacement</i>, <i>fraction</i>, <i>seed</i>) </td>
  <td> Sample a fraction <i>fraction</i> of the data, with or without replacement, using a given random number generator seed. </td>
</tr>
<tr>
  <td> <b>union</b>(<i>otherDataset</i>) </td>
  <td> Return a new dataset that contains the union of the elements in the source dataset and the argument. </td>
</tr>
<tr>
  <td> <b>intersection</b>(<i>otherDataset</i>) </td>
  <td> Return a new RDD that contains the intersection of elements in the source dataset and the argument. </td>
</tr>
<tr>
  <td> <b>distinct</b>([<i>numTasks</i>])) </td>
  <td> Return a new dataset that contains the distinct elements of the source dataset.</td>
</tr>
<tr>
  <td> <b>groupByKey</b>([<i>numTasks</i>]) <a name="GroupByLink"></a> </td>
  <td> When called on a dataset of (K, V) pairs, returns a dataset of (K, Iterable&lt;V&gt;) pairs. <br>
    <b>Note:</b> If you are grouping in order to perform an aggregation (such as a sum or
      average) over each key, using <code>reduceByKey</code> or <code>aggregateByKey</code> will yield much better 
      performance.
    <br>
    <b>Note:</b> By default, the level of parallelism in the output depends on the number of partitions of the parent RDD.
      You can pass an optional <code>numTasks</code> argument to set a different number of tasks.
  </td>
</tr>
<tr>
  <td> <b>reduceByKey</b>(<i>func</i>, [<i>numTasks</i>]) <a name="ReduceByLink"></a> </td>
  <td> When called on a dataset of (K, V) pairs, returns a dataset of (K, V) pairs where the values for each key are aggregated using the given reduce function <i>func</i>, which must be of type (V,V) =&gt; V. Like in <code>groupByKey</code>, the number of reduce tasks is configurable through an optional second argument. </td>
</tr>
<tr>
  <td> <b>aggregateByKey</b>(<i>zeroValue</i>)(<i>seqOp</i>, <i>combOp</i>, [<i>numTasks</i>]) <a name="AggregateByLink"></a> </td>
  <td> When called on a dataset of (K, V) pairs, returns a dataset of (K, U) pairs where the values for each key are aggregated using the given combine functions and a neutral "zero" value. Allows an aggregated value type that is different than the input value type, while avoiding unnecessary allocations. Like in <code>groupByKey</code>, the number of reduce tasks is configurable through an optional second argument. </td>
</tr>
<tr>
  <td> <b>sortByKey</b>([<i>ascending</i>], [<i>numTasks</i>]) <a name="SortByLink"></a> </td>
  <td> When called on a dataset of (K, V) pairs where K implements Ordered, returns a dataset of (K, V) pairs sorted by keys in ascending or descending order, as specified in the boolean <code>ascending</code> argument.</td>
</tr>
<tr>
  <td> <b>join</b>(<i>otherDataset</i>, [<i>numTasks</i>]) <a name="JoinLink"></a> </td>
  <td> When called on datasets of type (K, V) and (K, W), returns a dataset of (K, (V, W)) pairs with all pairs of elements for each key.
    Outer joins are supported through <code>leftOuterJoin</code>, <code>rightOuterJoin</code>, and <code>fullOuterJoin</code>.
  </td>
</tr>
<tr>
  <td> <b>cogroup</b>(<i>otherDataset</i>, [<i>numTasks</i>]) <a name="CogroupLink"></a> </td>
  <td> When called on datasets of type (K, V) and (K, W), returns a dataset of (K, (Iterable&lt;V&gt;, Iterable&lt;W&gt;)) tuples. This operation is also called <code>groupWith</code>. </td>
</tr>
<tr>
  <td> <b>cartesian</b>(<i>otherDataset</i>) </td>
  <td> When called on datasets of types T and U, returns a dataset of (T, U) pairs (all pairs of elements). </td>
</tr>
<tr>
  <td> <b>pipe</b>(<i>command</i>, <i>[envVars]</i>) </td>
  <td> Pipe each partition of the RDD through a shell command, e.g. a Perl or bash script. RDD elements are written to the
    process's stdin and lines output to its stdout are returned as an RDD of strings. </td>
</tr>
<tr>
  <td> <b>coalesce</b>(<i>numPartitions</i>) <a name="CoalesceLink"></a> </td>
  <td> Decrease the number of partitions in the RDD to numPartitions. Useful for running operations more efficiently
    after filtering down a large dataset. </td>
</tr>
<tr>
  <td> <b>repartition</b>(<i>numPartitions</i>) </td>
  <td> Reshuffle the data in the RDD randomly to create either more or fewer partitions and balance it across them.
    This always shuffles all data over the network. <a name="RepartitionLink"></a></td>
</tr>
<tr>
  <td> <b>repartitionAndSortWithinPartitions</b>(<i>partitioner</i>) <a name="Repartition2Link"></a></td>
  <td> Repartition the RDD according to the given partitioner and, within each resulting partition,
  sort records by their keys. This is more efficient than calling <code>repartition</code> and then sorting within 
  each partition because it can push the sorting down into the shuffle machinery. </td>
</tr>
</tbody></table>

## Actions

The following table lists some of the common actions supported by Spark. Refer to the RDD API doc (Scala, Java, Python, R) and pair RDD functions doc (Scala, Java) for details.
下面的表格列出了常用的动作。选择RDD的接口可以查看详情。

<table class="table"><tbody><tr><th>Action</th><th>Meaning</th></tr><tr>  <td> <b>reduce</b>(<i>func</i>) </td>  <td> Aggregate the elements of the dataset using a function <i>func</i> (which takes two arguments and returns one). The function should be commutative and associative so that it can be computed correctly in parallel. </td></tr><tr>  <td> <b>collect</b>() </td>  <td> Return all the elements of the dataset as an array at the driver program. This is usually useful after a filter or other operation that returns a sufficiently small subset of the data. </td></tr><tr>  <td> <b>count</b>() </td>  <td> Return the number of elements in the dataset. </td></tr><tr>  <td> <b>first</b>() </td>  <td> Return the first element of the dataset (similar to take(1)). </td></tr><tr>  <td> <b>take</b>(<i>n</i>) </td>  <td> Return an array with the first <i>n</i> elements of the dataset. </td></tr><tr>  <td> <b>takeSample</b>(<i>withReplacement</i>, <i>num</i>, [<i>seed</i>]) </td>  <td> Return an array with a random sample of <i>num</i> elements of the dataset, with or without replacement, optionally pre-specifying a random number generator seed.</td></tr><tr>  <td> <b>takeOrdered</b>(<i>n</i>, <i>[ordering]</i>) </td>  <td> Return the first <i>n</i> elements of the RDD using either their natural order or a custom comparator. </td></tr><tr>  <td> <b>saveAsTextFile</b>(<i>path</i>) </td>  <td> Write the elements of the dataset as a text file (or set of text files) in a given directory in the local filesystem, HDFS or any other Hadoop-supported file system. Spark will call toString on each element to convert it to a line of text in the file. </td></tr><tr>  <td> <b>saveAsSequenceFile</b>(<i>path</i>) <br> (Java and Scala) </td>  <td> Write the elements of the dataset as a Hadoop SequenceFile in a given path in the local filesystem, HDFS or any other Hadoop-supported file system. This is available on RDDs of key-value pairs that implement Hadoop's Writable interface. In Scala, it is also   available on types that are implicitly convertible to Writable (Spark includes conversions for basic types like Int, Double, String, etc). </td></tr><tr>  <td> <b>saveAsObjectFile</b>(<i>path</i>) <br> (Java and Scala) </td>  <td> Write the elements of the dataset in a simple format using Java serialization, which can then be loaded using    <code>SparkContext.objectFile()</code>. </td></tr><tr>  <td> <b>countByKey</b>() <a name="CountByLink"></a> </td>  <td> Only available on RDDs of type (K, V). Returns a hashmap of (K, Int) pairs with the count of each key. </td></tr><tr>  <td> <b>foreach</b>(<i>func</i>) </td>  <td> Run a function <i>func</i> on each element of the dataset. This is usually done for side effects such as updating an <a href="#AccumLink">Accumulator</a> or interacting with external storage systems.   <br><b>Note</b>: modifying variables other than Accumulators outside of the <code>foreach()</code> may result in undefined behavior. See <a href="#ClosuresLink">Understanding closures </a> for more details.</td></tr></tbody></table>

## Shuffle operations
Certain operations within Spark trigger an event known as the shuffle. The shuffle is Spark’s mechanism for re-distributing data so that it’s grouped differently across partitions. This typically involves copying data across executors and machines, making the shuffle a complex and costly operation.

Spark的某个操作出发一个叫做混洗的事件。 混洗是Spark的机制，用来重新分配数据从而让不同的分区见可以按键值成组。这个典型的涉及到从执行器和机器拷贝数据，使混洗成为复杂和昂贵的操作。

### Background
To understand what happens during the shuffle we can consider the example of the reduceByKey operation. The reduceByKey operation generates a new RDD where all values for a single key are combined into a tuple - the key and the result of executing a reduce function against all values associated with that key. The challenge is that not all values for a single key necessarily reside on the same partition, or even the same machine, but they must be co-located to compute the result.

为了理解在混洗过程中发生的事情，我们可以考虑观察reduceByKey操作。操作生成一个新的RDD，在这里键的所有值都合并到一个数组中。挑战是并非所有键的值都在相同的分区中，或者同一个机器中，但是它们必须集合起来才能计算结果。

In Spark, data is generally not distributed across partitions to be in the necessary place for a specific operation. During computations, a single task will operate on a single partition - thus, to organize all the data for a single reduceByKey reduce task to execute, Spark needs to perform an all-to-all operation. It must read from all partitions to find all the values for all keys, and then bring together values across partitions to compute the final result for each key - this is called the shuffle.

在Spark中，数据大体上并非分布在分区中，为了一个特殊的操作。在计算中，一个单独的任务会执行在一个单独的分区中。因此，为了组织所有的数据为了一个单独的reduceByKey reduce任务执行，Spark需要执行一个全部到全部的操作。它必须从所有分区中读取所有键的值，然后将值带离分区来进行每个键的值的和，这就叫做混洗。

Although the set of elements in each partition of newly shuffled data will be deterministic, and so is the ordering of partitions themselves, the ordering of these elements is not. If one desires predictably ordered data following shuffle then it’s possible to use:

* mapPartitions to sort each partition using, for example, .sorted
* repartitionAndSortWithinPartitions to efficiently sort partitions while simultaneously repartitioning
* sortBy to make a globally ordered RDD

尽管每个分区中的元素集会被确定，还有分区的顺序，这些元素的顺序没有确定。如果需要排好序的数据，他需要用：

* mapPartitions可以排序分区，利用.sorted
* 来有效的排序分区，同时有效的重新排序
* sortBy可以生成一个全局有序RDD

Operations which can cause a shuffle include repartition operations like repartition and coalesce, ‘ByKey operations (except for counting) like groupByKey and reduceByKey, and join operations like cogroup and join.

会造成混洗的操作，比如repartition和coalesce，ByKey操作，比如groupByKey和reduceByKey操作，还有join 操作，比如cogroup和join。

## Performance Impact
The Shuffle is an expensive operation since it involves disk I/O, data serialization, and network I/O. To organize data for the shuffle, Spark generates sets of tasks - map tasks to organize the data, and a set of reduce tasks to aggregate it. This nomenclature comes from MapReduce and does not directly relate to Spark’s map and reduce operations.

混洗操作时昂贵的，因为它涉及到磁盘IO，数据序列化和网络IO。为了组织混洗的数据，Spark生成任务集合——map任务来组织数据，和一系列reduce任务来聚合它。这些命名来自MapReduce，并没有字节涉及到Spark的map和reduce操作。

Internally, results from individual map tasks are kept in memory until they can’t fit. Then, these are sorted based on the target partition and written to a single file. On the reduce side, tasks read the relevant sorted blocks.

内在的，单独的map任务的输出是存在内存中。然后这些输出会排序并且写入一个文件。在reduce阶段，任务读取相关的数据块。

Certain shuffle operations can consume significant amounts of heap memory since they employ in-memory data structures to organize records before or after transferring them. Specifically, reduceByKey and aggregateByKey create these structures on the map side, and 'ByKey operations generate these on the reduce side. When data does not fit in memory Spark will spill these tables to disk, incurring the additional overhead of disk I/O and increased garbage collection.

特定的混洗操作可以消耗掉大量内存，因为它利用内存数据结构来组织数据组。特别的，reduceByKey和aggregateByKey在map阶段创造了这些数据结构，并且在reduce阶段由ByKey操作生成这些结构。当数据不合适在内存中时，Spark会将这些表写到磁盘中，承受磁盘IO的额外大的开支，并且增加垃圾收集。
Shuffle also generates a large number of intermediate files on disk. As of Spark 1.3, these files are preserved until the corresponding RDDs are no longer used and are garbage collected. This is done so the shuffle files don’t need to be re-created if the lineage is re-computed. Garbage collection may happen only after a long period time, if the application retains references to these RDDs or if GC does not kick in frequently. This means that long-running Spark jobs may consume a large amount of disk space. The temporary storage directory is specified by the spark.local.dir configuration parameter when configuring the Spark context.混洗会在磁盘上生成大量的中间数据。在Spark1.3中，这些文件会保存，直到相应的RDD被销毁。所以混洗文件不必重新建立，如果传承重新计算的话。垃圾回收只在下面两件事发生很久以后才可能发生，如果1应用程序保留这些RDD的引用，2 GC没有频繁的操作。这意味着长时间执行的Saprk作业会消耗大量的磁盘空间。中间存储目录在配置Spark选项时被spark.local.dir这个配置参数指定。

Shuffle behavior can be tuned by adjusting a variety of configuration parameters. See the ‘Shuffle Behavior’ section within the Spark Configuration Guide.

混洗行为可以进行调优，参考Spark Configuration Guide这一章节。

## RDD Persistence
One of the most important capabilities in Spark is persisting (or caching) a dataset in memory across operations. When you persist an RDD, each node stores any partitions of it that it computes in memory and reuses them in other actions on that dataset (or datasets derived from it). This allows future actions to be much faster (often by more than 10x). Caching is a key tool for iterative algorithms and fast interactive use.

Spark的一个最重要的才能是在各种操作时保存数据集在内存中。当你们保留RDD时，每个节点都保存它的所有计算分区在内存中，并且重用它们在其它的行为中。这个使未来的行为更加快速。缓存时迭代算法加速的关键工具。

You can mark an RDD to be persisted using the persist() or cache() methods on it. The first time it is computed in an action, it will be kept in memory on the nodes. Spark’s cache is fault-tolerant – if any partition of an RDD is lost, it will automatically be recomputed using the transformations that originally created it.

你可以标记需要留存的RDD用persist()或者cache()方法。当它第一次计算行为时，它会在节点的内存中保存。Spark的缓存是错误容忍的，如果一个RDD的任何分区丢失了，它会自动重新计算出该分区。

In addition, each persisted RDD can be stored using a different storage level, allowing you, for example, to persist the dataset on disk, persist it in memory but as serialized Java objects (to save space), replicate it across nodes, or store it off-heap in Tachyon. These levels are set by passing a StorageLevel object (Scala, Java, Python) to persist(). The cache() method is a shorthand for using the default storage level, which is StorageLevel.MEMORY_ONLY (store deserialized objects in memory). The full set of storage levels is:

另外，每个留存的RDD都可以用不同存储级别，允许你保存数据集在磁盘上和内存中（以Java对象的方式），在节点间拷贝它，或者存储在Tachyon的离线堆上。这些级别通过传递一个StorageLevel对象给persiss函数。Cache()函数呢，使用默认的存储级别，就是storageLevel.MEMORY_ONLY了。

The full set of storage levels is: 
所有存储级别如下：
<table class="table"><tbody><tr><th style="width:23%">Storage Level</th><th>Meaning</th></tr><tr>  <td> MEMORY_ONLY </td>  <td> Store RDD as deserialized Java objects in the JVM. If the RDD does not fit in memory, some partitions will    not be cached and will be recomputed on the fly each time they're needed. This is the default level. </td></tr><tr>  <td> MEMORY_AND_DISK </td>  <td> Store RDD as deserialized Java objects in the JVM. If the RDD does not fit in memory, store the    partitions that don't fit on disk, and read them from there when they're needed. </td></tr><tr>  <td> MEMORY_ONLY_SER </td>  <td> Store RDD as <i>serialized</i> Java objects (one byte array per partition).    This is generally more space-efficient than deserialized objects, especially when using a    <a href="tuning.html">fast serializer</a>, but more CPU-intensive to read.  </td></tr><tr>  <td> MEMORY_AND_DISK_SER </td>  <td> Similar to MEMORY_ONLY_SER, but spill partitions that don't fit in memory to disk instead of    recomputing them on the fly each time they're needed. </td></tr><tr>  <td> DISK_ONLY </td>  <td> Store the RDD partitions only on disk. </td></tr><tr>  <td> MEMORY_ONLY_2, MEMORY_AND_DISK_2, etc.  </td>  <td> Same as the levels above, but replicate each partition on two cluster nodes. </td></tr><tr>  <td> OFF_HEAP (experimental) </td>  <td> Store RDD in serialized format in <a href="http://tachyon-project.org">Tachyon</a>.    Compared to MEMORY_ONLY_SER, OFF_HEAP reduces garbage collection overhead and allows executors    to be smaller and to share a pool of memory, making it attractive in environments with    large heaps or multiple concurrent applications. Furthermore, as the RDDs reside in Tachyon,    the crash of an executor does not lead to losing the in-memory cache. In this mode, the memory    in Tachyon is discardable. Thus, Tachyon does not attempt to reconstruct a block that it evicts    from memory. If you plan to use Tachyon as the off heap store, Spark is compatible with Tachyon    out-of-the-box. Please refer to this <a href="http://tachyon-project.org/master/Running-Spark-on-Tachyon.html">page</a>    for the suggested version pairings.  </td></tr></tbody></table>

Note: In Python, stored objects will always be serialized with the Pickle library, so it does not matter whether you choose a serialized level.
注意：在Python中，排序过的对象一直会被序列化，所以你选择序列化级别是没用的。

Spark also automatically persists some intermediate data in shuffle operations (e.g. reduceByKey), even without users calling persist. This is done to avoid recomputing the entire input if a node fails during the shuffle. We still recommend users call persist on the resulting RDD if they plan to reuse it.

Spark 还会自动的保存一些混洗操作中的中间数据，即使没有用户调用persisit。这些工作是为了避免重新计算所有的输入数据，假如一个节点在混洗过程中崩溃。但是我们依然推荐用户对结果RDD调用persist函数，如果他们计划重用它。## Which Storage Level to Choose?

Spark’s storage levels are meant to provide different trade-offs between memory usage and CPU efficiency. We recommend going through the following process to select one:

* If your RDDs fit comfortably with the default storage level (MEMORY_ONLY), leave them that way. This is the most CPU-efficient option, allowing operations on the RDDs to run as fast as possible.

* If not, try using MEMORY_ONLY_SER and selecting a fast serialization library to make the objects much more space-efficient, but still reasonably fast to access.

* Don’t spill to disk unless the functions that computed your datasets are expensive, or they filter a large amount of the data. Otherwise, recomputing a partition may be as fast as reading it from disk.

* Use the replicated storage levels if you want fast fault recovery (e.g. if using Spark to serve requests from a web application). All the storage levels provide full fault tolerance by recomputing lost data, but the replicated ones let you continue running tasks on the RDD without waiting to recompute a lost partition.

Spark的存储级别意味着提供不同的再内存利用和CPU效率之间的权衡。我们推荐利用下面的流程来选择。

* 如果你的RDD适合默认的存储级别，那就不要修改。这是CPU效率最好的选项，允许RDD的操作执行的尽可能快。
* 如果不是，尝试使用MEMORY_ONLY_SER参数并且选择一个快速序列化的库来是对象少占空间又能快速连接。
* 如果计算数据集的函数是昂贵的，或者他们有大量的数据，才可以写数据到磁盘。否则，重新计算一个分区，可能和从磁盘读取它一样消耗时间。
* 使用复制存储级别，如果你想加速 错误恢复时间。所有的存储级别都提供完全错误容忍通过重新计算丢失的数据，但是复制级别让你继续在RDD上执行任务，而不必等待丢失部分的计算。

* In environments with high amounts of memory or multiple applications, the experimental OFF_HEAP mode has several advantages:

 - It allows multiple executors to share the same pool of memory in Tachyon.
 - It significantly reduces garbage collection costs.
 - Cached data is not lost if individual executors crash.
 
* 在有大量内存和多个应用程序时，离线堆模式有几个优势：
 - 它允许多个执行器分享相同的Tachyon的内存。
 - 它显著的减少了垃圾回收代价。
 - 如果个别执行器崩溃，缓存数据并没有丢失。
 
### Removing Data
Spark automatically monitors cache usage on each node and drops out old data partitions in a least-recently-used (LRU) fashion. If you would like to manually remove an RDD instead of waiting for it to fall out of the cache, use the RDD.unpersist() method.

Spark自动的监控每个节点上的缓存利用，并且丢掉古旧的数据分区。如果你愿意手动的删除一个RDD，而不是等它被系统删除，利用RDD.unpersist()方法。

## Shared Variables
Normally, when a function passed to a Spark operation (such as map or reduce) is executed on a remote cluster node, it works on separate copies of all the variables used in the function. These variables are copied to each machine, and no updates to the variables on the remote machine are propagated back to the driver program. Supporting general, read-write shared variables across tasks would be inefficient. However, Spark does provide two limited types of shared variables for two common usage patterns: broadcast variables and accumulators.

通常，当一个函数是被执行在远程集群中时，它依靠函数中用到的所有变量的一个拷贝。这些变量被拷贝到每个机器，没有远程机器的变量变化会传播到驱动程序。支持人物之间的通用的读写共享的变量会有效率。然而，Spark却是提供两种有限的共享变量给两种常用的使用模式：广播变量和累计器.

## Broadcast Variables
Broadcast variables allow the programmer to keep a read-only variable cached on each machine rather than shipping a copy of it with tasks. They can be used, for example, to give every node a copy of a large input dataset in an efficient manner. Spark also attempts to distribute broadcast variables using efficient broadcast algorithms to reduce communication cost.

广播变量允许程序员保持一个只读变量在每个节点的缓存中，而不是跟随任务复制一个变量。他们可以用来给每个节点一个大输入数据集的拷贝，用一种有效率的方式。Spark还尝试用有效的广播算法来分发广播变量，以减少通信代价。

Spark actions are executed through a set of stages, separated by distributed “shuffle” operations. Spark automatically broadcasts the common data needed by tasks within each stage. The data broadcasted this way is cached in serialized form and deserialized before running each task. This means that explicitly creating broadcast variables is only useful when tasks across multiple stages need the same data or when caching the data in deserialized form is important.

Spark行为时通过一系列阶段执行的，被分离的混洗操作分割。Spark自动的广播被任务需要的共同数据。这样传播的数据以序列化的方式存在缓存中，并且在每个任务执行之前反序列化。 这意味着直接的创造广播变量仅仅在一下两个情况中有用，1 当多个阶段的任务需要同一个数据；2 需要被放进缓存的序列化的数据很重要。

Broadcast variables are created from a variable v by calling SparkContext.broadcast(v). The broadcast variable is a wrapper around v, and its value can be accessed by calling the value method. The code below shows this:

广播变量从变量V中创建，通过调用SparkContext.broadcast(v)。传播变量是v的包装者，并且它的值可以通过调用value方法得到。代码如下:

### python

	>>> broadcastVar = sc.broadcast([1, 2, 3])
	<pyspark.broadcast.Broadcast object at 0x102789f10>
	
	>>> broadcastVar.value
	[1, 2, 3]
	
After the broadcast variable is created, it should be used instead of the value v in any functions run on the cluster so that v is not shipped to the nodes more than once. In addition, the object v should not be modified after it is broadcast in order to ensure that all nodes get the same value of the broadcast variable (e.g. if the variable is shipped to a new node later).

在广播变量创建之后，它会在每个执行在集群上的函数中替代v，以至于v不会发出到节点多次。此外，对象v在广播之后不能被修改，为了确保广播变量在每个节点是相同的。

## Accumulators
Accumulators are variables that are only “added” to through an associative operation and can therefore be efficiently supported in parallel. They can be used to implement counters (as in MapReduce) or sums. Spark natively supports accumulators of numeric types, and programmers can add support for new types. If accumulators are created with a name, they will be displayed in Spark’s UI. This can be useful for understanding the progress of running stages (NOTE: this is not yet supported in Python).

累计器是一个变量，它只能从联合操作中变更值，因此有效的支持并行处理。他们可以被用来执行计数和加运算。Spark天生支持数字类型的累计，程序员可以增加对其它类型的支持。如果累计器创建时有名字，它们会显示在Spark的界面上。这个有助于了解程序运行的阶段。

An accumulator is created from an initial value v by calling SparkContext.accumulator(v). Tasks running on the cluster can then add to it using the add method or the += operator (in Scala and Python). However, they cannot read its value. Only the driver program can read the accumulator’s value, using its value method.

一个累计器以有初始值v调用SparkContext.accumulator(v)来创建的。运行在集群中的任务可以对其执行+操作或者+=操作。然而，它们不能读取累计器的值。只有驱动程序可以读取累计器的值，使用value方法。

The code below shows an accumulator being used to add up the elements of an array:

下面代码展示一个累计器用来加数组的元素。

### python

	>>> accum = sc.accumulator(0)
	Accumulator<id=0, value=0>
	
	>>> sc.parallelize([1, 2, 3, 4]).foreach(lambda x: accum.add(x))
	...
	10/09/29 18:41:08 INFO SparkContext: Tasks finished in 0.317106 s
	
	scala> accum.value
	10While this code used the built-in support for accumulators of type Int, programmers can also create their own types by subclassing AccumulatorParam. The AccumulatorParam interface has two methods: zero for providing a “zero value” for your data type, and addInPlace for adding two values together. For example, supposing we had a Vector class representing mathematical vectors, we could write:

当这些代码使用int类型的累计器，程序员可以创造自己的数据类型。累计器参数的接口有两个方法：zero和addInPlace。举个例子，假设我们有一个向量类代表数学向量，我们可以这样写：

	class VectorAccumulatorParam(AccumulatorParam):
	    def zero(self, initialValue):
	        return Vector.zeros(initialValue.size)
	
	    def addInPlace(self, v1, v2):
	        v1 += v2
	        return v1
	
	# Then, create an Accumulator of this type:
	vecAccum = sc.accumulator(Vector(...), VectorAccumulatorParam())
	
For accumulator updates performed inside actions only, Spark guarantees that each task’s update to the accumulator will only be applied once, i.e. restarted tasks will not update the value. In transformations, users should be aware of that each task’s update may be applied more than once if tasks or job stages are re-executed.

因为累计器只有在行为时才会更新，Spark保证每个任务的累计器的更新只会申请一次，比如重启任务不会更新值。在转化中，用户应该意识到，每个任务的更新超过一次，如果任务或者作业的阶段被重新执行过。

Accumulators do not change the lazy evaluation model of Spark. If they are being updated within an operation on an RDD, their value is only updated once that RDD is computed as part of an action. Consequently, accumulator updates are not guaranteed to be executed when made within a lazy transformation like map(). The below code fragment demonstrates this property:

累计器没有改变Spark的懒惰赋值模式。如果它们被RDD的一个操作更改了值，它们的值仅仅被修改一次，就是RDD当做行为的一部分被计算是。结果是，累计器的修改没有被保证执行，当被map函数创造时。下面的代码描述了这个属性。

### python

	accum = sc.accumulator(0)
	def g(x):
	  accum.add(x)
	  return f(x)
	data.map(g)
	# Here, accum is still 0 because no actions have caused the `map` to be computed.## Deploying to a Cluster

The application submission guide describes how to submit applications to a cluster. In short, once you package your application into a JAR (for Java/Scala) or a set of .py or .zip files (for Python), the bin/spark-submit script lets you submit it to any supported cluster manager.

## Launching Spark jobs from Java / Scala

The org.apache.spark.launcher package provides classes for launching Spark jobs as child processes using a simple Java API.

## Unit Testing

Spark is friendly to unit testing with any popular unit test framework. Simply create a SparkContext in your test with the master URL set to local, run your operations, and then call SparkContext.stop() to tear it down. Make sure you stop the context within a finally block or the test framework’s tearDown method, as Spark does not support two contexts running concurrently in the same program.

## Migrating from pre-1.0 Versions of Spark

### Python

Spark 1.0 freezes the API of Spark Core for the 1.X series, in that any API available today that is not marked “experimental” or “developer API” will be supported in future versions. The only change for Python users is that the grouping operations, e.g. groupByKey, cogroup and join, have changed from returning (key, list of values) pairs to (key, iterable of values).

Migration guides are also available for Spark Streaming, MLlib and GraphX.

## Where to Go from Here

You can see some example Spark programs on the Spark website. In addition, Spark includes several samples in the examples directory (Scala, Java, Python, R). You can run Java and Scala examples by passing the class name to Spark’s bin/run-example script; for instance:

	./bin/run-example SparkPi
	
For Python examples, use spark-submit instead:

	./bin/spark-submit examples/src/main/python/pi.py
	
For R examples, use spark-submit instead:

	./bin/spark-submit examples/src/main/r/dataframe.R
	
For help on optimizing your programs, the configuration and tuning guides provide information on best practices. They are especially important for making sure that your data is stored in memory in an efficient format. For help on deploying, the cluster mode overview describes the components involved in distributed operation and supported cluster managers.

Finally, full API documentation is available in Scala, Java, Python and R.












  
  
  
