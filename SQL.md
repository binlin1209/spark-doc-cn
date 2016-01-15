# Spark SQL and DataFrame Guide

* Overview
* DataFrames
  - Starting Point: SQL Context
  - Creating DataFrames
  - DataFrames Operations
  - Running SQL Queries Programmatically
  - Interoperating with RDDs
     * Inferring the Schema Using Reflection
     * Programmatically Specifying the Schema
* Data Sources
  - Generic Load/Save Functions
     * Manually Specifying Options
     * Save Modes
     * Saving to Persistent Tables
  - Parquet Files
     * Loading Data Programmatically
     * Partition Discovery
     * Schema Merging
     * Hive metastore Parquet table conversion
        * Hive/Parquet Schema Reconciliation
        * Metadata Refreshing
     * Configuration
   - JSON Datasets
   - Hive Tables
     * Interacting with Different Versions of Hive Metastore
   - JDBC To Other Databases
   - Troubleshooting
* Performance Tuning
  - Caching Data In Memory
  - Other Configuration Options
* Distributed SQL Engine
  - Running the Thrift JDBC/ODBC server
  - Running the Spark SQL CLI
* Migration Guide
  - Upgrading From Spark SQL 1.4 to 1.5
  - Upgrading from Spark SQL 1.3 to 1.4
     * DataFrame data reader/writer interface
     * DataFrame.groupBy retains grouping columns
  - Upgrading from Spark SQL 1.0-1.2 to 1.3
     * Rename of SchemaRDD to DataFrame
     * Unification of the Java and Scala APIs
     * Isolation of Implicit Conversions and Removal of dsl Package (Scala-only)
     * Removal of the type aliases in org.apache.spark.sql for DataType (Scala-only)
     * UDF Registration Moved to sqlContext.udf (Java & Scala)
     * Python DataTypes No Longer Singletons
  - Migration Guide for Shark Users
     * Scheduling
     * Reducer number
     * Caching
  - Compatibility with Apache Hive
     * Deploying in Existing Hive Warehouses
     * Supported Hive Features
     * Unsupported Hive Functionality
* Reference
  - Data Types
  - NaN Semantics    


## Overview

Spark SQL is a Spark module for structured data processing. It provides a programming abstraction called DataFrames and can also act as distributed SQL query engine.

spark SQL 是spark的一个结构化数据处理的模块。它提供了一个程序抽象称为DataFrames，并且能够作为分布式SQL 查询引擎。

Spark SQL can also be used to read data from an existing Hive installation. For more on how to configure this feature, please refer to the Hive Tables section.

Spark SQL还能够被用作从一个已经存在的Hive装备来读取数据。如果想要了解更多的关于这种特性的配置，请阅读 Hive table 部分。

## DataFrames

A DataFrame is a distributed collection of data organized into named columns. It is conceptually equivalent to a table in a relational database or a data frame in R/Python, but with richer optimizations under the hood. DataFrames can be constructed from a wide array of sources such as: structured data files, tables in Hive, external databases, or existing RDDs.

DataFrame 是一个分布式的数据集，通过列名来组织的。从概念上等价于一个关系数据库中的表，或者 R/Python 中的数据框架，但是其优化性能更好。DataFrames 可以被一系列数组源创建，比如结构化的数据文件，Hive中的表，外围数据库，或者存在的RDDs。

The DataFrame API is available in Scala, Java, Python, and R.

Scala, Java, Python, 和 R 都有DataFrame的API。

All of the examples on this page use sample data included in the Spark distribution and can be run in the spark-shell, pyspark shell, or sparkR shell.

在本页中的所有例子使用spark分布的样例数据，可以在spark-shell, pyspark shell, or sparkR shell中运行。

### Starting Point: SQLContext
__Python__

The entry point into all relational functionality in Spark is the SQLContext class, or one of its decedents. To create a basic SQLContext, all you need is a SparkContext.

spark中所有关系功能的进入点都是 SQLContext 类，或者它的子类。为了创建一个基本的SQLContext， 你所需要的就是一个SparkContext。


	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	
In addition to the basic SQLContext, you can also create a HiveContext, which provides a superset of the functionality provided by the basic SQLContext. Additional features include the ability to write queries using the more complete HiveQL parser, access to Hive UDFs, and the ability to read data from Hive tables. To use a HiveContext, you do not need to have an existing Hive setup, and all of the data sources available to a SQLContext are still available. HiveContext is only packaged separately to avoid including all of Hive’s dependencies in the default Spark build. If these dependencies are not a problem for your application then using HiveContext is recommended for the 1.3 release of Spark. Future releases will focus on bringing SQLContext up to feature parity with a HiveContext.

除了基本的SQLContext，你还需要创建一个HiveContext, 它提供了基本的SQLContext的功能的超集。额外的特性包括使用更加全面的HiveQL程序能写查询。为了能够使用一个HiveContext，你不需要拥有一个已经存在的Hive设置，SQLContext所有可用的数据源依然可以用。在默认的spark部署中，HiveContext仅仅会被各自打包来避免包括所有的Hive依赖。对于你的应用，如果这些依赖都不是问题，那么1.3及其之后的spark版本是被推荐用HiveContext。未来发布的版本将会集中于带来SQLContext的特性的平衡。

The specific variant of SQL that is used to parse queries can also be selected using the spark.sql.dialect option. This parameter can be changed using either the setConf method on a SQLContext or by using a SET key=value command in SQL. For a SQLContext, the only dialect available is “sql” which uses a simple SQL parser provided by Spark SQL. In a HiveContext, the default is “hiveql”, though “sql” is also available. Since the HiveQL parser is much more complete, this is recommended for most use cases.

SQL的详细变量被用作查询，也可以被用作spark.sql.dialect来选择。这些参数可以在setconf 方法或者SET key=value 来改变，仅仅可用的方言就是sql，这是由Spark SQL提供的最简单的sql语句。在HiveContext中，默认值是hiveql，尽管sql依然可用。由于HiveQL更加全面，所以推荐用HiveQL。

### Creating DataFrames

With a SQLContext, applications can create DataFrames from an existing RDD, from a Hive table, or from data sources.

在SQLContext下，应用程序能够从一个存在的RDD中创建一个 DataFrames，从一个Hive表或者从一个数据源。

As an example, the following creates a DataFrame based on the content of a JSON file:

根据JSON文件，下面一个例子就创建了一个dataframe。


__Python__

	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	
	df = sqlContext.read.json("examples/src/main/resources/people.json")
	
	# Displays the content of the DataFrame to stdout
	df.show()
	
### DataFrame Operations

DataFrames provide a domain-specific language for structured data manipulation in Scala, Java, and Python.

DataFrames提供了一种特定的语言操作。

Here we include some basic examples of structured data processing using DataFrames:

接下来一些结构化数据结构的例子就使用了DataFrames。

__Python__

In Python it’s possible to access a DataFrame’s columns either by attribute (df.age) or by indexing (df['age']). While the former is convenient for interactive data exploration, users are highly encouraged to use the latter form, which is future proof and won’t break with column names that are also attributes on the DataFrame class.

在python中，可以通过属性或者索引来获取DataFrame中的列。尽管前者对于交互式的数据索引是非常方便的，用户被强烈推荐使用后者，这在着眼于将来且列名不会破坏，这也可以认为是DataFrame类中的属性。

	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	
	# Create the DataFrame
	df = sqlContext.read.json("examples/src/main/resources/people.json")
	
	# Show the content of the DataFrame
	df.show()
	## age  name
	## null Michael
	## 30   Andy
	## 19   Justin
	
	# Print the schema in a tree format
	df.printSchema()
	## root
	## |-- age: long (nullable = true)
	## |-- name: string (nullable = true)
	
	# Select only the "name" column
	df.select("name").show()
	## name
	## Michael
	## Andy
	## Justin
	
	# Select everybody, but increment the age by 1
	df.select(df['name'], df['age'] + 1).show()
	## name    (age + 1)
	## Michael null
	## Andy    31
	## Justin  20
	
	# Select people older than 21
	df.filter(df['age'] > 21).show()
	## age name
	## 30  Andy
	
	# Count people by age
	df.groupBy("age").count().show()
	## age  count
	## null 1
	## 19   1
	## 30   1
	
For a complete list of the types of operations that can be performed on a DataFrame refer to the API Documentation.

对于一个完整类型的操作列表，这依然可以在DataFrame中执行，可以参考API文档。

In addition to simple column references and expressions, DataFrames also have a rich library of functions including string manipulation, date arithmetic, common math operations and more. The complete list is available in the DataFrame Function Reference.

除了简单的列参考和表达，DataFrames也包括丰富的函数包，这包括线性操作，日期计算，普通数学计算。全部的列表在DataFrame Function Reference中。

### Running SQL Queries Programmatically

The sql function on a SQLContext enables applications to run SQL queries programmatically and returns the result as a DataFrame.

SQLContext中的SQL函数使得应用程序能够运行sql查询语句，以DataFrame格式返回结果。

__python__

	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	df = sqlContext.sql("SELECT * FROM table")
	
### Interoperating with RDDs

Spark SQL supports two different methods for converting existing RDDs into DataFrames. The first method uses reflection to infer the schema of an RDD that contains specific types of objects. This reflection based approach leads to more concise code and works well when you already know the schema while writing your Spark application.

spark SQL 支持两种不同的方法来转化已经存在的RDDs为DataFrames。第一个方法就是使用映射来推断出RDD的纲要，这包括对象的特定类型。这种映射根据方法转化更加精确的代码，当你知道纲要时，当你写你的spark应用时。

The second method for creating DataFrames is through a programmatic interface that allows you to construct a schema and then apply it to an existing RDD. While this method is more verbose, it allows you to construct DataFrames when the columns and their types are not known until runtime.

第二种创建DataFrames的方法就是通过编程接口，这将允许你创建一个纲要，然后把它运用在已经存在的RDD上。尽管这种方法更加冗长，它允许你创建DataFrames，当它的列和它们的类型还不知道，知道运行它们的时候。



### Inferring the Schema Using Reflection

__python__

Spark SQL can convert an RDD of Row objects to a DataFrame, inferring the datatypes. Rows are constructed by passing a list of key/value pairs as kwargs to the Row class. The keys of this list define the column names of the table, and the types are inferred by looking at the first row. Since we currently only look at the first row, it is important that there is no missing data in the first row of the RDD. In future versions we plan to more completely infer the schema by looking at more data, similar to the inference that is performed on JSON files.

spark SQL能转化行RDD为一个DataFrame，推断其数据类型。通过一个键／值对，行被创建了。列表的键定义了表中列的名字，通过观察表的第一行，表的类型已经被推断出来了。因为我们当前仅仅只是观察第一行，确保RDD中的第一行没有错误时非常重要的。在将来的版本中，我们计划推理出更加全面的类型，当处理更多的数据时。就像在JSON文件的推理一样。

	# sc is an existing SparkContext.
	from pyspark.sql import SQLContext, Row
	sqlContext = SQLContext(sc)
	
	# Load a text file and convert each line to a Row.
	lines = sc.textFile("examples/src/main/resources/people.txt")
	parts = lines.map(lambda l: l.split(","))
	people = parts.map(lambda p: Row(name=p[0], age=int(p[1])))
	
	# Infer the schema, and register the DataFrame as a table.
	schemaPeople = sqlContext.createDataFrame(people)
	schemaPeople.registerTempTable("people")
	
	# SQL can be run over DataFrames that have been registered as a table.
	teenagers = sqlContext.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")
	
	# The results of SQL queries are RDDs and support all the normal RDD operations.
	teenNames = teenagers.map(lambda p: "Name: " + p.name)
	for teenName in teenNames.collect():
	  print(teenName)


### Programmatically Specifying the Schema

__python__

When a dictionary of kwargs cannot be defined ahead of time (for example, the structure of records is encoded in a string, or a text dataset will be parsed and fields will be projected differently for different users), a DataFrame can be created programmatically with three steps.

1. Create an RDD of tuples or lists from the original RDD;
2. Create the schema represented by a StructType matching the structure of tuples or lists in the RDD created in the step 1.
3. Apply the schema to the RDD via createDataFrame method provided by SQLContext.

当一个字典kwargs不能提前定义时，（例如结构记录会被编码进字符，文本数据集将被解析和字段根据不同的用户来进行映射）一个dataframe会通过程序式创建，只需要三步。

1. 从最初的RDD创建一个RDD元组或者列表
2. 创建模式由StructType表示匹配的结构元组或列出在步骤1中创建的抽样。
3. 将模式应用于RDD通过SQLContext提供的createDataFrame方法。



For example:

	# Import SQLContext and data types
	from pyspark.sql import SQLContext
	from pyspark.sql.types import *
	
	# sc is an existing SparkContext.
	sqlContext = SQLContext(sc)
	
	# Load a text file and convert each line to a tuple.
	lines = sc.textFile("examples/src/main/resources/people.txt")
	parts = lines.map(lambda l: l.split(","))
	people = parts.map(lambda p: (p[0], p[1].strip()))
	
	# The schema is encoded in a string.
	schemaString = "name age"
	
	fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
	schema = StructType(fields)
	
	# Apply the schema to the RDD.
	schemaPeople = sqlContext.createDataFrame(people, schema)
	
	# Register the DataFrame as a table.
	schemaPeople.registerTempTable("people")
	
	# SQL can be run over DataFrames that have been registered as a table.
	results = sqlContext.sql("SELECT name FROM people")
	
	# The results of SQL queries are RDDs and support all the normal RDD operations.
	names = results.map(lambda p: "Name: " + p.name)
	for name in names.collect():
	  print(name)
	  
### Data Sources

Spark SQL supports operating on a variety of data sources through the DataFrame interface. A DataFrame can be operated on as normal RDDs and can also be registered as a temporary table. Registering a DataFrame as a table allows you to run SQL queries over its data. This section describes the general methods for loading and saving data using the Spark Data Sources and then goes into specific options that are available for the built-in data sources.

spark SQL支持操作在各种不同的数据源的DataFrame接口中。DataFrame可以在正规的RDDs上操作，也可以在刚刚注册的临时表上操作。刚注册的临时的DataFrame表允许你在数据集上运行SQL查询语句。这部分描述了加载和保存数据的大体上的方法，特定的选项在内置数据源上也是可以用的。

### Generic Load/Save Functions

In the simplest form, the default data source (parquet unless otherwise configured by spark.sql.sources.default) will be used for all operations.

在最简单的表上，默认的数据源将会被用在所有的操作中。

__python__

	df = sqlContext.read.load("examples/src/main/resources/users.parquet")
	df.select("name", "favorite_color").write.save("namesAndFavColors.parquet")

### Manually Specifying Options

You can also manually specify the data source that will be used along with any extra options that you would like to pass to the data source. Data sources are specified by their fully qualified name (i.e., org.apache.spark.sql.parquet), but for built-in sources you can also use their short names (json, parquet, jdbc). DataFrames of any type can be converted into other types using this syntax.

你也可以手动的设置数据源，这将会被用在所有额外的选项中。数据源已经被它们的命名方式所指定了。对于内置的源，你也可以用它们的简称，比如json, parquet, jdbc。DataFrames的任何其它类型也会被转化为其它类型，使用这种语法。

__python__

	df = sqlContext.read.load("examples/src/main/resources/people.json", format="json")
	df.select("name", "age").write.save("namesAndAges.parquet", format="parquet")

### Save Modes

Save operations can optionally take a SaveMode, that specifies how to handle existing data if present. It is important to realize that these save modes do not utilize any locking and are not atomic. Additionally, when performing a Overwrite, the data will be deleted before writing out the new data.

可以采取SaveMode保存操作，这具体说明的怎样管理已经存在的数据。意识到这些保存的模式没有使用任何锁操作或者原子操作是非常重要的。而且，当执行一个覆盖操作时，当写一个新数据时，这些数据将会被检测到。

<table class="table"><tbody><tr><th>Scala/Java</th><th>Any Language</th><th>Meaning</th></tr><tr>  <td><code>SaveMode.ErrorIfExists</code> (default)</td>  <td><code>"error"</code> (default)</td>  <td>    When saving a DataFrame to a data source, if data already exists,    an exception is expected to be thrown.  </td></tr><tr>  <td><code>SaveMode.Append</code></td>  <td><code>"append"</code></td>  <td>    When saving a DataFrame to a data source, if data/table already exists,    contents of the DataFrame are expected to be appended to existing data.  </td></tr><tr>  <td><code>SaveMode.Overwrite</code></td>  <td><code>"overwrite"</code></td>  <td>    Overwrite mode means that when saving a DataFrame to a data source,    if data/table already exists, existing data is expected to be overwritten by the contents of    the DataFrame.  </td></tr><tr>  <td><code>SaveMode.Ignore</code></td>  <td><code>"ignore"</code></td>  <td>    Ignore mode means that when saving a DataFrame to a data source, if data already exists,    the save operation is expected to not save the contents of the DataFrame and to not    change the existing data.  This is similar to a <code>CREATE TABLE IF NOT EXISTS</code> in SQL.  </td></tr></tbody></table>


### Saving to Persistent Tables

When working with a HiveContext, DataFrames can also be saved as persistent tables using the saveAsTable command. Unlike the registerTempTable command, saveAsTable will materialize the contents of the dataframe and create a pointer to the data in the HiveMetastore. Persistent tables will still exist even after your Spark program has restarted, as long as you maintain your connection to the same metastore. A DataFrame for a persistent table can be created by calling the table method on a SQLContext with the name of the table.

当在HiveContext下工作时，使用saveAsTable命令工作时，DataFrames也会被保存为永久表格。与注册临时表的指令不同，saveAsTable指令将会保存为dataframe的内容，且在HiveMetastore中创建一个指针。当你的spark程序再次启动的时候，永久表格依然存在，只要你保持你的连接到相同的metastore。一个永久表格的DataFrame会被创建，通过调用在SQLContext表格的方法，

By default saveAsTable will create a “managed table”, meaning that the location of the data will be controlled by the metastore. Managed tables will also have their data deleted automatically when a table is dropped.

默认情况下，saveAsTable会创建一个managed table，这意味着数据在表格中的位置会被metastore控制。当一个表格被放弃时，Managed tables 将会自动把数据删除。

## Parquet Files

Parquet is a columnar format that is supported by many other data processing systems. Spark SQL provides support for both reading and writing Parquet files that automatically preserves the schema of the original data.

Parquet是个以列为主的格式，其被其它的数据处理系统所支持。Spark SQL支持对Parquet文件的读和写操作，并且自动保护原始数据的纲要。

### Loading Data Programmatically

Using the data from the above example:

__python__

	# sqlContext from the previous example is used in this example.
	
	schemaPeople # The DataFrame from the previous example.

	# DataFrames can be saved as Parquet files, maintaining the schema information.
	schemaPeople.write.parquet("people.parquet")
	
	# Read in the Parquet file created above.  Parquet files are self-describing so the schema is preserved.
	# The result of loading a parquet file is also a DataFrame.
	parquetFile = sqlContext.read.parquet("people.parquet")
	
	# Parquet files can also be registered as tables and then used in SQL statements.
	parquetFile.registerTempTable("parquetFile");
	teenagers = sqlContext.sql("SELECT name FROM parquetFile WHERE age >= 13 AND age <= 19")
	teenNames = teenagers.map(lambda p: "Name: " + p.name)
	for teenName in teenNames.collect():
	  print(teenName)
	  
### Partition Discovery

Table partitioning is a common optimization approach used in systems like Hive. In a partitioned table, data are usually stored in different directories, with partitioning column values encoded in the path of each partition directory. The Parquet data source is now able to discover and infer partitioning information automatically. For example, we can store all our previously used population data into a partitioned table using the following directory structure, with two extra columns, gender and country as partitioning columns:

表格分区是一个通用的优化方法，比如在Hive中。在一个分区的表格中，数据通常被保存在不同的文件夹下，分区列表值被编码进每个分区文件夹的路径中。Parquet数据源现在能够被发现和自动推断分区信息。例如，我们可以存储我们先前最常用的数据在分区表格中，使用当前的文件夹结构，使用两个额外的列。

	path
	└── to
	    └── table
	        ├── gender=male
	        │   ├── ...
	        │   │
	        │   ├── country=US
	        │   │   └── data.parquet
	        │   ├── country=CN
	        │   │   └── data.parquet
	        │   └── ...
	        └── gender=female
	            ├── ...
	            │
	            ├── country=US
	            │   └── data.parquet
	            ├── country=CN
	            │   └── data.parquet
	            └── ...
	            
By passing path/to/table to either SQLContext.read.parquet or SQLContext.read.load, Spark SQL will automatically extract the partitioning information from the paths. Now the schema of the returned DataFrame becomes:

通过table路径到SQLContext.read.parquet或者SQLContext.read.load，Spark SQL会自动从路径中提取出分区信息。现在返回的DataFrame的纲要变成：

	root
	|-- name: string (nullable = true)
	|-- age: long (nullable = true)
	|-- gender: string (nullable = true)
	|-- country: string (nullable = true)

Notice that the data types of the partitioning columns are automatically inferred. Currently, numeric data types and string type are supported. Sometimes users may not want to automatically infer the data types of the partitioning columns. For these use cases, the automatic type inference can be configured by spark.sql.sources.partitionColumnTypeInference.enabled, which is default to true. When type inference is disabled, string type will be used for the partitioning columns.

注意，分区列表的数据类型会被自动推断出。目前，数值型数据类型和字符型数据类型都被支持。有时用户不想自动推断各个分区的数据类型。在这种情况下，自动类型推断会被spark.sql.sources.partitionColumnTypeInference.enabled来设置。在默认情况下，设置为true。当类型推断实效时，字符型会被应用在各个分区列表中。

### Schema Merging

Like ProtocolBuffer, Avro, and Thrift, Parquet also supports schema evolution. Users can start with a simple schema, and gradually add more columns to the schema as needed. In this way, users may end up with multiple Parquet files with different but mutually compatible schemas. The Parquet data source is now able to automatically detect this case and merge schemas of all these files. 

就像ProtocolBuffer, Avro, and Thrift一样，Parquet也支持模式演变。用户可以以一个简单的模式开始，然后逐渐的添加新的列，在需要的时候。以这种方式，用户最后可能会以多项Parquet文件结束，不同的但是兼容的模式。Parquet数据源能够自动检测出这种情况，合并所有的文件模式。


Since schema merging is a relatively expensive operation, and is not a necessity in most cases, we turned it off by default starting from 1.5.0. You may enable it by

1. setting data source option mergeSchema to true when reading Parquet files (as shown in the examples below), or
2. setting the global SQL option spark.sql.parquet.mergeSchema to true.

文件合并相对而言是一个昂贵的操作，在许多情况下，又是不必要的。自从1.5.0版本开始，就已经被关闭了。你必须使它：

1. 设置数据源选项合并模式为真，当读取Parquet文件时
2. 设置全局SQL选项spark.sql.parquet.mergeSchema为真。

__python__

	# sqlContext from the previous example is used in this example.
	
	# Create a simple DataFrame, stored into a partition directory
	df1 = sqlContext.createDataFrame(sc.parallelize(range(1, 6))\
	                                   .map(lambda i: Row(single=i, double=i * 2)))
	df1.write.parquet("data/test_table/key=1")
	
	# Create another DataFrame in a new partition directory,
	# adding a new column and dropping an existing column
	df2 = sqlContext.createDataFrame(sc.parallelize(range(6, 11))
	                                   .map(lambda i: Row(single=i, triple=i * 3)))
	df2.write.parquet("data/test_table/key=2")
	
	# Read the partitioned table
	df3 = sqlContext.read.option("mergeSchema", "true").parquet("data/test_table")
	df3.printSchema()
	
	# The final schema consists of all 3 columns in the Parquet files together
	# with the partitioning column appeared in the partition directory paths.
	# root
	# |-- single: int (nullable = true)
	# |-- double: int (nullable = true)
	# |-- triple: int (nullable = true)
	# |-- key : int (nullable = true)

### Hive metastore Parquet table conversion

When reading from and writing to Hive metastore Parquet tables, Spark SQL will try to use its own Parquet support instead of Hive SerDe for better performance. This behavior is controlled by the spark.sql.hive.convertMetastoreParquet configuration, and is turned on by default.

当读取或者写Hive metastore Parquet表时，Spark SQL 将尝试使用它们自己的 Parquet支持的文件，而不是Hive SerDe，为了更好的性能。这些行为被spark.sql.hive.convertMetastoreParquet配置文件控制，默认的情况下时打开的。

#### Hive/Parquet Schema Reconciliation

There are two key differences between Hive and Parquet from the perspective of table schema processing.

1. Hive is case insensitive, while Parquet is not
2. Hive considers all columns nullable, while nullability in Parquet is significant

从从表模式的角度出发，在 Hive 和 Parquet之间有两个重要的区别。

1. Hive对事件不敏感， Parquet则不是
2. Hive考虑所有的列空值，空值性在Parquet也是很重要的。

Due to this reason, we must reconcile Hive metastore schema with Parquet schema when converting a Hive metastore Parquet table to a Spark SQL Parquet table. The reconciliation rules are:

1. Fields that have the same name in both schema must have the same data type regardless of nullability. The reconciled field should have the data type of the Parquet side, so that nullability is respected.
2. The reconciled schema contains exactly those fields defined in Hive metastore schema.
 - Any fields that only appear in the Parquet schema are dropped in the reconciled schema.
 - Any fileds that only appear in the Hive metastore schema are added as nullable field in the reconciled schema.

由于这种原因，我们必须使Hive metastore与Parquet模式一致，当转化Hive metastore Parquet表为Spark SQL Parquet表时。协调机制为：

1. 具有相同的名称在这两个模式字段必须具有相同的数据类型，无论空性。和解的现场应该有实木复合地板侧的数据类型，以便为空性得到尊重。
2. 该和解方案只包含在蜂巢metastore架构中定义的这些领域。

 - 只有出现在平面架构的任何字段都下降了和解方案。
 - 只有出现在蜂巢metastore架构的任何个行业被添加为可为空字段中协调模式。

#### Metadata Refreshing

Spark SQL caches Parquet metadata for better performance. When Hive metastore Parquet table conversion is enabled, metadata of those converted tables are also cached. If these tables are updated by Hive or other external tools, you need to refresh them manually to ensure consistent metadata.

为了更好的性能，Spark SQL 缓存 Parquet metadata。当 Hive metastore Parquet表转化称为可能时，所有被转化的metadata表也被缓存了。如果这些表被Hive或者其它的工具更新，你必须手动刷新它们，这样才能保持一致的metadata。

__python__

	# sqlContext is an existing HiveContext
	sqlContext.refreshTable("my_table")

#### Configuration

Configuration of Parquet can be done using the setConf method on SQLContext or by running SET key=value commands using SQL.

使用SQLContext中的setConf方法可以配置Parquet，或者通过运行SQL中的SET key=value指令。

<table class="table"><tbody><tr><th>Property Name</th><th>Default</th><th>Meaning</th></tr><tr>  <td><code>spark.sql.parquet.binaryAsString</code></td>  <td>false</td>  <td>    Some other Parquet-producing systems, in particular Impala, Hive, and older versions of Spark SQL, do    not differentiate between binary data and strings when writing out the Parquet schema. This    flag tells Spark SQL to interpret binary data as a string to provide compatibility with these systems.  </td></tr><tr>  <td><code>spark.sql.parquet.int96AsTimestamp</code></td>  <td>true</td>  <td>    Some Parquet-producing systems, in particular Impala and Hive, store Timestamp into INT96.  This    flag tells Spark SQL to interpret INT96 data as a timestamp to provide compatibility with these systems.  </td></tr><tr>  <td><code>spark.sql.parquet.cacheMetadata</code></td>  <td>true</td>  <td>    Turns on caching of Parquet schema metadata. Can speed up querying of static data.  </td></tr><tr>  <td><code>spark.sql.parquet.compression.codec</code></td>  <td>gzip</td>  <td>    Sets the compression codec use when writing Parquet files. Acceptable values include:    uncompressed, snappy, gzip, lzo.  </td></tr><tr>  <td><code>spark.sql.parquet.filterPushdown</code></td>  <td>true</td>  <td>Enables Parquet filter push-down optimization when set to true.</td></tr><tr>  <td><code>spark.sql.hive.convertMetastoreParquet</code></td>  <td>true</td>  <td>    When set to false, Spark SQL will use the Hive SerDe for parquet tables instead of the built in    support.  </td></tr><tr>  <td><code>spark.sql.parquet.output.committer.class</code></td>  <td><code>org.apache.parquet.hadoop.<br>ParquetOutputCommitter</code></td>  <td>    <p>      The output committer class used by Parquet. The specified class needs to be a subclass of      <code>org.apache.hadoop.<br>mapreduce.OutputCommitter</code>.  Typically, it's also a      subclass of <code>org.apache.parquet.hadoop.ParquetOutputCommitter</code>.    </p>    <p>      <b>Note:</b>      </p><ul>        <li>          This option is automatically ignored if <code>spark.speculation</code> is turned on.        <>        <li>          This option must be set via Hadoop <code>Configuration</code> rather than Spark          <code>SQLConf</code>.        <>        <li>          This option overrides <code>spark.sql.sources.<br>outputCommitterClass</code>.        <>      </ul>    <p></p>    <p>      Spark SQL comes with a builtin      <code>org.apache.spark.sql.<br>parquet.DirectParquetOutputCommitter</code>, which can be more      efficient then the default Parquet output committer when writing data to S3.    </p>  </td></tr><tr>  <td><code>spark.sql.parquet.mergeSchema</code></td>  <td><code>false</code></td>  <td>    <p>      When true, the Parquet data source merges schemas collected from all data files, otherwise the      schema is picked from the summary file or a random data file if no summary file is available.    </p>  </td></tr></tbody></table>

### JSON Datasets

__python__

Spark SQL can automatically infer the schema of a JSON dataset and load it as a DataFrame. This conversion can be done using SQLContext.read.json on a JSON file.

Spark SQL可以自动的推断出JSON数据集的模式，或者加载它作为一个DataFrame。使用JSON文件中的SQLContext.read.json 可以执行这种转变。

Note that the file that is offered as a json file is not a typical JSON file. Each line must contain a separate, self-contained valid JSON object. As a consequence, a regular multi-line JSON file will most often fail.

注意，被json提供的文件并不是一个典型的JSON文件。每行都必须包含各自的，自我包含的JSON对象。结果，常规的多行JSON文件通常会失败。

	# sc is an existing SparkContext.
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	
	# A JSON dataset is pointed to by path.
	# The path can be either a single text file or a directory storing text files.
	people = sqlContext.read.json("examples/src/main/resources/people.json")
	
	# The inferred schema can be visualized using the printSchema() method.
	people.printSchema()
	# root
	#  |-- age: integer (nullable = true)
	#  |-- name: string (nullable = true)
	
	# Register this DataFrame as a table.
	people.registerTempTable("people")
	
	# SQL statements can be run by using the sql methods provided by `sqlContext`.
	teenagers = sqlContext.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")
	
	# Alternatively, a DataFrame can be created for a JSON dataset represented by
	# an RDD[String] storing one JSON object per string.
	anotherPeopleRDD = sc.parallelize([
	  '{"name":"Yin","address":{"city":"Columbus","state":"Ohio"}}'])
	anotherPeople = sqlContext.jsonRDD(anotherPeopleRDD)
### Hive Tables

Spark SQL also supports reading and writing data stored in Apache Hive. However, since Hive has a large number of dependencies, it is not included in the default Spark assembly. Hive support is enabled by adding the -Phive and -Phive-thriftserver flags to Spark’s build. This command builds a new assembly jar that includes Hive. Note that this Hive assembly jar must also be present on all of the worker nodes, as they will need access to the Hive serialization and deserialization libraries (SerDes) in order to access data stored in Hive.

Spark SQL也支持读和写数据，存储于Apache Hive。然而，由于Hive有许多数量的依赖，这不包括spark默认的配置。Hive通过添加-Phive 和 -Phive-thriftserver 标记来构建支持。这个指令构建了一个新的jar包，包含Hive. 注意这个Hive装配包必须在所有的工作节点上安装，因为它们需要访问Hive连续的或者不连续的包，为了能够把数据存储在Hive中。

Configuration of Hive is done by placing your hive-site.xml file in conf/. Please note when running the query on a YARN cluster (yarn-cluster mode), the datanucleus jars under the lib_managed/jars directory and hive-site.xml under conf/ directory need to be available on the driver and all executors launched by the YARN cluster. The convenient way to do this is adding them through the --jars option and --file option of the spark-submit command.

Hive的配置是在conf文件夹下放hive-site.xml。请注意当在YARN集群下运行时，lib_managed/jars下的datanucleus和conf文件夹下的hive-site.xml 是可用的，在驱动程序和执行程序下，通过YARN来管理。运行这个最方便的方法就是添加它们，通过spark-submit命令。

__python__

When working with Hive one must construct a HiveContext, which inherits from SQLContext, and adds support for finding tables in the MetaStore and writing queries using HiveQL.

当在Hive下工作时，必须创建一个HiveContext，它继承自SQLContext，在MetaStore中添加支持来寻找表，用HiveQL来写查询。

	# sc is an existing SparkContext.
	from pyspark.sql import HiveContext
	sqlContext = HiveContext(sc)
	
	sqlContext.sql("CREATE TABLE IF NOT EXISTS src (key INT, value STRING)")
	sqlContext.sql("LOAD DATA LOCAL INPATH 'examples/src/main/resources/kv1.txt' INTO TABLE src")
	
	# Queries can be expressed in HiveQL.
	results = sqlContext.sql("FROM src SELECT key, value").collect()
	
	
### Interacting with Different Versions of Hive Metastore

One of the most important pieces of Spark SQL’s Hive support is interaction with Hive metastore, which enables Spark SQL to access metadata of Hive tables. Starting from Spark 1.4.0, a single binary build of Spark SQL can be used to query different versions of Hive metastores, using the configuration described below. Note that independent of the version of Hive that is being used to talk to the metastore, internally Spark SQL will compile against Hive 1.2.1 and use those classes for internal execution (serdes, UDFs, UDAFs, etc).

Spark SQL’s Hive支持的最终的一点是互动,使得Spark SQL访问元数据表。从Spark 1.4.0版本开始，一个二进制构建的spark sql 能够查询不同版本的hive元数据，使用下面描述的配置。 注意，Hive的独立版本被用作存储元数据，spark sql将会编译 Hive 1.2.1版本，使用这些类来内部执行。

The following options can be used to configure the version of Hive that is used to retrieve metadata:

下面的选项被用作配置Hive版本，被用作恢复元数据。

<table class="table">  <tbody><tr><th>Property Name</th><th>Default</th><th>Meaning</th></tr>  <tr>    <td><code>spark.sql.hive.metastore.version</code></td>    <td><code>1.2.1</code></td>    <td>      Version of the Hive metastore. Available      options are <code>0.12.0</code> through <code>1.2.1</code>.    </td>  </tr>  <tr>    <td><code>spark.sql.hive.metastore.jars</code></td>    <td><code>builtin</code></td>    <td>      Location of the jars that should be used to instantiate the HiveMetastoreClient. This      property can be one of three options:      <ol>        <li><code>builtin</code><>        Use Hive 1.2.1, which is bundled with the Spark assembly jar when <code>-Phive</code> is        enabled. When this option is chosen, <code>spark.sql.hive.metastore.version</code> must be        either <code>1.2.1</code> or not defined.        <li><code>maven</code><>        Use Hive jars of specified version downloaded from Maven repositories.  This configuration        is not generally recommended for production deployments.         <li>A classpath in the standard format for the JVM.  This classpath must include all of Hive         and its dependencies, including the correct version of Hadoop.  These jars only need to be        present on the driver, but if you are running in yarn cluster mode then you must ensure        they are packaged with you application.<>      </ol>    </td>  </tr>  <tr>    <td><code>spark.sql.hive.metastore.sharedPrefixes</code></td>    <td><code>com.mysql.jdbc,<br>org.postgresql,<br>com.microsoft.sqlserver,<br>oracle.jdbc</code></td>    <td>      <p>        A comma separated list of class prefixes that should be loaded using the classloader that is        shared between Spark SQL and a specific version of Hive. An example of classes that should        be shared is JDBC drivers that are needed to talk to the metastore. Other classes that need        to be shared are those that interact with classes that are already shared. For example,        custom appenders that are used by log4j.      </p>    </td>  </tr>  <tr>    <td><code>spark.sql.hive.metastore.barrierPrefixes</code></td>    <td><code>(empty)</code></td>    <td>      <p>        A comma separated list of class prefixes that should explicitly be reloaded for each version        of Hive that Spark SQL is communicating with. For example, Hive UDFs that are declared in a        prefix that typically would be shared (i.e. <code>org.apache.spark.*</code>).      </p>    </td>  </tr></tbody></table>

### JDBC To Other Databases

Spark SQL also includes a data source that can read data from other databases using JDBC. This functionality should be preferred over using JdbcRDD. This is because the results are returned as a DataFrame and they can easily be processed in Spark SQL or joined with other data sources. The JDBC data source is also easier to use from Java or Python as it does not require the user to provide a ClassTag. (Note that this is different than the Spark SQL JDBC server, which allows other applications to run queries using Spark SQL).

Spark SQL也包括了使用JDBC，从其它数据源读取的数据。使用JdbcRDD，这个功能应该优先选择。这是因为这个结果是按DataFrame格式返回的。它们可以很容易的在Spark SQL下执行，于其它数据源连接。JDBC数据源也很容易在java和python中使用，因为它不需要用户提供ClassTag。（注意，这与Spark SQL JDBC服务器不同，它允许其它应用使用spark sql来运行查询语句。）

To get started you will need to include the JDBC driver for you particular database on the spark classpath. For example, to connect to postgres from the Spark Shell you would run the following command:
刚开始，你将需要用JDBC驱动程序来运行你spark路径下的数据集。例如，为了从spark shell中连接你的Postgres，你需要运行下列命令：

	SPARK_CLASSPATH=postgresql-9.3-1102-jdbc41.jar bin/spark-shell
	
Tables from the remote database can be loaded as a DataFrame or Spark SQL Temporary table using the Data Sources API. The following options are supported:

来自远程数据集的表被加载为DataFrame或者Spark SQL临时表，使用数据源API。支持以下选项。

<table class="table">  <tbody><tr><th>Property Name</th><th>Meaning</th></tr>  <tr>    <td><code>url</code></td>    <td>      The JDBC URL to connect to.    </td>  </tr>  <tr>    <td><code>dbtable</code></td>    <td>      The JDBC table that should be read.  Note that anything that is valid in a <code>FROM</code> clause of      a SQL query can be used.  For example, instead of a full table you could also use a      subquery in parentheses.    </td>  </tr>  <tr>    <td><code>driver</code></td>    <td>      The class name of the JDBC driver needed to connect to this URL.  This class will be loaded      on the master and workers before running an JDBC commands to allow the driver to      register itself with the JDBC subsystem.    </td>  </tr>  <tr>    <td><code>partitionColumn, lowerBound, upperBound, numPartitions</code></td>    <td>      These options must all be specified if any of them is specified.  They describe how to      partition the table when reading in parallel from multiple workers.      <code>partitionColumn</code> must be a numeric column from the table in question. Notice      that <code>lowerBound</code> and <code>upperBound</code> are just used to decide the      partition stride, not for filtering the rows in table. So all rows in the table will be      partitioned and returned.    </td>  </tr></tbody></table>

__python__

	df = sqlContext.read.format('jdbc').options(url='jdbc:postgresql:dbserver', dbtable='schema.tablename').load()
	
### Troubleshooting

* The JDBC driver class must be visible to the primordial class loader on the client session and on all executors. This is because Java’s DriverManager class does a security check that results in it ignoring all drivers not visible to the primordial class loader when one goes to open a connection. One convenient way to do this is to modify compute_classpath.sh on all worker nodes to include your driver JARs.
* Some databases, such as H2, convert all names to upper case. You’ll need to use upper case to refer to those names in Spark SQL.
* 在客户端部分和所有的执行部分，JDBC驱动类对于原始类加载者是可见的。这是因为JAVA的驱动管理类会进行一个安全的检查，这将导致忽略所有的不可见的原始类下载者，当一个连接被打开时。一个非常方便的方法就是修改所有工作节点上的compute_classpath.sh文件，包括你自己的驱动包。
* 一些数据集比如H2，转化所有的名字为上一个事件。你将需要使用上一个事件来指代那个在spark SQL中使用的名字。



## Performance Tuning

For some workloads it is possible to improve performance by either caching data in memory, or by turning on some experimental options.

对于一些工作，通过缓存数据在内存中，很有可能提高其性能，或者通过打开一些实验选项。

### Caching Data In Memory

Spark SQL can cache tables using an in-memory columnar format by calling sqlContext.cacheTable("tableName") or dataFrame.cache(). Then Spark SQL will scan only required columns and will automatically tune compression to minimize memory usage and GC pressure. You can call sqlContext.uncacheTable("tableName") to remove the table from memory.

spark SQL能够通过内存列形式来缓存表，通过调用sqlContext.cacheTable("tableName") 或 dataFrame.cache()。然后spark sql将会扫描需要的列，将会自动调节最小内存使用的压缩和GC压力。你也可以调用sqlContext.uncacheTable("tableName")命令来移除内存中的表。

Configuration of in-memory caching can be done using the setConf method on SQLContext or by running SET key=value commands using SQL.

内存中缓存的配置可以通过setConf方法来执行，或者通过运行 SET key=value 命令来使用。

<table class="table"><tbody><tr><th>Property Name</th><th>Default</th><th>Meaning</th></tr><tr>  <td><code>spark.sql.inMemoryColumnarStorage.compressed</code></td>  <td>true</td>  <td>    When set to true Spark SQL will automatically select a compression codec for each column based    on statistics of the data.  </td></tr><tr>  <td><code>spark.sql.inMemoryColumnarStorage.batchSize</code></td>  <td>10000</td>  <td>    Controls the size of batches for columnar caching.  Larger batch sizes can improve memory utilization    and compression, but risk OOMs when caching data.  </td></tr></tbody></table>

## Other Configuration Options

The following options can also be used to tune the performance of query execution. It is possible that these options will be deprecated in future release as more optimizations are performed automatically.

下列选项被用作调节查询性能。很有可能这些选项在将来会被放弃掉，应该更多的优化会被自动进行。

<table class="table">  <tbody><tr><th>Property Name</th><th>Default</th><th>Meaning</th></tr>  <tr>    <td><code>spark.sql.autoBroadcastJoinThreshold</code></td>    <td>10485760 (10 MB)</td>    <td>      Configures the maximum size in bytes for a table that will be broadcast to all worker nodes when      performing a join.  By setting this value to -1 broadcasting can be disabled.  Note that currently      statistics are only supported for Hive Metastore tables where the command      <code>ANALYZE TABLE &lt;tableName&gt; COMPUTE STATISTICS noscan</code> has been run.    </td>  </tr>  <tr>    <td><code>spark.sql.tungsten.enabled</code></td>    <td>true</td>    <td>      When true, use the optimized Tungsten physical execution backend which explicitly manages memory      and dynamically generates bytecode for expression evaluation.    </td>  </tr>  <tr>    <td><code>spark.sql.shuffle.partitions</code></td>    <td>200</td>    <td>      Configures the number of partitions to use when shuffling data for joins or aggregations.    </td>  </tr>  <tr>    <td><code>spark.sql.planner.externalSort</code></td>    <td>true</td>    <td>      When true, performs sorts spilling to disk as needed otherwise sort each partition in memory.    </td>  </tr></tbody></table>


## Distributed SQL Engine

Spark SQL can also act as a distributed query engine using its JDBC/ODBC or command-line interface. In this mode, end-users or applications can interact with Spark SQL directly to run SQL queries, without the need to write any code.

spark sql也能够被用作分布式查询引擎，通过使用它的JDBC/ODBC或者命令行界面。在这种模式下，最终的用户或者应用能够与Spark SQL直接相互作用来运行SQL查询语句，而不需要写任何代码。

### Running the Thrift JDBC/ODBC server

The Thrift JDBC/ODBC server implemented here corresponds to the HiveServer2 in Hive 1.2.1 You can test the JDBC server with the beeline script that comes with either Spark or Hive 1.2.1.

Thrift JDBC/ODBC服务器在这边执行，对应着Hive 1.2.1中的HiveServer2。你也可以测试JDBC服务端，使用beeline脚本，来自spark 或者Hive 1.2.1。

To start the JDBC/ODBC server, run the following in the Spark directory:

要开启JDBC/ODBC服务端，运行下列spark文件夹下的命令：

	./sbin/start-thriftserver.sh
	
This script accepts all bin/spark-submit command line options, plus a --hiveconf option to specify Hive properties. You may run ./sbin/start-thriftserver.sh --help for a complete list of all available options. By default, the server listens on localhost:10000. You may override this behaviour via either environment variables, i.e.:

这个脚本接受所有的bin/spark-submit命令选项。包括一个--hiveconf选项来具体说明Hive特性。你也可以运行./sbin/start-thriftserver.sh --help来得到一个完整的可用选项列表。默认，服务端监控localhost:10000端口。你可以通过设置环境变量来运行它。

	export HIVE_SERVER2_THRIFT_PORT=<listening-port>
	export HIVE_SERVER2_THRIFT_BIND_HOST=<listening-host>
	./sbin/start-thriftserver.sh \
	  --master <master-uri> \
	  ...
	  
or system properties:

	./sbin/start-thriftserver.sh \
	  --hiveconf hive.server2.thrift.port=<listening-port> \
	  --hiveconf hive.server2.thrift.bind.host=<listening-host> \
	  --master <master-uri>
	  ...
	  
Now you can use beeline to test the Thrift JDBC/ODBC server:

现在，你可以使用beeline来测试Thrift JDBC/ODBC服务端：

	./bin/beeline
	
Connect to the JDBC/ODBC server in beeline with:

连接beeline下的JDBC/ODBC服务端：

	beeline> !connect jdbc:hive2://localhost:10000
	
Beeline will ask you for a username and password. In non-secure mode, simply enter the username on your machine and a blank password. For secure mode, please follow the instructions given in the beeline documentation.

Beeline会向你要求一个用户名和密码。在非安全模式下，仅仅需要输入用户名和一个空白的密码就可以了。对于安全模式，只需要根据beeline文件的指示就运行就可以了。

Configuration of Hive is done by placing your hive-site.xml file in conf/.

通过把hive-site.xml文件放在conf文件夹，就可以配置Hive了。

You may also use the beeline script that comes with Hive.

你也可以使用Hive自带的beeline脚本来使用。

Thrift JDBC server also supports sending thrift RPC messages over HTTP transport. Use the following setting to enable HTTP mode as system property or in hive-site.xml file in conf/:

Thrift JDBC服务端也支持发送thrift RPC消息，通过HTTP通道。使用下面的设置来使HTTP模块作为系统特性或者在conf中放置hive-site.xml文件。

	hive.server2.transport.mode - Set this to value: http
	hive.server2.thrift.http.port - HTTP port number fo listen on; default is 10001
	hive.server2.http.endpoint - HTTP endpoint; default is cliservice
	
To test, use beeline to connect to the JDBC/ODBC server in http mode with:

为了检测，在http模式下，使用beeline来连接JDBC/ODBC服务端。

	beeline> !connect jdbc:hive2://<host>:<port>/<database>?hive.server2.transport.mode=http;hive.server2.thrift.http.path=<http_endpoint>

### Running the Spark SQL CLI

The Spark SQL CLI is a convenient tool to run the Hive metastore service in local mode and execute queries input from the command line. Note that the Spark SQL CLI cannot talk to the Thrift JDBC server.

Spark SQL CLI 是一个非常方便的工具来运行Hive metastore服务，在本地模式或者命令行执行查询输入。注意，the Spark SQL CLI不能够与Thrift JDBC 服务端通信。

To start the Spark SQL CLI, run the following in the Spark directory:

为了开始Spark SQL CLI，运行spark文件夹下面的命令：

	./bin/spark-sql

Configuration of Hive is done by placing your hive-site.xml file in conf/. You may run ./bin/spark-sql --help for a complete list of all available options.

把hive-site.xml文件放置到conf文件夹下，Hive的配置已经算完成了。你也可以运行./bin/spark-sql --help 来获得所有的命令选项。

## Migration Guide
### Upgrading From Spark SQL 1.4 to 1.5

* Optimized execution using manually managed memory (Tungsten) is now enabled by default, along with code generation for expression evaluation. These features can both be disabled by setting spark.sql.tungsten.enabled to `false.

* 优化执行使用人工管理的内存现在是默认启用的，通过表达式求值来生成代码。通过设置spark.sql.tungsten.enabled为false，这些特性都会消失。

* Parquet schema merging is no longer enabled by default. It can be re-enabled by setting spark.sql.parquet.mergeSchema to true.

* Parquet模式合并不再是默认启用。通过设置spark.sql.parquet.mergeSchema为true，将会被再次启动。

* Resolution of strings to columns in python now supports using dots (.) to qualify the column or access nested values. For example df['table.column.nestedField']. However, this means that if your column name contains any dots you must now escape them using backticks (e.g., table.`column.with.dots`.nested).

在python中，列表解析字符串支持使用点来获取列或者获取嵌套的值。例如df['table.column.nestedField']。然而，这意味着如果你的列名包含任何点，你现在必须使用引号来逃避它们

* In-memory columnar storage partition pruning is on by default. It can be disabled by setting spark.sql.inMemoryColumnarStorage.partitionPruning to false.

* 内存中的柱状存储分区修剪是默认打开的。通过设置spark.sql.inMemoryColumnarStorage.partitionPruning为false，也可以关闭它。

* Unlimited precision decimal columns are no longer supported, instead Spark SQL enforces a maximum precision of 38. When inferring schema from BigDecimal objects, a precision of (38, 18) is now used. When no precision is specified in DDL then the default remains Decimal(10, 0).

* 没有限制的精确的十进制列不再被支持，相反的，Spark SQL被实行，且精确度到38位。当从大的十进制来推断模式，精度在(38, 18)是被使用的。当没有精度被特别指定时，默认值仍然为(10, 0)。

* Timestamps are now stored at a precision of 1us, rather than 1ns

时间标签被存储为1us，而不是1ns。

* In the sql dialect, floating point numbers are now parsed as decimal. HiveQL parsing remains unchanged.

在sql语句中，浮点数是用十进制来表示的。HiveQL parsing仍然没有改变。

* The canonical name of SQL/DataFrame functions are now lower case (e.g. sum vs SUM).

* SQL/DataFrame的规范名称的函数都是小写字母。

* It has been determined that using the DirectOutputCommitter when speculation is enabled is unsafe and thus this output committer will not be used when speculation is on, independent of configuration.

* 使用DirectOutputCommitter已经确定,当推测出不安全。这个输出提交者将不会用于投机，而不管配置。

* JSON data source will not automatically load new files that are created by other applications (i.e. files that are not inserted to the dataset through Spark SQL). For a JSON persistent table (i.e. the metadata of the table is stored in Hive Metastore), users can use REFRESH TABLE SQL command or HiveContext’s refreshTable method to include those new files to the table. For a DataFrame representing a JSON dataset, users need to recreate the DataFrame and the new DataFrame will include new files.

* JSON数据源将不会自动加被其它应用创建的新文件，例如，文件没有通过Spark SQL插入到数据集。对于一个JSON类型持久的表，用户能够使用 REFRESH TABLE SQL命令或者HiveContext刷新表的方法来包含这些新文件到表中。对于一个DataFrame代表JSON数据集，用户需要再创造DataFrame，这个新的DataFrame将包含新的文件。


### Upgrading from Spark SQL 1.3 to 1.4
#### DataFrame data reader/writer interface

Based on user feedback, we created a new, more fluid API for reading data in (SQLContext.read) and writing data out (DataFrame.write), and deprecated the old APIs (e.g. SQLContext.parquetFile, SQLContext.jsonFile).

根据用户的反馈，我们创建了一个新的，更加流水化的API来读取数据和写数据，放弃了旧的API.

See the API docs for SQLContext.read ( Scala, Java, Python ) and DataFrame.write ( Scala, Java, Python ) more information.

通过SQLContext.read来阅读API文档，DataFrame.write更多的信息。

#### DataFrame.groupBy retains grouping columns

Based on user feedback, we changed the default behavior of DataFrame.groupBy().agg() to retain the grouping columns in the resulting DataFrame. To keep the behavior in 1.3, set spark.sql.retainGroupColumns to false.

根据用户的反馈，我们改变了DataFrame.groupBy().agg()默认的行为来保留列名。为了保持1.3的版本，设置spark.sql.retainGroupColumns为false。

__python__

	import pyspark.sql.functions as func
	
	# In 1.3.x, in order for the grouping column "department" to show up,
	# it must be included explicitly as part of the agg function call.
	df.groupBy("department").agg("department"), func.max("age"), func.sum("expense"))
	
	# In 1.4+, grouping column "department" is included automatically.
	df.groupBy("department").agg(func.max("age"), func.sum("expense"))
	
	# Revert to 1.3.x behavior (not retaining grouping column) by:
	sqlContext.setConf("spark.sql.retainGroupColumns", "false")
	
### Upgrading from Spark SQL 1.0-1.2 to 1.3

In Spark 1.3 we removed the “Alpha” label from Spark SQL and as part of this did a cleanup of the available APIs. From Spark 1.3 onwards, Spark SQL will provide binary compatibility with other releases in the 1.X series. This compatibility guarantee excludes APIs that are explicitly marked as unstable (i.e., DeveloperAPI or Experimental).

在spark 1.3 版本中，我们取消了Spark SQL中的“Alpha”标签，并且清理了所有的可用的APIS。从1.3版本向后，Spark SQL提供二进制兼容机制对其它的1.X系列。这种兼容性保证不包括明确标记为不稳定的api。

#### Rename of SchemaRDD to DataFrame

The largest change that users will notice when upgrading to Spark SQL 1.3 is that SchemaRDD has been renamed to DataFrame. This is primarily because DataFrames no longer inherit from RDD directly, but instead provide most of the functionality that RDDs provide though their own implementation. DataFrames can still be converted to RDDs by calling the .rdd method.

最大的改变就是用户将会通知，当更新spark SQL到1.3版本的时候，SchemaRDD被重新命名为DataFrame。着主要是因为DataFrames不再直接从RDD中继承而来。但是相反的提供了更多的功能，RDD通过它们自己来实现。通过调用.rdd方法，DataFrames仍然可以转化为RDDs。

In Scala there is a type alias from SchemaRDD to DataFrame to provide source compatibility for some use cases. It is still recommended that users update their code to use DataFrame instead. Java and Python users will need to update their code.

在scala中，有一个类型别名SchemaRDD DataFrame提供源兼容性对于一些用例。仍然被推荐为使用用户更新它们的代码来使用DataFrame。Java 和 Python用户会被用来更新它们的代码。

#### Unification of the Java and Scala APIs

Prior to Spark 1.3 there were separate Java compatible classes (JavaSQLContext and JavaSchemaRDD) that mirrored the Scala API. In Spark 1.3 the Java API and Scala API have been unified. Users of either language should use SQLContext and DataFrame. In general theses classes try to use types that are usable from both languages (i.e. Array instead of language specific collections). In some cases where no common type exists (e.g., for passing in closures or Maps) function overloading is used instead.



Additionally the Java specific types API has been removed. Users of both Scala and Java should use the classes present in org.apache.spark.sql.types to describe schema programmatically.

#### Isolation of Implicit Conversions and Removal of dsl Package (Scala-only)

Many of the code examples prior to Spark 1.3 started with import sqlContext._, which brought all of the functions from sqlContext into scope. In Spark 1.3 we have isolated the implicit conversions for converting RDDs into DataFrames into an object inside of the SQLContext. Users should now write import sqlContext.implicits._.

Additionally, the implicit conversions now only augment RDDs that are composed of Products (i.e., case classes or tuples) with a method toDF, instead of applying automatically.

When using function inside of the DSL (now replaced with the DataFrame API) users used to import org.apache.spark.sql.catalyst.dsl. Instead the public dataframe functions API should be used: import org.apache.spark.sql.functions._.

#### Removal of the type aliases in org.apache.spark.sql for DataType (Scala-only)

Spark 1.3 removes the type aliases that were present in the base sql package for DataType. Users should instead import the classes in org.apache.spark.sql.types

#### UDF Registration Moved to sqlContext.udf (Java & Scala)

Functions that are used to register UDFs, either for use in the DataFrame DSL or SQL, have been moved into the udf object in SQLContext.

__scala__

	sqlContext.udf.register("strLen", (s: String) => s.length())

Python UDF registration is unchanged.

#### Python DataTypes No Longer Singletons

When using DataTypes in Python you will need to construct them (i.e. StringType()) instead of referencing a singleton.

## Migration Guide for Shark Users
### Scheduling

To set a Fair Scheduler pool for a JDBC client session, users can set the spark.sql.thriftserver.scheduler.pool variable:

	SET spark.sql.thriftserver.scheduler.pool=accounting;

### Reducer number

In Shark, default reducer number is 1 and is controlled by the property mapred.reduce.tasks. Spark SQL deprecates this property in favor of spark.sql.shuffle.partitions, whose default value is 200. Users may customize this property via SET:

	SET spark.sql.shuffle.partitions=10;
	SELECT page, count(*) c
	FROM logs_last_month_cached
	GROUP BY page ORDER BY c DESC LIMIT 10;
	
You may also put this property in hive-site.xml to override the default value.

For now, the mapred.reduce.tasks property is still recognized, and is converted to spark.sql.shuffle.partitions automatically.

### Caching

The shark.cache table property no longer exists, and tables whose name end with _cached are no longer automatically cached. Instead, we provide CACHE TABLE and UNCACHE TABLE statements to let user control table caching explicitly:

	CACHE TABLE logs_last_month;
	UNCACHE TABLE logs_last_month;
__NOTE__: CACHE TABLE tbl is now eager by default not lazy. Don’t need to trigger cache materialization manually anymore.

Spark SQL newly introduced a statement to let user control table caching whether or not lazy since Spark 1.2.0:

	CACHE [LAZY] TABLE [AS SELECT] ...
	
Several caching related features are not supported yet:

* User defined partition level cache eviction policy
* RDD reloading
* In-memory cache write through policy

### Compatibility with Apache Hive

Spark SQL is designed to be compatible with the Hive Metastore, SerDes and UDFs. Currently Hive SerDes and UDFs are based on Hive 1.2.1, and Spark SQL can be connected to different versions of Hive Metastore (from 0.12.0 to 1.2.1. Also see http://spark.apache.org/docs/latest/sql-programming-guide.html#interacting-with-different-versions-of-hive-metastore).

#### Deploying in Existing Hive Warehouses

The Spark SQL Thrift JDBC server is designed to be “out of the box” compatible with existing Hive installations. You do not need to modify your existing Hive Metastore or change the data placement or partitioning of your tables.

### Supported Hive Features

Spark SQL supports the vast majority of Hive features, such as:

* Hive query statements, including:
 - SELECT
 - GROUP BY
 - ORDER BY
 - CLUSTER BY
 - SORT BY
* All Hive operators, including:
 - Relational operators (=, ⇔, ==, <>, <, >, >=, <=, etc)
 - Arithmetic operators (+, -, *, /, %, etc)
 - Logical operators (AND, &&, OR, ||, etc)
 - Complex type constructors
 - Mathematical functions (sign, ln, cos, etc)
 - String functions (instr, length, printf, etc)
* User defined functions (UDF)
* User defined aggregation functions (UDAF)
* User defined serialization formats (SerDes)
* Window functions
* Joins
 - JOIN
 - {LEFT|RIGHT|FULL} OUTER JOIN
 - LEFT SEMI JOIN
 - CROSS JOIN
* Unions
* Sub-queries
 - SELECT col FROM ( SELECT a + b AS col from t1) t2
* Sampling
* Explain
* Partitioned tables including dynamic partition insertion
* View
* All Hive DDL Functions, including:
 - CREATE TABLE
 - CREATE TABLE AS SELECT
 - ALTER TABLE
* Most Hive Data types, including:
  - TINYINT
  - SMALLINT
  - INT
  - BIGINT
  - BOOLEAN
  - FLOAT
  - DOUBLE
  - STRING
  - BINARY
  - TIMESTAMP
  - DATE
  - ARRAY<>
  - MAP<>
  - STRUCT<>

### Unsupported Hive Functionality

Below is a list of Hive features that we don’t support yet. Most of these features are rarely used in Hive deployments.

#### Major Hive Features

* Tables with buckets: bucket is the hash partitioning within a Hive table partition. Spark SQL doesn’t support buckets yet.

#### Esoteric Hive Features

* UNION type
* Unique join
* Column statistics collecting: Spark SQL does not piggyback scans to collect column statistics at the moment and only supports populating the sizeInBytes field of the hive metastore.

#### Hive Input/Output Formats

* File format for CLI: For results showing back to the CLI, Spark SQL only supports TextOutputFormat.
* Hadoop archive

#### Hive Optimizations

A handful of Hive optimizations are not yet included in Spark. Some of these (such as indexes) are less important due to Spark SQL’s in-memory computational model. Others are slotted for future releases of Spark SQL.

* Block level bitmap indexes and virtual columns (used to build indexes)
* Automatically determine the number of reducers for joins and groupbys: Currently in Spark SQL, you need to control the degree of parallelism post-shuffle using “SET spark.sql.shuffle.partitions=[num_tasks];”.
* Meta-data only query: For queries that can be answered by using only meta data, Spark SQL still launches tasks to compute the result.
* Skew data flag: Spark SQL does not follow the skew data flags in Hive.
* STREAMTABLE hint in join: Spark SQL does not follow the STREAMTABLE hint.
* Merge multiple small files for query results: if the result output contains multiple small files, Hive can optionally merge the small files into fewer large files to avoid overflowing the HDFS metadata. Spark SQL does not support that.


## Reference
### Data Types

Spark SQL and DataFrames support the following data types:

* Numeric types
 - ByteType: Represents 1-byte signed integer numbers. The range of numbers is from -128 to 127.
 - ShortType: Represents 2-byte signed integer numbers. The range of numbers is from -32768 to 32767.
 - IntegerType: Represents 4-byte signed integer numbers. The range of numbers is from -2147483648 to 2147483647.
 - LongType: Represents 8-byte signed integer numbers. The range of numbers is from -9223372036854775808 to 9223372036854775807.
 - FloatType: Represents 4-byte single-precision floating point numbers.
 - DoubleType: Represents 8-byte double-precision floating point numbers.
 - DecimalType: Represents arbitrary-precision signed decimal numbers. Backed internally by java.math.BigDecimal. A BigDecimal consists of an arbitrary precision integer unscaled value and a 32-bit integer scale.
* String type
 - StringType: Represents character string values.
* Binary type
 - BinaryType: Represents byte sequence values.
* Boolean type
 - BooleanType: Represents boolean values.
* Datetime type
 - TimestampType: Represents values comprising values of fields year, month, day, hour, minute, and second.
 - DateType: Represents values comprising values of fields year, month, day.
* Complex types
 - ArrayType(elementType, containsNull): Represents values comprising a sequence of elements with the type of elementType. containsNull is used to indicate if elements in a ArrayType value can have null values.
 - MapType(keyType, valueType, valueContainsNull): Represents values comprising a set of key-value pairs. The data type of keys are described by keyType and the data type of values are described by valueType. For a MapType value, keys are not allowed to have null values. valueContainsNull is used to indicate if values of a MapType value can have null values.
 - StructType(fields): Represents values with the structure described by a sequence of StructFields (fields).
   * StructField(name, dataType, nullable): Represents a field in a StructType. The name of a field is indicated by name. The data type of a field is indicated by dataType. nullable is used to indicate if values of this fields can have null values.

__python__

All data types of Spark SQL are located in the package of pyspark.sql.types. You can access them by doing

	from pyspark.sql.types import *

<table class="table"><tbody><tr>  <th style="width:20%">Data type</th>  <th style="width:40%">Value type in Python</th>  <th>API to access or create a data type</th></tr><tr>  <td> <b>ByteType</b> </td>  <td>  int or long <br>  <b>Note:</b> Numbers will be converted to 1-byte signed integer numbers at runtime.  Please make sure that numbers are within the range of -128 to 127.  </td>  <td>  ByteType()  </td></tr><tr>  <td> <b>ShortType</b> </td>  <td>  int or long <br>  <b>Note:</b> Numbers will be converted to 2-byte signed integer numbers at runtime.  Please make sure that numbers are within the range of -32768 to 32767.  </td>  <td>  ShortType()  </td></tr><tr>  <td> <b>IntegerType</b> </td>  <td> int or long </td>  <td>  IntegerType()  </td></tr><tr>  <td> <b>LongType</b> </td>  <td>  long <br>  <b>Note:</b> Numbers will be converted to 8-byte signed integer numbers at runtime.  Please make sure that numbers are within the range of  -9223372036854775808 to 9223372036854775807.  Otherwise, please convert data to decimal.Decimal and use DecimalType.  </td>  <td>  LongType()  </td></tr><tr>  <td> <b>FloatType</b> </td>  <td>  float <br>  <b>Note:</b> Numbers will be converted to 4-byte single-precision floating  point numbers at runtime.  </td>  <td>  FloatType()  </td></tr><tr>  <td> <b>DoubleType</b> </td>  <td> float </td>  <td>  DoubleType()  </td></tr><tr>  <td> <b>DecimalType</b> </td>  <td> decimal.Decimal </td>  <td>  DecimalType()  </td></tr><tr>  <td> <b>StringType</b> </td>  <td> string </td>  <td>  StringType()  </td></tr><tr>  <td> <b>BinaryType</b> </td>  <td> bytearray </td>  <td>  BinaryType()  </td></tr><tr>  <td> <b>BooleanType</b> </td>  <td> bool </td>  <td>  BooleanType()  </td></tr><tr>  <td> <b>TimestampType</b> </td>  <td> datetime.datetime </td>  <td>  TimestampType()  </td></tr><tr>  <td> <b>DateType</b> </td>  <td> datetime.date </td>  <td>  DateType()  </td></tr><tr>  <td> <b>ArrayType</b> </td>  <td> list, tuple, or array </td>  <td>  ArrayType(<i>elementType</i>, [<i>containsNull</i>])<br>  <b>Note:</b> The default value of <i>containsNull</i> is <i>True</i>.  </td></tr><tr>  <td> <b>MapType</b> </td>  <td> dict </td>  <td>  MapType(<i>keyType</i>, <i>valueType</i>, [<i>valueContainsNull</i>])<br>  <b>Note:</b> The default value of <i>valueContainsNull</i> is <i>True</i>.  </td></tr><tr>  <td> <b>StructType</b> </td>  <td> list or tuple </td>  <td>  StructType(<i>fields</i>)<br>  <b>Note:</b> <i>fields</i> is a Seq of StructFields. Also, two fields with the same  name are not allowed.  </td></tr><tr>  <td> <b>StructField</b> </td>  <td> The value type in Python of the data type of this field  (For example, Int for a StructField with the data type IntegerType) </td>  <td>  StructField(<i>name</i>, <i>dataType</i>, <i>nullable</i>)  </td></tr></tbody></table>

### NaN Semantics

There is specially handling for not-a-number (NaN) when dealing with float or double types that does not exactly match standard floating point semantics. Specifically:

* NaN = NaN returns true.
* In aggregations all NaN values are grouped together.
* NaN is treated as a normal value in join keys.
* NaN values go last when in ascending order, larger than any other numeric value.
