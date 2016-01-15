# MLlib - Data Types
* Local vector
* Labeled point
* Local matrix
* Distributed matrix
  - RowMatrix
  - IndexedRowMatrix
  - CoordinateMatrix
  - BlockMatrix

* 本地向量
* 标记的点
* 本地矩阵
* 分布式矩阵
  - 行矩阵
  - 索引行矩阵
  - 坐标矩阵
  - 块矩阵

MLlib supports local vectors and matrices stored on a single machine, as well as distributed matrices backed by one or more RDDs. Local vectors and local matrices are simple data models that serve as public interfaces. The underlying linear algebra operations are provided by Breeze and jblas. A training example used in supervised learning is called a “labeled point” in MLlib.

机器学习包支持存储在单机中的本地向量和矩阵，也支持一个或多个弹性分布式数据集的分布式矩阵。本地向量和矩阵是单一的数据模型，并作为一种公共接口。线性代数矩阵是由Breeze and jblas提供的。在机器学习包中，使用监督学习的一个训练例子被成为标记点。

## Local vector

A local vector has integer-typed and 0-based indices and double-typed values, stored on a single machine. MLlib supports two types of local vectors: dense and sparse. A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values. For example, a vector (1.0, 0.0, 3.0) can be represented in dense format as [1.0, 0.0, 3.0] or in sparse format as (3, [0, 2], [1.0, 3.0]), where 3 is the size of the vector.

本地向量拥有整数型，0基的索引和double型的值，这些值都存储在单一的机器中。MLlib支持两种类型的本地向量：浓密型和稀疏型。一个浓密型的向量由一个double型的数组表示，代表了它的进入值，然而一个稀疏的矩阵是由两个平行的数组组成:索引和值。例如，(1.0, 0.0, 3.0)向量可以表示为稠密的形式如：[1.0, 0.0, 3.0]，也可以表示为稀疏的形式如：(3, [0, 2], [1.0, 3.0])，3是向量的大小。

### python

MLlib recognizes the following types as dense vectors:

* NumPy’s array
* Python’s list, e.g., [1, 2, 3]

MLlib识别下面的类型作为稠密的向量：

* NumPy’s 数组
* Python’s 列表, e.g., [1, 2, 3]


and the following as sparse vectors:

* MLlib’s SparseVector.
* SciPy’s csc_matrix with a single column

和下面的类型作为稀疏的矩阵

* MLlib’s 稀疏矩阵
* 单一列的 SciPy’s csc_matrix

We recommend using NumPy arrays over lists for efficiency, and using the factory methods implemented in Vectors to create sparse vectors.

我们推荐使用numpy的数组作为列表更有效率，使用向量的factory方法来创建稀疏向量。

    import numpy as np
    import scipy.sparse as sps
	from pyspark.mllib.linalg import Vectors
	
	 # Use a NumPy array as a dense vector.
	dv1 = np.array([1.0, 0.0, 3.0])
	 # Use a Python list as a dense vector.
	dv2 = [1.0, 0.0, 3.0]
	 # Create a SparseVector.
	sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])
	 # Use a single-column SciPy csc_matrix as a sparse vector.
	sv2 = sps.csc_matrix((np.array([1.0, 3.0]), np.array([0, 2]), np.array([0, 2])), shape = (3, 1))

## Labeled point

A labeled point is a local vector, either dense or sparse, associated with a label/response. In MLlib, labeled points are used in supervised learning algorithms. We use a double to store a label, so we can use labeled points in both regression and classification. For binary classification, a label should be either 0 (negative) or 1 (positive). For multiclass classification, labels should be class indices starting from zero: 0, 1, 2, ....

一个标记的点就是一个本地向量，或者是稠密的向量或者是稀疏的向量，结合一个标签。在MLlib中，标记的点被用于监督学习算法。我们使用double型来储存标签，所以我们能够在回归和分类中打标签。对于两个类型的分类，标签应该为0或者1。对于多种类型的分类，标签应该从0开始依次为1，2，3 ...

### python

A labeled point is represented by LabeledPoint.

	from pyspark.mllib.linalg import SparseVector
	from pyspark.mllib.regression import LabeledPoint
	
	# Create a labeled point with a positive label and a dense feature vector.
	pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
	
	# Create a labeled point with a negative label and a sparse feature vector.
	neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))
	
	
_Sparse data_

It is very common in practice to have sparse training data. MLlib supports reading training examples stored in LIBSVM format, which is the default format used by LIBSVM and LIBLINEAR. It is a text format in which each line represents a labeled sparse feature vector using the following format:

在实践中用到稀疏训练模型是很常见的。MLlib支持以LIBSVM的格式来读取训练数据，这是使用LIBSVM和LIBLINEAR的默认格式。这是一种文本格式，每一行代表了一个标记的稀疏特征向量。

	label index1:value1 index2:value2 ...
	
where the indices are one-based and in ascending order. After loading, the feature indices are converted to zero-based.

索引是一个以一为基础的，递增的顺序。加载过后，特征索引转变为以0为基础的索引。

### python

MLUtils.loadLibSVMFile reads training examples stored in LIBSVM format.

MLUtils.loadLibSVMFile读取存储在LIBSVM格式中的训练样本

	from pyspark.mllib.util import MLUtils
	
	examples = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	
## Local matrix

A local matrix has integer-typed row and column indices and double-typed values, stored on a single machine. MLlib supports dense matrices, whose entry values are stored in a single double array in column-major order, and sparse matrices, whose non-zero entry values are stored in the Compressed Sparse Column (CSC) format in column-major order. For example, the following dense matrix

>1   2
>
>3   4
>
>5   6

is stored in a one-dimensional array [1.0, 3.0, 5.0, 2.0, 4.0, 6.0] with the matrix size (3, 2).

一个本地的矩阵拥有一个整数型的行和列索引和double型的值，这些数据都存储在单机中。MLlib支持稠密矩阵，它们的进入值都存储在一个double型的排好序的数组中，稀疏矩阵，它们的非零进入值存储在压缩稀疏列中，以列为基的顺序排列。例如下面的稠密矩阵存储在一维数组中，且矩阵大小为(3, 2)。

### python

The base class of local matrices is Matrix, and we provide two implementations: DenseMatrix, and SparseMatrix. We recommend using the factory methods implemented in Matrices to create local matrices. Remember, local matrices in MLlib are stored in column-major order.

当地的矩阵是矩阵的基类，我们提供了两种实施方式：稠密矩阵和稀疏矩阵。我们推荐使用代理方法来创建本地矩阵。记住，在MLlib中，本地矩阵是储存在以列为主的序列中的。

	import org.apache.spark.mllib.linalg.{Matrix, Matrices}
	
	// Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
	dm2 = Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])
	
	// Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
	sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])

## Distributed matrix

A distributed matrix has long-typed row and column indices and double-typed values, stored distributively in one or more RDDs. It is very important to choose the right format to store large and distributed matrices. Converting a distributed matrix to a different format may require a global shuffle, which is quite expensive. Three types of distributed matrices have been implemented so far.

一个分布式的矩阵有长的行和列索引，double型的值，分布式的储存在一个或多个RDDs数据集中。选择正确的形式来储存大的和分布式的矩阵是很重要的。转变一个分布式的矩阵变成不同的形式可能需要一个全局的shuffle，这是很重要的。转变一个分布式的矩阵为一个不同的形式可能需要一个全局的shuffle，这是很昂贵的。

The basic type is called RowMatrix. A RowMatrix is a row-oriented distributed matrix without meaningful row indices, e.g., a collection of feature vectors. It is backed by an RDD of its rows, where each row is a local vector. We assume that the number of columns is not huge for a RowMatrix so that a single local vector can be reasonably communicated to the driver and can also be stored / operated on using a single node. An IndexedRowMatrix is similar to a RowMatrix but with row indices, which can be used for identifying rows and executing joins. A CoordinateMatrix is a distributed matrix stored in coordinate list (COO) format, backed by an RDD of its entries.

这个主要类型被称为行矩阵。一个行矩阵是一个以行为方向分布的矩阵，但是并没有有意义的行索引。例如：一个有特色的向量集合。这个被RDD的行所备份，每一行都是一个本地向量。我们假定列的数目并没有大到容纳一个行矩阵，所以一个本地向量可以合理的与驱动程序进行交互，也能够在单个节点上进行储存和操作。一个索引行矩阵是和行矩阵类似的，但是对于行索引，也能够用来识别行和执行交点。一个坐标矩阵是一个分布式的矩阵，以坐标列表(COO)形式存储的，备份在RDD的入口中。

_Note_

The underlying RDDs of a distributed matrix must be deterministic, because we cache the matrix size. In general the use of non-deterministic RDDs can lead to errors.

底层的抽样分布矩阵必须是确定的，因为我们缓存矩阵的大小。通常来讲，使用非决定的RDDs会导致错误。



## RowMatrix

A RowMatrix is a row-oriented distributed matrix without meaningful row indices, backed by an RDD of its rows, where each row is a local vector. Since each row is represented by a local vector, the number of columns is limited by the integer range but it should be much smaller in practice.

一个行矩阵是一个以行为基的分布式矩阵，但是并没有丰富的行索引，其备份在RDD的行中，每一行都是一个本地向量。因为每一行都代表一个本地向量，列的数目被整数的范围限制了，但是在实践中这个情况出现应该非常小。


### python

A RowMatrix can be created from an RDD of vectors.

一个行矩阵能够被一个RDD向量所创建。

	from pyspark.mllib.linalg.distributed import RowMatrix
	
	# Create an RDD of vectors.
	rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
	
	# Create a RowMatrix from an RDD of vectors.
	mat = RowMatrix(rows)
	
	# Get its size.
	m = mat.numRows()  # 4
	n = mat.numCols()  # 3
	
	# Get the rows as an RDD of vectors again.
	rowsRDD = mat.rows
	
## IndexedRowMatrix

An IndexedRowMatrix is similar to a RowMatrix but with meaningful row indices. It is backed by an RDD of indexed rows, so that each row is represented by its index (long-typed) and a local vector.

一个索引行矩阵和一个有丰富行索引的行矩阵是类似的。它被一个RDD行索引所备份，所以每一行都可以用一个索引和一个本地向量所代表。

### python

An IndexedRowMatrix can be created from an RDD of IndexedRows, where IndexedRow is a wrapper over (long, vector). An IndexedRowMatrix can be converted to a RowMatrix by dropping its row indices.

一个索引行矩阵能够从一个RDD的索引行中创建，索引行包裹着向量。一个索引行矩阵能够转化一个行矩阵通过放弃它的行索引。

from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
	
	# Create an RDD of indexed rows.
	#   - This can be done explicitly with the IndexedRow class:
	indexedRows = sc.parallelize([IndexedRow(0, [1, 2, 3]), 
	                              IndexedRow(1, [4, 5, 6]), 
	                              IndexedRow(2, [7, 8, 9]), 
	                              IndexedRow(3, [10, 11, 12])])
	#   - or by using (long, vector) tuples:
	indexedRows = sc.parallelize([(0, [1, 2, 3]), (1, [4, 5, 6]), 
	                              (2, [7, 8, 9]), (3, [10, 11, 12])])
	
	# Create an IndexedRowMatrix from an RDD of IndexedRows.
	mat = IndexedRowMatrix(indexedRows)
	
	# Get its size.
	m = mat.numRows()  # 4
	n = mat.numCols()  # 3
	
	# Get the rows as an RDD of IndexedRows.
	rowsRDD = mat.rows
	
	# Convert to a RowMatrix by dropping the row indices.
	rowMat = mat.toRowMatrix()
	
	# Convert to a CoordinateMatrix.
	coordinateMat = mat.toCoordinateMatrix()
	
	# Convert to a BlockMatrix.
	blockMat = mat.toBlockMatrix()
	
	
## CoordinateMatrix

A CoordinateMatrix is a distributed matrix backed by an RDD of its entries. Each entry is a tuple of (i: Long, j: Long, value: Double), where i is the row index, j is the column index, and value is the entry value. A CoordinateMatrix should be used only when both dimensions of the matrix are huge and the matrix is very sparse.

一个坐标矩阵就是一个分布式的备份在RDD条目中的矩阵。每一个条目都是一个元组（i: Long, j: Long, value: Double），并且i 是行索引，j是列索引，value是条目的值。一个坐标矩阵应该能够使用，当矩阵的所有维度都足够大时，矩阵是很稀疏。

### python

A CoordinateMatrix can be created from an RDD of MatrixEntry entries, where MatrixEntry is a wrapper over (long, long, float). A CoordinateMatrix can be converted to a RowMatrix by calling toRowMatrix, or to an IndexedRowMatrix with sparse rows by calling toIndexedRowMatrix.

一个坐标矩阵能够从一个RDD的矩阵条目中创建，并且矩阵条目是裹起来的。一个坐标矩阵能够转化为一个行矩阵，这称为toRowMatrix，如果转化为一个稀疏行索引矩阵，这称为toIndexedRowMatrix。

	from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
	
	# Create an RDD of coordinate entries.
	#   - This can be done explicitly with the MatrixEntry class:
	entries = sc.parallelize([MatrixEntry(0, 0, 1.2), MatrixEntry(1, 0, 2.1), MatrixEntry(6, 1, 3.7)])
	#   - or using (long, long, float) tuples:
	entries = sc.parallelize([(0, 0, 1.2), (1, 0, 2.1), (2, 1, 3.7)])
	
	# Create an CoordinateMatrix from an RDD of MatrixEntries.
	mat = CoordinateMatrix(entries)
	
	# Get its size.
	m = mat.numRows()  # 3
	n = mat.numCols()  # 2
	
	# Get the entries as an RDD of MatrixEntries.
	entriesRDD = mat.entries
	
	# Convert to a RowMatrix.
	rowMat = mat.toRowMatrix()
	
	# Convert to an IndexedRowMatrix.
	indexedRowMat = mat.toIndexedRowMatrix()
	
	# Convert to a BlockMatrix.
	blockMat = mat.toBlockMatrix()
	
## BlockMatrix

A BlockMatrix is a distributed matrix backed by an RDD of MatrixBlocks, where a MatrixBlock is a tuple of ((Int, Int), Matrix), where the (Int, Int) is the index of the block, and Matrix is the sub-matrix at the given index with size rowsPerBlock x colsPerBlock. BlockMatrix supports methods such as add and multiply with another BlockMatrix. BlockMatrix also has a helper function validate which can be used to check whether the BlockMatrix is set up properly.

一个块矩阵是一个备份的RDD分布式矩阵，矩阵块是一个元组，整数是矩阵块的索引，在给定的索引下，矩阵是子矩阵。块矩阵支持若干种方法，例如添加或者乘以另一个块矩阵。块矩阵也有一个帮助函数，也可以用来检测块矩阵是否设置正确。

### python

A BlockMatrix can be created from an RDD of sub-matrix blocks, where a sub-matrix block is a ((blockRowIndex, blockColIndex), sub-matrix) tuple.

块矩阵能够从一个RDD的子矩阵创建，哪里一个子矩阵就是一个元组。

	from pyspark.mllib.linalg import Matrices
	from pyspark.mllib.linalg.distributed import BlockMatrix
	
	# Create an RDD of sub-matrix blocks.
	blocks = sc.parallelize([((0, 0), Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])), 
	                         ((1, 0), Matrices.dense(3, 2, [7, 8, 9, 10, 11, 12]))])
	
	# Create a BlockMatrix from an RDD of sub-matrix blocks.
	mat = BlockMatrix(blocks, 3, 2)
	
	# Get its size.
	m = mat.numRows() # 6
	n = mat.numCols() # 2
	
	# Get the blocks as an RDD of sub-matrix blocks.
	blocksRDD = mat.blocks
	
	# Convert to a LocalMatrix.
	localMat = mat.toLocalMatrix()

	# Convert to an IndexedRowMatrix.
	indexedRowMat = mat.toIndexedRowMatrix()
	
	# Convert to a CoordinateMatrix.
	coordinateMat = mat.toCoordinateMatrix()