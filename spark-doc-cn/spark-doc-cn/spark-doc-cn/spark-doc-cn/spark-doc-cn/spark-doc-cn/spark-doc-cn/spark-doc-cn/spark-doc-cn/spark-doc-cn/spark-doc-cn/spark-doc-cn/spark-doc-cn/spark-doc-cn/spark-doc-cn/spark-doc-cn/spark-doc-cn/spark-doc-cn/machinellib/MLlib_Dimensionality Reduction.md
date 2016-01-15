# MLlib - Dimensionality Reduction

降维

* Singular value decomposition (SVD)（奇异值分解）
  - Performance
  - SVD Example
* Principal component analysis (PCA)（主成份分析）

Dimensionality reduction is the process of reducing the number of variables under consideration. It can be used to extract latent features from raw and noisy features or compress data while maintaining the structure. MLlib provides support for dimensionality reduction on the RowMatrix class.

降维就是在合理的情况下减少变量的数目。可以被用作提取隐含的数据特征来原始的或者噪音特征或者压缩的数据源中，并且与此同时还保持数据的结构。MLlib在行矩阵类中，提供了支持降维的算法。

## Singular value decomposition (SVD)
Singular value decomposition (SVD) factorizes a matrix into three matrices: U, Σ, and V such that

奇异值分解因式分解一个矩阵为三个矩阵，分别为U, Σ, 和 V。 

A=UΣVT,

where

* U is an orthonormal matrix, whose columns are called left singular vectors,
* Σ is a diagonal matrix with non-negative diagonals in descending order, whose diagonals are called singular values,
* V is an orthonormal matrix, whose columns are called right singular vectors.

* U 是一个标准正交的矩阵，它的列被称为左奇异向量
* Σ 是一个对角线矩阵，且是非负的，降序排列。矩阵中的对角线上的值被称为奇异值
* V 是一个标准的正交矩阵，它的列被称为右奇异值

For large matrices, usually we don’t need the complete factorization but only the top singular values and its associated singular vectors. This can save storage, de-noise and recover the low-rank structure of the matrix.

对于大的矩阵，通常来讲，我们不需要完全的因式分解，但是需要最顶层的奇异值或者它的相关的奇异矩阵。这可以节省存储、de-noise和从矩阵的低秩结构中恢复过来。

If we keep the top k singular values, then the dimensions of the resulting low-rank matrix will be:

如果我们保持最顶层的k奇异值，然后低级的矩阵维数将会是：

* U: m×k,
* Σ: k×k,
* V: n×k.

### Performance

We assume n is smaller than m. The singular values and the right singular vectors are derived from the eigenvalues and the eigenvectors of the Gramian matrix ATA. The matrix storing the left singular vectors U, is computed via matrix multiplication as U=A(VS−1), if requested by the user via the computeU parameter. The actual method to use is determined automatically based on the computational cost:

我们假设n是小于m的。这个奇异值和右边的奇异向量是源自Gramian matrix ATA的特征值和特征向量。这个值是存储了左边的奇异向量U，是通过矩阵相乘来计算得到的。 这个矩阵存储了左奇异向量U，是通过矩阵相乘计算得来的。如果通过计算U参数，被用户要求得到。根据计算的值，这个真实的方法被自动决定的。

* If n is small (n<100) or k is large compared with n (k>n/2), we compute the Gramian matrix first and then compute its top eigenvalues and eigenvectors locally on the driver. This requires a single pass with O(n2) storage on each executor and on the driver, and O(n2k) time on the driver.
* Otherwise, we compute (ATA)v in a distributive way and send it to ARPACK to compute (ATA)’s top eigenvalues and eigenvectors on the driver node. This requires O(k) passes, O(n) storage on each executor, and O(nk) storage on the driver.

* 如果n比较小或者k比n大，我们首先计算Gramian矩阵，然后计算它的最大的特征值和特征向量，在本地驱动程序中。这需要一个单一的通过，O(n2)存储在每一个执行的节点和驱动节点中。O(n2k)在驱动节点中。
* 否则，我们在分布式的方式下计算 (ATA)v ，然后发送到ARPACK来计算ATA的最大的特征值和特征向量，在驱动节点上。这需要O(k)通过，O(n)存储在每一个执行节点，O(nk)存储在驱动节点。

### SVD Example

MLlib provides SVD functionality to row-oriented matrices, provided in the RowMatrix class.

MLlib提供了一个以行为方向的SVD函数矩阵，其在以行为矩阵的类中提供。

### Scala

	import org.apache.spark.mllib.linalg.Matrix
	import org.apache.spark.mllib.linalg.distributed.RowMatrix
	import org.apache.spark.mllib.linalg.SingularValueDecomposition
	
	val mat: RowMatrix = ...
	
	// Compute the top 20 singular values and corresponding singular vectors.
	val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(20, computeU = true)
	val U: RowMatrix = svd.U // The U factor is a RowMatrix.
	val s: Vector = svd.s // The singular values are stored in a local dense vector.
	val V: Matrix = svd.V // The V factor is a local dense matrix.
	
The same code applies to IndexedRowMatrix if U is defined as an IndexedRowMatrix.

如果U被定义为一个以行为索引的矩阵中，相同的代码将会被运用在行矩阵中。

### Principal component analysis (PCA)

Principal component analysis (PCA) is a statistical method to find a rotation such that the first coordinate has the largest variance possible, and each succeeding coordinate in turn has the largest variance possible. The columns of the rotation matrix are called principal components. PCA is used widely in dimensionality reduction.

MLlib supports PCA for tall-and-skinny matrices stored in row-oriented format and any Vectors.

主成份分析是一个统计方法来发现一个循环，第一个坐标有可能性方差。随后的坐标依次拥有最大的可能性方差。旋转矩阵的列被称为主成份。PCA被广泛应用与降维。
MLlib支持“又高又廋”存储在以行为方向的格式中和任何其他向量中的主成份分析。

### Scala

The following code demonstrates how to compute principal components on a RowMatrix and use them to project the vectors into a low-dimensional space.

下面的代码演示了在一个行矩阵的来计算主成份分析，然后使用它们来映射向量到一个低维的空间。

	import org.apache.spark.mllib.linalg.Matrix
	import org.apache.spark.mllib.linalg.distributed.RowMatrix
	
	val mat: RowMatrix = ...
	
	// Compute the top 10 principal components.
	val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
	
	// Project the rows to the linear space spanned by the top 10 principal components.
	val projected: RowMatrix = mat.multiply(pc)
The following code demonstrates how to compute principal components on source vectors and use them to project the vectors into a low-dimensional space while keeping associated labels:

下面的代码演示了在源向量怎样计算主成份分析。使用他们映射一个向量到一个低维的空间，与此同时要保持对应联系的标签。

	import org.apache.spark.mllib.regression.LabeledPoint
	import org.apache.spark.mllib.feature.PCA
	
	val data: RDD[LabeledPoint] = ...
	
	// Compute the top 10 principal components.
	val pca = new PCA(10).fit(data.map(_.features))
	
	// Project vectors to the linear space spanned by the top 10 principal components, keeping the label
	val projected = data.map(p => p.copy(features = pca.transform(p.features)))

In order to run the above application, follow the instructions provided in the Self-Contained Applications section of the Spark quick-start guide. Be sure to also include spark-mllib to your build file as a dependency.

为了能够运行上面的应用，请参考Spark quick-start guide中的Self-Contained应用部分。一定要包括spark-mllib作为一个依赖你的构建文件。