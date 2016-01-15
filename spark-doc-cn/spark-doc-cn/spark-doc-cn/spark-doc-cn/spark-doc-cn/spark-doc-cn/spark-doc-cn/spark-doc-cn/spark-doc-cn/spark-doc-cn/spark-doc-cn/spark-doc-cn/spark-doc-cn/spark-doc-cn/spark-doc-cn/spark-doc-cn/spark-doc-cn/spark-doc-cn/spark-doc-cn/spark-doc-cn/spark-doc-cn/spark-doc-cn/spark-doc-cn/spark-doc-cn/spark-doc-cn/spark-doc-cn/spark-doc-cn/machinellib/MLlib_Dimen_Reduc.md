# MLlib - Dimensionality Reduction
* Singular value decomposition (SVD)
  - Performance
  - SVD Example
* Principal component analysis (PCA)

Dimensionality reduction is the process of reducing the number of variables under consideration. It can be used to extract latent features from raw and noisy features or compress data while maintaining the structure. MLlib provides support for dimensionality reduction on the RowMatrix class.

## Singular value decomposition (SVD)
Singular value decomposition (SVD) factorizes a matrix into three matrices: U, Σ, and V such that

A=UΣVT,

where

* U is an orthonormal matrix, whose columns are called left singular vectors,
* Σ is a diagonal matrix with non-negative diagonals in descending order, whose diagonals are called singular values,
* V is an orthonormal matrix, whose columns are called right singular vectors.

For large matrices, usually we don’t need the complete factorization but only the top singular values and its associated singular vectors. This can save storage, de-noise and recover the low-rank structure of the matrix.

If we keep the top k singular values, then the dimensions of the resulting low-rank matrix will be:

* U: m×k,
* Σ: k×k,
* V: n×k.

### Performance

We assume n is smaller than m. The singular values and the right singular vectors are derived from the eigenvalues and the eigenvectors of the Gramian matrix ATA. The matrix storing the left singular vectors U, is computed via matrix multiplication as U=A(VS−1), if requested by the user via the computeU parameter. The actual method to use is determined automatically based on the computational cost:

* If n is small (n<100) or k is large compared with n (k>n/2), we compute the Gramian matrix first and then compute its top eigenvalues and eigenvectors locally on the driver. This requires a single pass with O(n2) storage on each executor and on the driver, and O(n2k) time on the driver.
* Otherwise, we compute (ATA)v in a distributive way and send it to ARPACK to compute (ATA)’s top eigenvalues and eigenvectors on the driver node. This requires O(k) passes, O(n) storage on each executor, and O(nk) storage on the driver.

### SVD Example

MLlib provides SVD functionality to row-oriented matrices, provided in the RowMatrix class.

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

### Principal component analysis (PCA)

Principal component analysis (PCA) is a statistical method to find a rotation such that the first coordinate has the largest variance possible, and each succeeding coordinate in turn has the largest variance possible. The columns of the rotation matrix are called principal components. PCA is used widely in dimensionality reduction.

MLlib supports PCA for tall-and-skinny matrices stored in row-oriented format and any Vectors.

### Scala

The following code demonstrates how to compute principal components on a RowMatrix and use them to project the vectors into a low-dimensional space.

	import org.apache.spark.mllib.linalg.Matrix
	import org.apache.spark.mllib.linalg.distributed.RowMatrix
	
	val mat: RowMatrix = ...
	
	// Compute the top 10 principal components.
	val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
	
	// Project the rows to the linear space spanned by the top 10 principal components.
	val projected: RowMatrix = mat.multiply(pc)
The following code demonstrates how to compute principal components on source vectors and use them to project the vectors into a low-dimensional space while keeping associated labels:

	import org.apache.spark.mllib.regression.LabeledPoint
	import org.apache.spark.mllib.feature.PCA
	
	val data: RDD[LabeledPoint] = ...
	
	// Compute the top 10 principal components.
	val pca = new PCA(10).fit(data.map(_.features))
	
	// Project vectors to the linear space spanned by the top 10 principal components, keeping the label
	val projected = data.map(p => p.copy(features = pca.transform(p.features)))
In order to run the above application, follow the instructions provided in the Self-Contained Applications section of the Spark quick-start guide. Be sure to also include spark-mllib to your build file as a dependency.