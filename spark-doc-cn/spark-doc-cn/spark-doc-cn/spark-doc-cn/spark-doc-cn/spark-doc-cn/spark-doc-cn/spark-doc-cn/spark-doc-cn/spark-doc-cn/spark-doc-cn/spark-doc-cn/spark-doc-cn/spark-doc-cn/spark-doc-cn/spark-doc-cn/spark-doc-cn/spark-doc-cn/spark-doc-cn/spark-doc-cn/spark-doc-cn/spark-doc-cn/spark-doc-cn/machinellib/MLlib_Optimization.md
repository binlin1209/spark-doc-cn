# MLlib - Optimization(优化)
* Mathematical description
  - Gradient descent
  - Stochastic gradient descent (SGD)
  - Update schemes for distributed SGD
  - Limited-memory BFGS (L-BFGS)
  - Choosing an Optimization Method
* Implementation in MLlib
  - Gradient descent and stochastic gradient descent
  - L-BFGS
* Developer’s notes

## Mathematical description
### Gradient descent（梯度下降法）
The simplest method to solve optimization problems of the form minw∈ℝdf(w) is gradient descent. Such first-order optimization methods (including gradient descent and stochastic variants thereof) are well-suited for large-scale and distributed computation.

解决优化问题的最简单的方法就是梯度下降法。这样一阶优化方法非常适合大规模分布式计算。

Gradient descent methods aim to find a local minimum of a function by iteratively taking steps in the direction of steepest descent, which is the negative of the derivative (called the gradient) of the function at the current point, i.e., at the current parameter value. If the objective function f is not differentiable at all arguments, but still convex, then a sub-gradient is the natural generalization of the gradient, and assumes the role of the step direction. In any case, computing a gradient or sub-gradient of f is expensive — it requires a full pass through the complete dataset, in order to compute the contributions from all loss terms.

梯度下降法主要目的是寻找一个本地最小化方程，通过迭代地采取措施沿着最快速的下降方向。在当前点，函数的导数值为负。如果目标函数f是没有可微的点,但是仍然凸,然后sub-gradient梯度就是一个自然的推广,并假设步骤的作用方向。在任何情况下，计算一个梯度或者f的sub-gradient是很昂贵的。所有的数据集都需要传递的，为了计算所有丢失数据的贡献值。

### Stochastic gradient descent (SGD)
Optimization problems whose objective function f is written as a sum are particularly suitable to be solved using stochastic gradient descent (SGD). In our case, for the optimization formulations commonly used in supervised machine learning,

对于主函数为f的优化问题的求和是非常适合用于解决SGD的。在我们的案例中，监督是机器学习中通常用这种优化方法,

f(w):=λR(w)+1n∑i=1nL(w;xi,yi) .(1)

this is especially natural, because the loss is written as an average of the individual losses coming from each datapoint.

这是非常自然的，因为这种损失被认为是单个个体损失的平均值，单个个体来自每个数据点。

A stochastic subgradient is a randomized choice of a vector, such that in expectation, we obtain a true subgradient of the original objective function. Picking one datapoint i∈[1..n] uniformly at random, we obtain a stochastic subgradient of (1), with respect to w as follows:

SGD方法就是一个向量的随机选择，在预期中，我们获得一个真正的原始主函数的次梯度。随机的选择一个非正式的数据点，我们获得一个随机的次梯度公式，如下所示：

f′w,i:=L′w,i+λR′w ,

where L′w,i∈ℝd is a subgradient of the part of the loss function determined by the i-th datapoint, that is L′w,i∈∂∂wL(w;xi,yi). Furthermore, R′w is a subgradient of the regularizer R(w), i.e. R′w∈∂∂wR(w). The term R′w does not depend on which random datapoint is picked. Clearly, in expectation over the random choice of i∈[1..n], we have that f′w,i is a subgradient of the original objective f, meaning that 𝔼[f′w,i]∈∂∂wf(w).

Running SGD now simply becomes walking in the direction of the negative stochastic subgradient f′w,i, that is

运行SGD现在变成了简单的SGD方向行走了，

w(t+1):=w(t)−γf′w,i .(2)

**Step-size.** The parameter γ is the step-size, which in the default implementation is chosen decreasing with the square root of the iteration counter, i.e. γ:=st√ in the t-th iteration, with the input parameter s= stepSize. Note that selecting the best step-size for SGD methods can often be delicate in practice and is a topic of active research.

**步数：** γ参数就是步数，在默认情况下被用作降低的平方根迭代计数器。输入的参数值s为步数。注意，对于SGD方法选择最好的步数在实践中是非常好的，也是一个非常活跃的研究方向。

**Gradients.** A table of (sub)gradients of the machine learning methods implemented in MLlib, is available in the classification and regression section.

**梯度.** 在MLlib中，用于机器学习的次梯度表是可用的在分类和回归部分。

**Proximal Updates.** As an alternative to just use the subgradient R′(w) of the regularizer in the step direction, an improved update for some cases can be obtained by using the proximal operator instead. For the L1-regularizer, the proximal operator is given by soft thresholding, as implemented in L1Updater.

**近端更新.** 作为替代就是用次梯度R′(w)方向的调整步骤,改进更新某些情况下可以通过使用近端操作符。对于L1-regularizer,近端操作符是由软阈值给定的,如L1Updater实现。

## Update schemes for distributed SGD

The SGD implementation in GradientDescent uses a simple (distributed) sampling of the data examples. We recall that the loss part of the optimization problem (1) is 1n∑ni=1L(w;xi,yi), and therefore 1n∑ni=1L′w,i would be the true (sub)gradient. Since this would require access to the full data set, the parameter miniBatchFraction specifies which fraction of the full data to use instead. The average of the gradients over this subset, i.e.

1|S|∑i∈SL′w,i ,

is a stochastic gradient. Here S is the sampled subset of size |S|= miniBatchFraction ⋅n.

In each iteration, the sampling over the distributed dataset (RDD), as well as the computation of the sum of the partial results from each worker machine is performed by the standard spark routines.

在每一次迭代中，在RDD上的样例和部分结果的总和的计算来自每个工作机器将会被标准的spark路线执行。

If the fraction of points miniBatchFraction is set to 1 (default), then the resulting step in each iteration is exact (sub)gradient descent. In this case there is no randomness and no variance in the used step directions. On the other extreme, if miniBatchFraction is chosen very small, such that only a single point is sampled, i.e. |S|= miniBatchFraction ⋅n=1, then the algorithm is equivalent to standard SGD. In that case, the step direction depends from the uniformly random sampling of the point.

如果在默认情况下，最小批次片断点被设置为1，然后相应的每一个迭代步数是很精确的在梯度递减中。在这种情况下，没有随机性和方差在使用步骤的方向上。在另一个极端，如果miniBatchFraction选择的非常小，在这种情况下，只有一个点将会被采用。当|S|= miniBatchFraction ⋅n=1时，这个算法是与标准的SGD算法是等价的。在这种情况下，步数方向是依靠非正式随机采样点的。

## Limited-memory BFGS (L-BFGS)

L-BFGS is an optimization algorithm in the family of quasi-Newton methods to solve the optimization problems of the form minw∈ℝdf(w). The L-BFGS method approximates the objective function locally as a quadratic without evaluating the second partial derivatives of the objective function to construct the Hessian matrix. The Hessian matrix is approximated by previous gradient evaluations, so there is no vertical scalability issue (the number of training features) when computing the Hessian matrix explicitly in Newton’s method. As a result, L-BFGS often achieves rapider convergence compared with other first-order optimization.

L-BFGS是一个优化函数，在quasi-Newton方法中，是为了解决优化问题来达到minw∈ℝdf(w)。这个L-BFGS方法近似于主函数而没有评估第二局部偏导主函数来构建一个Hessian矩阵。通过之前的梯度评估来得到近似的Hessian矩阵，所以没有垂直的可伸缩性问题，当用Newton方法来计算显示的计算Hessian矩阵的时。结果，比起其它的一阶优化， L-BFGS可以快速的收敛。

## Choosing an Optimization Method

Linear methods use optimization internally, and some linear methods in MLlib support both SGD and L-BFGS. Different optimization methods can have different convergence guarantees depending on the properties of the objective function, and we cannot cover the literature here. In general, when L-BFGS is available, we recommend using it instead of SGD since L-BFGS tends to converge faster (in fewer iterations).

线性方法在内部优化使用，在MLlib中，一些线性方法支持SGD和L-BFGS。不同的优化方法有不同的收敛保证，依据主函数的性质，在这里我们不能掩盖文字。通常而言，当L-BFGS可用时，自从L-BFGS收敛变快时，我们就推荐使用L-BFGS而不是SGD。

## Implementation in MLlib

### Gradient descent and stochastic gradient descent

Gradient descent methods including stochastic subgradient descent (SGD) as included as a low-level primitive in MLlib, upon which various ML algorithms are developed, see the linear methods section for example.

梯度下降方法包括随机次梯度下降法(SGD)包括MLlib中低级原始、以及各种开发的机器学习算法,例如参见线性方法部分。

The SGD class GradientDescent sets the following parameters:

* Gradient is a class that computes the stochastic gradient of the function being optimized, i.e., with respect to a single training example, at the current parameter value. MLlib includes gradient classes for common loss functions, e.g., hinge, logistic, least-squares. The gradient class takes as input a training example, its label, and the current parameter value.
* Updater is a class that performs the actual gradient descent step, i.e. updating the weights in each iteration, for a given gradient of the loss part. The updater is also responsible to perform the update from the regularization part. MLlib includes updaters for cases without regularization, as well as L1 and L2 regularizers.
* stepSize is a scalar value denoting the initial step size for gradient descent. All updaters in MLlib use a step size at the t-th step equal to stepSize /t√.
* numIterations is the number of iterations to run.
* regParam is the regularization parameter when using L1 or L2 regularization.
* miniBatchFraction is the fraction of the total data that is sampled in each iteration, to compute the gradient direction.
  - Sampling still requires a pass over the entire RDD, so decreasing miniBatchFraction may not speed up optimization much. Users will see the greatest speedup when the gradient is expensive to compute, for only the chosen samples are used for computing the gradient.

SGD类梯度递增有下面的参数：

* 梯度是一个类，这个类是用来计算函数的随机梯度优化。例如考虑到单个的训练例子，在当前的参数值中。MLlib包括梯度类常见的损失函数，例如，铰链，回归，最小二乘。梯度类需要输入一个训练的例子，它的标签，和当前参数值。
* 更新程序就是一个执行实际梯度下降步骤的类，即更新每个迭代中中的权值，对损失部分给定的梯度而言。该更新还负责从规则化部分进行更新。 MLlib包括无规则化的更新，以及L1和L2正规化更新。
* 步骤大小是一个标量值对于梯度下降而言表示为初始步长。在MLlib所有更新程序使用的步长在第t步等于步骤大小stepSize /√t。
* numIterations就是所运行的迭代数目
* regParam就是规划参数，当使用L1和L2正规化时。
* miniBatchFraction就是所有数据的片断。这些数据是从每次迭代中采样的，用来计算梯度方向。
   - 采样仍然需要传递所有的RDD，所以降低miniBatchFraction可能不会太多的优化。当梯度用于计算时非常昂贵，用户将会看到最大的加速。用于计算梯度的仅仅只是所选择的采样。


## L-BFGS
L-BFGS is currently only a low-level optimization primitive in MLlib. If you want to use L-BFGS in various ML algorithms such as Linear Regression, and Logistic Regression, you have to pass the gradient of objective function, and updater into optimizer yourself instead of using the training APIs like LogisticRegressionWithSGD. See the example below. It will be addressed in the next release.

在MLlib中，L-BFGS是当前的一个低层次的原始优化方法。如果你想使用L-BFGS在不同的机器学习算法中，比如线性回归和罗辑回归，你需要传递主函数的梯度。然后更新本身的优化方法，而不是使用已经训练好的APIs如LogisticRegressionWithSGD。请参考下面的例子，这将在未来的版本中得到解决。

The L1 regularization by using L1Updater will not work since the soft-thresholding logic in L1Updater is designed for gradient descent. See the developer’s note.

通过使用L1Updater，L1规则化将不会工作，自从L1Updater中的软阈值被用来设计为梯度递减。

The L-BFGS method LBFGS.runLBFGS has the following parameters:

* Gradient is a class that computes the gradient of the objective function being optimized, i.e., with respect to a single training example, at the current parameter value. MLlib includes gradient classes for common loss functions, e.g., hinge, logistic, least-squares. The gradient class takes as input a training example, its label, and the current parameter value.
* Updater is a class that computes the gradient and loss of objective function of the regularization part for L-BFGS. MLlib includes updaters for cases without regularization, as well as L2 regularizer.
* numCorrections is the number of corrections used in the L-BFGS update. 10 is recommended.
* maxNumIterations is the maximal number of iterations that L-BFGS can be run.
* regParam is the regularization parameter when using regularization.
* convergenceTol controls how much relative change is still allowed when L-BFGS is considered to converge. This must be nonnegative. Lower values are less tolerant and therefore generally cause more iterations to be run. This value looks at both average improvement and the norm of gradient inside Breeze LBFGS.

L-BFGS方法中的LBFGS.runLBFGS有下面的参数:

* 梯度是一个类，这个类是用来计算函数的随机梯度优化。例如考虑到单个的训练例子，在当前的参数值中。MLlib包括梯度类常见的损失函数，例如，铰链，回归，最小二乘。梯度类需要输入一个训练的例子，它的标签，和当前参数值。
* 更新程序是一个类，用来计算梯度和失去的规则化主函数部分。对于没有规则化的事件而言，MLlib包括更新主程序和L2正则化矩阵。
* 在L-BFGS更新中，numCorrections代表着纠正的数目。通常的推荐值是10.
* maxNumIterations代表着L-BFGS能够运行的最大的迭代数目。
* 当使用正则化方法时，regParam代表着正则化参数
* convergenceTol代表着多少相关改变将被允许，当L-BFGS被考虑用作收敛时。这个值必须是非负的。较低的值误差比较小,因此通常导致更多的迭代运行。这个值监督着平均值的改善和内部LBFGS的梯度规则。

The return is a tuple containing two elements. The first element is a column matrix containing weights for every feature, and the second element is an array containing the loss computed for every iteration.

这个返回值是一个元祖包含两个元素。第一个元素就是列矩阵包括每个特征的权值，第二个元素是一个数组包括每次迭代的丢失值。

Here is an example to train binary logistic regression with L2 regularization using L-BFGS optimizer.

下面的例子是用L2正则化来训练二进制线性回归的，使用 L-BFGS优化方法。

### Scala

	import org.apache.spark.SparkContext
	import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
	import org.apache.spark.mllib.linalg.Vectors
	import org.apache.spark.mllib.util.MLUtils
	import org.apache.spark.mllib.classification.LogisticRegressionModel
	import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
	
	val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
	val numFeatures = data.take(1)(0).features.size
	
	// Split data into training (60%) and test (40%).
	val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
	
	// Append 1 into the training data as intercept.
	val training = splits(0).map(x => (x.label, MLUtils.appendBias(x.features))).cache()
	
	val test = splits(1)
	
	// Run training algorithm to build the model
	val numCorrections = 10
	val convergenceTol = 1e-4
	val maxNumIterations = 20
	val regParam = 0.1
	val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))
	
	val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
	  training,
	  new LogisticGradient(),
	  new SquaredL2Updater(),
	  numCorrections,
	  convergenceTol,
	  maxNumIterations,
	  regParam,
	  initialWeightsWithIntercept)
	
	val model = new LogisticRegressionModel(
	  Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
	  weightsWithIntercept(weightsWithIntercept.size - 1))
	
	// Clear the default threshold.
	model.clearThreshold()
	
	// Compute raw scores on the test set.
	val scoreAndLabels = test.map { point =>
	  val score = model.predict(point.features)
	  (score, point.label)
	}
	
	// Get evaluation metrics.
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	val auROC = metrics.areaUnderROC()
	
	println("Loss of each step in training process")
	loss.foreach(println)
	println("Area under ROC = " + auROC)

## Developer’s notes

Since the Hessian is constructed approximately from previous gradient evaluations, the objective function can not be changed during the optimization process. As a result, Stochastic L-BFGS will not work naively by just using miniBatch; therefore, we don’t provide this until we have better understanding.

自从Hessisan从之间的梯度评估中被近似的创建，在接下来的优化过程中，主函数是不能改变的。结果，仅仅同时使用miniBatch,随机L-BFGS就不会再工作了。因此，我们不会提供这种方法，直到我们更好的理解了为止。

Updater is a class originally designed for gradient decent which computes the actual gradient descent step. However, we’re able to take the gradient and loss of objective function of regularization for L-BFGS by ignoring the part of logic only for gradient decent such as adaptive step size stuff. We will refactorize this into regularizer to replace updater to separate the logic between regularization and step update later.

Updater就是一个类，用来设计为梯度递减，并用来计算实际的梯度下降步骤。然而，我们能够采取目标函数的梯度和损失的正规化主函数L-BFGS。通过忽视一部分逻辑的像样的梯度，例如自适应步长。我们将重新因式分解进入正则化来替换更新程序来分离正则化和随后步骤更新的逻辑。
