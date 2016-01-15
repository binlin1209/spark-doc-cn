# MLlib - Classification and Regression

MLlib supports various methods for binary classification, multiclass classification, and regression analysis. The table below outlines the supported algorithms for each type of problem.

MLlib支持不同的方法，有二分类，多重分类和回归分析。下面的表格概述了每种类型所支持的算法。

<table class="table">  <thead>    <tr><th>Problem Type</th><th>Supported Methods</th></tr>  </thead>  <tbody>    <tr>      <td>Binary Classification</td><td>linear SVMs, logistic regression, decision trees, random forests, gradient-boosted trees, naive Bayes</td>    </tr>    <tr>      <td>Multiclass Classification</td><td>logistic regression, decision trees, random forests, naive Bayes</td>    </tr>    <tr>      <td>Regression</td><td>linear least squares, Lasso, ridge regression, decision trees, random forests, gradient-boosted trees, isotonic regression</td>    </tr>  </tbody></table>

More details for these methods can be found here:

* Linear models
  - classification (SVMs, logistic regression)
  - linear regression (least squares, Lasso, ridge)
* Decision trees
* Ensembles of decision trees
  - random forests
  - gradient-boosted trees
* Naive Bayes
* Isotonic regression