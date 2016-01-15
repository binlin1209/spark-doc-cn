# MLlib - Evaluation Metrics(评价指标)

* Classification model evaluation
  - Binary classification
     * Threshold tuning
  - Multiclass classification
     * Label based metrics
  - Multilabel classification
  - Ranking systems
* Regression model evaluation

Spark’s MLlib comes with a number of machine learning algorithms that can be used to learn from and make predictions on data. When these algorithms are applied to build machine learning models, there is a need to evaluate the performance of the model on some criteria, which depends on the application and its requirements. Spark’s MLlib also provides a suite of metrics for the purpose of evaluating the performance of machine learning models.

Spark的MLlib带来了许多的机器学习算法，可以用来学习和预测数据。当这些算法被用来建立机器学习模型时，这就需要一些准则来评价这个模型的性能。这主要依赖于它们的应用途径和需要。为了能够评估机器学习的模型，Spark的MLlib也提供了一系列的指标来衡量。

Specific machine learning algorithms fall under broader types of machine learning applications like classification, regression, clustering, etc. Each of these types have well established metrics for performance evaluation and those metrics that are currently available in Spark’s MLlib are detailed in this section.

特殊的机器学习算法属于更广泛的类型的机器学习应用程序，比如分类、回归和聚类。每种类型建立绩效评价指标,在spark MLlib中，这些指标在这部分内容中将被详细说明了。

## Classification model evaluation
While there are many different types of classification algorithms, the evaluation of classification models all share similar principles. In a supervised classification problem, there exists a true output and a model-generated predicted output for each data point. For this reason, the results for each data point can be assigned to one of four categories:

尽管分类算法有很多不同的类型，评价分类算法的模型却共享了相似的准则。在监督式分类问题中，存在一个真正的输出和每个数据点的model-generated预测输出结果。出于这个原因,每个数据点的结果可以分配给四个类别中的一个

* True Positive (TP) - label is positive and prediction is also positive
* True Negative (TN) - label is negative and prediction is also negative
* False Positive (FP) - label is negative but prediction is positive
* False Negative (FN) - label is positive but prediction is negative
* True Positive (TP) - 标签是正的，预测是正的
* True Negative (TN) - 标签是负的，预测是负的
* False Positive (FP) - 标签是负的，预测是正的
* False Negative (FN) - 标签是正的，预测是负的

These four numbers are the building blocks for most classifier evaluation metrics. A fundamental point when considering classifier evaluation is that pure accuracy (i.e. was the prediction correct or incorrect) is not generally a good metric. The reason for this is because a dataset may be highly unbalanced. For example, if a model is designed to predict fraud from a dataset where 95% of the data points are not fraud and 5% of the data points are fraud, then a naive classifier that predicts not fraud, regardless of input, will be 95% accurate. For this reason, metrics like precision and recall are typically used because they take into account the type of error. In most applications there is some desired balance between precision and recall, which can be captured by combining the two into a single metric, called the F-measure.

这四个数字是大多数分类器评价指标的构建块。当考虑分类器评价一个基本观点时，纯粹的精度通常不是一个好的指标。这是因为数据集的原因可能会高度不平衡。例如,如果一个模型是用来预测欺诈从一个数据集,95%的数据点没有欺诈和5%的数据点是欺诈,那么天真的分类器预测不欺诈,不管输入值也将有95%的准确率。出于这个原因,通常使用指标精度和召回,因为他们考虑LE 类型错误。在大多数应用程序中,在精度和召回之间会有一些我们所需的平衡,这可以把两个指标结合成一个单一的指标,称为F-measure。

### Binary classification
Binary classifiers are used to separate the elements of a given dataset into one of two possible groups (e.g. fraud or not fraud) and is a special case of multiclass classification. Most binary classification metrics can be generalized to multiclass classification metrics.

二进制分类器用于分离给定的数据集的元素为两个可能的组中的一个，并且是多类分类的一个特例。大多数二元分类指标可以推广到多类分类指标。

### Threshold tuning

It is import to understand that many classification models actually output a “score” (often times a probability) for each class, where a higher score indicates higher likelihood. In the binary case, the model may output a probability for each class: P(Y=1|X) and P(Y=0|X). Instead of simply taking the higher probability, there may be some cases where the model might need to be tuned so that it only predicts a class when the probability is very high (e.g. only block a credit card transaction if the model predicts fraud with >90% probability). Therefore, there is a prediction threshold which determines what the predicted class will be based on the probabilities that the model outputs.

很多分类模型实际上为每个类输出“分数”，其中较高的分数表示可能性更高。在二分类中，模型回味每个类产生一个概率，如:P(Y=1|X) and P(Y=0|X)。并不是简单的采用较高的概率，在某些可能的情况下，该模型可能需要进行调整，以使其只预测一个类时的概率非常高。因此，有一个预测阈来确定什么预测类将基于概率的模型输出。

Tuning the prediction threshold will change the precision and recall of the model and is an important part of model optimization. In order to visualize how precision, recall, and other metrics change as a function of the threshold it is common practice to plot competing metrics against one another, parameterized by threshold. A P-R curve plots (precision, recall) points for different threshold values, while a receiver operating characteristic, or ROC, curve plots (recall, false positive rate) points.

调整预测门槛将改变模型的精度和召回，这是模型优化的重要组成部分。为了直观地了解精度，召回以及其他指标改变来作为阈值的函数，用来竞争彼此的指标，这是很常见的做法，由阈值参数化。一个P-R曲线图指出不同的阈值，然而一个接收器操作一个特性或者ROC，点曲线图。

### Available metrics

<table class="table">
  <thead>
    <tr><th>Metric</th><th>Definition</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Precision (Postive Predictive Value)</td>
      <td><span class="MathJax_Preview" style="color: inherit;"></span><span class="MathJax" id="MathJax-Element-3-Frame"><nobr><span class="math" id="MathJax-Span-25" role="math" style="width: 8.039em; display: inline-block;"><span style="display: inline-block; position: relative; width: 6.67em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.729em 1000em 3.455em -999.997em); top: -2.795em; left: 0.003em;"><span class="mrow" id="MathJax-Span-26"><span class="mi" id="MathJax-Span-27" style="font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mi" id="MathJax-Span-28" style="font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mi" id="MathJax-Span-29" style="font-family: MathJax_Math-italic;">V<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.182em;"></span></span><span class="mo" id="MathJax-Span-30" style="font-family: MathJax_Main; padding-left: 0.301em;">=</span><span class="mfrac" id="MathJax-Span-31" style="padding-left: 0.301em;"><span style="display: inline-block; position: relative; width: 2.741em; height: 0px; margin-right: 0.122em; margin-left: 0.122em;"><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.402em; left: 50%; margin-left: -0.533em;"><span class="mrow" id="MathJax-Span-32"><span class="mi" id="MathJax-Span-33" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-34" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.229em -999.997em); top: -3.568em; left: 50%; margin-left: -1.307em;"><span class="mrow" id="MathJax-Span-35"><span class="mi" id="MathJax-Span-36" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-37" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mo" id="MathJax-Span-38" style="font-size: 70.7%; font-family: MathJax_Main;">+</span><span class="mi" id="MathJax-Span-39" style="font-size: 70.7%; font-family: MathJax_Math-italic;">F<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-40" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(0.836em 1000em 1.253em -999.997em); top: -1.307em; left: 0.003em;"><span style="border-left-width: 2.741em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.25px; vertical-align: 0.003em;"></span><span style="display: inline-block; width: 0px; height: 1.074em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.801em;"></span></span></span><span style="border-left-width: 0.004em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.789em; vertical-align: -0.639em;"></span></span></nobr></span><script type="math/tex" id="MathJax-Element-3">PPV=\frac{TP}{TP + FP}</script></td>
    </tr>
    <tr>
      <td>Recall (True Positive Rate)</td>
      <td><span class="MathJax_Preview" style="color: inherit;"></span><span class="MathJax" id="MathJax-Element-4-Frame"><nobr><span class="math" id="MathJax-Span-41" role="math" style="width: 11.313em; display: inline-block;"><span style="display: inline-block; position: relative; width: 9.408em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.729em 1000em 3.455em -999.997em); top: -2.795em; left: 0.003em;"><span class="mrow" id="MathJax-Span-42"><span class="mi" id="MathJax-Span-43" style="font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mi" id="MathJax-Span-44" style="font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mi" id="MathJax-Span-45" style="font-family: MathJax_Math-italic;">R</span><span class="mo" id="MathJax-Span-46" style="font-family: MathJax_Main; padding-left: 0.301em;">=</span><span class="mfrac" id="MathJax-Span-47" style="padding-left: 0.301em;"><span style="display: inline-block; position: relative; width: 1.134em; height: 0px; margin-right: 0.122em; margin-left: 0.122em;"><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.402em; left: 50%; margin-left: -0.533em;"><span class="mrow" id="MathJax-Span-48"><span class="mi" id="MathJax-Span-49" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-50" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -3.568em; left: 50%; margin-left: -0.235em;"><span class="mi" id="MathJax-Span-51" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(0.836em 1000em 1.253em -999.997em); top: -1.307em; left: 0.003em;"><span style="border-left-width: 1.134em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.25px; vertical-align: 0.003em;"></span><span style="display: inline-block; width: 0px; height: 1.074em;"></span></span></span></span><span class="mo" id="MathJax-Span-52" style="font-family: MathJax_Main; padding-left: 0.301em;">=</span><span class="mfrac" id="MathJax-Span-53" style="padding-left: 0.301em;"><span style="display: inline-block; position: relative; width: 2.86em; height: 0px; margin-right: 0.122em; margin-left: 0.122em;"><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.402em; left: 50%; margin-left: -0.533em;"><span class="mrow" id="MathJax-Span-54"><span class="mi" id="MathJax-Span-55" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-56" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.229em -999.997em); top: -3.568em; left: 50%; margin-left: -1.366em;"><span class="mrow" id="MathJax-Span-57"><span class="mi" id="MathJax-Span-58" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-59" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mo" id="MathJax-Span-60" style="font-size: 70.7%; font-family: MathJax_Main;">+</span><span class="mi" id="MathJax-Span-61" style="font-size: 70.7%; font-family: MathJax_Math-italic;">F<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-62" style="font-size: 70.7%; font-family: MathJax_Math-italic;">N<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(0.836em 1000em 1.253em -999.997em); top: -1.307em; left: 0.003em;"><span style="border-left-width: 2.86em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.25px; vertical-align: 0.003em;"></span><span style="display: inline-block; width: 0px; height: 1.074em;"></span></span></span></span></span><span style="display: inline-block; width: 0px; height: 2.801em;"></span></span></span><span style="border-left-width: 0.004em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.789em; vertical-align: -0.639em;"></span></span></nobr></span><script type="math/tex" id="MathJax-Element-4">TPR=\frac{TP}{P}=\frac{TP}{TP + FN}</script></td>
    </tr>
    <tr>
      <td>F-measure</td>
      <td><span class="MathJax_Preview" style="color: inherit;"></span><span class="MathJax" id="MathJax-Element-5-Frame"><nobr><span class="math" id="MathJax-Span-63" role="math" style="width: 17.086em; display: inline-block;"><span style="display: inline-block; position: relative; width: 14.229em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.967em 1000em 4.17em -999.997em); top: -3.271em; left: 0.003em;"><span class="mrow" id="MathJax-Span-64"><span class="mi" id="MathJax-Span-65" style="font-family: MathJax_Math-italic;">F<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mo" id="MathJax-Span-66" style="font-family: MathJax_Main;">(</span><span class="mi" id="MathJax-Span-67" style="font-family: MathJax_Math-italic;">β<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span class="mo" id="MathJax-Span-68" style="font-family: MathJax_Main;">)</span><span class="mo" id="MathJax-Span-69" style="font-family: MathJax_Main; padding-left: 0.301em;">=</span><span class="mrow" id="MathJax-Span-70" style="padding-left: 0.301em;"><span class="mo" id="MathJax-Span-71" style="vertical-align: 0.003em;"><span style="font-family: MathJax_Size1;">(</span></span><span class="mn" id="MathJax-Span-72" style="font-family: MathJax_Main;">1</span><span class="mo" id="MathJax-Span-73" style="font-family: MathJax_Main; padding-left: 0.241em;">+</span><span class="msubsup" id="MathJax-Span-74" style="padding-left: 0.241em;"><span style="display: inline-block; position: relative; width: 1.074em; height: 0px;"><span style="position: absolute; clip: rect(3.098em 1000em 4.348em -999.997em); top: -3.985em; left: 0.003em;"><span class="mi" id="MathJax-Span-75" style="font-family: MathJax_Math-italic;">β<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; top: -4.342em; left: 0.658em;"><span class="mn" id="MathJax-Span-76" style="font-size: 70.7%; font-family: MathJax_Main;">2</span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span></span><span class="mo" id="MathJax-Span-77" style="vertical-align: 0.003em;"><span style="font-family: MathJax_Size1;">)</span></span></span><span class="mo" id="MathJax-Span-78" style="font-family: MathJax_Main; padding-left: 0.241em;">⋅</span><span class="mrow" id="MathJax-Span-79" style="padding-left: 0.241em;"><span class="mo" id="MathJax-Span-80" style="vertical-align: 0.003em;"><span style="font-family: MathJax_Size2;">(</span></span><span class="mfrac" id="MathJax-Span-81"><span style="display: inline-block; position: relative; width: 4.824em; height: 0px; margin-right: 0.122em; margin-left: 0.122em;"><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.402em; left: 50%; margin-left: -1.664em;"><span class="mrow" id="MathJax-Span-82"><span class="mi" id="MathJax-Span-83" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-84" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-85" style="font-size: 70.7%; font-family: MathJax_Math-italic;">V<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mo" id="MathJax-Span-86" style="font-size: 70.7%; font-family: MathJax_Main;">⋅</span><span class="mi" id="MathJax-Span-87" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-88" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-89" style="font-size: 70.7%; font-family: MathJax_Math-italic;">R</span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.158em 1000em 4.289em -999.997em); top: -3.449em; left: 50%; margin-left: -2.318em;"><span class="mrow" id="MathJax-Span-90"><span class="msubsup" id="MathJax-Span-91"><span style="display: inline-block; position: relative; width: 0.777em; height: 0px;"><span style="position: absolute; clip: rect(3.336em 1000em 4.289em -999.997em); top: -3.985em; left: 0.003em;"><span class="mi" id="MathJax-Span-92" style="font-size: 70.7%; font-family: MathJax_Math-italic;">β<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; top: -4.283em; left: 0.479em;"><span class="mn" id="MathJax-Span-93" style="font-size: 50%; font-family: MathJax_Main;">2</span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span></span><span class="mo" id="MathJax-Span-94" style="font-size: 70.7%; font-family: MathJax_Main;">⋅</span><span class="mi" id="MathJax-Span-95" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-96" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-97" style="font-size: 70.7%; font-family: MathJax_Math-italic;">V<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mo" id="MathJax-Span-98" style="font-size: 70.7%; font-family: MathJax_Main;">+</span><span class="mi" id="MathJax-Span-99" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-100" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-101" style="font-size: 70.7%; font-family: MathJax_Math-italic;">R</span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(0.836em 1000em 1.253em -999.997em); top: -1.307em; left: 0.003em;"><span style="border-left-width: 4.824em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.25px; vertical-align: 0.003em;"></span><span style="display: inline-block; width: 0px; height: 1.074em;"></span></span></span></span><span class="mo" id="MathJax-Span-102" style="vertical-align: 0.003em;"><span style="font-family: MathJax_Size2;">)</span></span></span></span><span style="display: inline-block; width: 0px; height: 3.277em;"></span></span></span><span style="border-left-width: 0.004em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 2.361em; vertical-align: -0.925em;"></span></span></nobr></span><script type="math/tex" id="MathJax-Element-5">F(\beta) = \left(1 + \beta^2\right) \cdot \left(\frac{PPV \cdot TPR}
          {\beta^2 \cdot PPV + TPR}\right)</script></td>
    </tr>
    <tr>
      <td>Receiver Operating Characteristic (ROC)</td>
      <td><span class="MathJax_Preview" style="color: inherit;"></span><span class="MathJax" id="MathJax-Element-6-Frame"><nobr><span class="math" id="MathJax-Span-103" role="math" style="width: 12.86em; display: inline-block;"><span style="display: inline-block; position: relative; width: 10.717em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(2.979em 1000em 6.134em -999.997em); top: -3.985em; left: 0.003em;"><span class="mrow" id="MathJax-Span-104"><span style="display: inline-block; position: relative; width: 10.717em; height: 0px;"><span style="position: absolute; clip: rect(2.979em 1000em 4.527em -999.997em); top: -3.985em; left: 0.003em;"><span class="mi" id="MathJax-Span-105" style="font-family: MathJax_Math-italic;">F<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mi" id="MathJax-Span-106" style="font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mi" id="MathJax-Span-107" style="font-family: MathJax_Math-italic;">R</span><span class="mo" id="MathJax-Span-108" style="font-family: MathJax_Main;">(</span><span class="mi" id="MathJax-Span-109" style="font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mo" id="MathJax-Span-110" style="font-family: MathJax_Main;">)</span><span class="mo" id="MathJax-Span-111" style="font-family: MathJax_Main; padding-left: 0.301em;">=</span><span class="msubsup" id="MathJax-Span-112" style="padding-left: 0.301em;"><span style="display: inline-block; position: relative; width: 1.491em; height: 0px;"><span style="position: absolute; clip: rect(2.979em 1000em 4.467em -999.997em); top: -3.985em; left: 0.003em;"><span class="mo" id="MathJax-Span-113" style="font-family: MathJax_Size1; vertical-align: 0.003em;">∫<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.515em 1000em 4.17em -999.997em); top: -4.521em; left: 0.717em;"><span class="mi" id="MathJax-Span-114" style="font-size: 70.7%; font-family: MathJax_Main;">∞</span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -3.628em; left: 0.479em;"><span class="texatom" id="MathJax-Span-115"><span class="mrow" id="MathJax-Span-116"><span class="mi" id="MathJax-Span-117" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span></span><span class="msubsup" id="MathJax-Span-118" style="padding-left: 0.182em;"><span style="display: inline-block; position: relative; width: 1.074em; height: 0px;"><span style="position: absolute; clip: rect(3.158em 1000em 4.17em -999.997em); top: -3.985em; left: 0.003em;"><span class="mi" id="MathJax-Span-119" style="font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; top: -3.807em; left: 0.658em;"><span class="mn" id="MathJax-Span-120" style="font-size: 70.7%; font-family: MathJax_Main;">0</span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span></span><span class="mo" id="MathJax-Span-121" style="font-family: MathJax_Main;">(</span><span class="mi" id="MathJax-Span-122" style="font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mo" id="MathJax-Span-123" style="font-family: MathJax_Main;">)</span><span class="mspace" id="MathJax-Span-124" style="height: 0.003em; vertical-align: 0.003em; width: 0.182em; display: inline-block; overflow: hidden;"></span><span class="mi" id="MathJax-Span-125" style="font-family: MathJax_Math-italic;">d<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span class="mi" id="MathJax-Span-126" style="font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(2.979em 1000em 4.527em -999.997em); top: -2.378em; left: 0.003em;"><span class="mspace" id="MathJax-Span-127" style="height: 0.003em; vertical-align: 0.003em; width: 0.003em; display: inline-block; overflow: hidden;"></span><span class="mi" id="MathJax-Span-128" style="font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mi" id="MathJax-Span-129" style="font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mi" id="MathJax-Span-130" style="font-family: MathJax_Math-italic;">R</span><span class="mo" id="MathJax-Span-131" style="font-family: MathJax_Main;">(</span><span class="mi" id="MathJax-Span-132" style="font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mo" id="MathJax-Span-133" style="font-family: MathJax_Main;">)</span><span class="mo" id="MathJax-Span-134" style="font-family: MathJax_Main; padding-left: 0.301em;">=</span><span class="msubsup" id="MathJax-Span-135" style="padding-left: 0.301em;"><span style="display: inline-block; position: relative; width: 1.491em; height: 0px;"><span style="position: absolute; clip: rect(2.979em 1000em 4.467em -999.997em); top: -3.985em; left: 0.003em;"><span class="mo" id="MathJax-Span-136" style="font-family: MathJax_Size1; vertical-align: 0.003em;">∫<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.515em 1000em 4.17em -999.997em); top: -4.521em; left: 0.717em;"><span class="mi" id="MathJax-Span-137" style="font-size: 70.7%; font-family: MathJax_Main;">∞</span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -3.628em; left: 0.479em;"><span class="texatom" id="MathJax-Span-138"><span class="mrow" id="MathJax-Span-139"><span class="mi" id="MathJax-Span-140" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span></span><span class="msubsup" id="MathJax-Span-141" style="padding-left: 0.182em;"><span style="display: inline-block; position: relative; width: 1.074em; height: 0px;"><span style="position: absolute; clip: rect(3.158em 1000em 4.17em -999.997em); top: -3.985em; left: 0.003em;"><span class="mi" id="MathJax-Span-142" style="font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; top: -3.807em; left: 0.658em;"><span class="mn" id="MathJax-Span-143" style="font-size: 70.7%; font-family: MathJax_Main;">1</span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span></span><span class="mo" id="MathJax-Span-144" style="font-family: MathJax_Main;">(</span><span class="mi" id="MathJax-Span-145" style="font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mo" id="MathJax-Span-146" style="font-family: MathJax_Main;">)</span><span class="mspace" id="MathJax-Span-147" style="height: 0.003em; vertical-align: 0.003em; width: 0.182em; display: inline-block; overflow: hidden;"></span><span class="mi" id="MathJax-Span-148" style="font-family: MathJax_Math-italic;">d<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span class="mi" id="MathJax-Span-149" style="font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span><span style="border-left-width: 0.004em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 3.504em; vertical-align: -2.425em;"></span></span></nobr></span><script type="math/tex" id="MathJax-Element-6">FPR(T)=\int^\infty_{T} P_0(T)\,dT \\ TPR(T)=\int^\infty_{T} P_1(T)\,dT</script></td>
    </tr>
    <tr>
      <td>Area Under ROC Curve</td>
      <td><span class="MathJax_Preview" style="color: inherit;"></span><span class="MathJax" id="MathJax-Element-7-Frame"><nobr><span class="math" id="MathJax-Span-150" role="math" style="width: 13.039em; display: inline-block;"><span style="display: inline-block; position: relative; width: 10.836em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.61em 1000em 3.396em -999.997em); top: -2.795em; left: 0.003em;"><span class="mrow" id="MathJax-Span-151"><span class="mi" id="MathJax-Span-152" style="font-family: MathJax_Math-italic;">A</span><span class="mi" id="MathJax-Span-153" style="font-family: MathJax_Math-italic;">U<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-154" style="font-family: MathJax_Math-italic;">R</span><span class="mi" id="MathJax-Span-155" style="font-family: MathJax_Math-italic;">O</span><span class="mi" id="MathJax-Span-156" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mo" id="MathJax-Span-157" style="font-family: MathJax_Main; padding-left: 0.301em;">=</span><span class="msubsup" id="MathJax-Span-158" style="padding-left: 0.301em;"><span style="display: inline-block; position: relative; width: 1.134em; height: 0px;"><span style="position: absolute; clip: rect(2.979em 1000em 4.467em -999.997em); top: -3.985em; left: 0.003em;"><span class="mo" id="MathJax-Span-159" style="font-family: MathJax_Size1; vertical-align: 0.003em;">∫<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.521em; left: 0.717em;"><span class="mn" id="MathJax-Span-160" style="font-size: 70.7%; font-family: MathJax_Main;">1</span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -3.628em; left: 0.479em;"><span class="texatom" id="MathJax-Span-161"><span class="mrow" id="MathJax-Span-162"><span class="mn" id="MathJax-Span-163" style="font-size: 70.7%; font-family: MathJax_Main;">0</span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span></span><span class="mfrac" id="MathJax-Span-164" style="padding-left: 0.182em;"><span style="display: inline-block; position: relative; width: 1.134em; height: 0px; margin-right: 0.122em; margin-left: 0.122em;"><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.402em; left: 50%; margin-left: -0.533em;"><span class="mrow" id="MathJax-Span-165"><span class="mi" id="MathJax-Span-166" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-167" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -3.568em; left: 50%; margin-left: -0.235em;"><span class="mi" id="MathJax-Span-168" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(0.836em 1000em 1.253em -999.997em); top: -1.307em; left: 0.003em;"><span style="border-left-width: 1.134em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.25px; vertical-align: 0.003em;"></span><span style="display: inline-block; width: 0px; height: 1.074em;"></span></span></span></span><span class="mi" id="MathJax-Span-169" style="font-family: MathJax_Math-italic;">d<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span class="mrow" id="MathJax-Span-170" style="padding-left: 0.182em;"><span class="mo" id="MathJax-Span-171" style="vertical-align: 0.003em;"><span style="font-family: MathJax_Size1;">(</span></span><span class="mfrac" id="MathJax-Span-172"><span style="display: inline-block; position: relative; width: 1.193em; height: 0px; margin-right: 0.122em; margin-left: 0.122em;"><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.402em; left: 50%; margin-left: -0.533em;"><span class="mrow" id="MathJax-Span-173"><span class="mi" id="MathJax-Span-174" style="font-size: 70.7%; font-family: MathJax_Math-italic;">F<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-175" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -3.568em; left: 50%; margin-left: -0.295em;"><span class="mi" id="MathJax-Span-176" style="font-size: 70.7%; font-family: MathJax_Math-italic;">N<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(0.836em 1000em 1.253em -999.997em); top: -1.307em; left: 0.003em;"><span style="border-left-width: 1.193em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.25px; vertical-align: 0.003em;"></span><span style="display: inline-block; width: 0px; height: 1.074em;"></span></span></span></span><span class="mo" id="MathJax-Span-177" style="vertical-align: 0.003em;"><span style="font-family: MathJax_Size1;">)</span></span></span></span><span style="display: inline-block; width: 0px; height: 2.801em;"></span></span></span><span style="border-left-width: 0.004em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.861em; vertical-align: -0.568em;"></span></span></nobr></span><script type="math/tex" id="MathJax-Element-7">AUROC=\int^1_{0} \frac{TP}{P} d\left(\frac{FP}{N}\right)</script></td>
    </tr>
    <tr>
      <td>Area Under Precision-Recall Curve</td>
      <td><span class="MathJax_Preview" style="color: inherit;"></span><span class="MathJax" id="MathJax-Element-8-Frame"><nobr><span class="math" id="MathJax-Span-178" role="math" style="width: 14.884em; display: inline-block;"><span style="display: inline-block; position: relative; width: 12.384em; height: 0px; font-size: 120%;"><span style="position: absolute; clip: rect(1.61em 1000em 3.455em -999.997em); top: -2.795em; left: 0.003em;"><span class="mrow" id="MathJax-Span-179"><span class="mi" id="MathJax-Span-180" style="font-family: MathJax_Math-italic;">A</span><span class="mi" id="MathJax-Span-181" style="font-family: MathJax_Math-italic;">U<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-182" style="font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span class="mi" id="MathJax-Span-183" style="font-family: MathJax_Math-italic;">R</span><span class="mi" id="MathJax-Span-184" style="font-family: MathJax_Math-italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mo" id="MathJax-Span-185" style="font-family: MathJax_Main; padding-left: 0.301em;">=</span><span class="msubsup" id="MathJax-Span-186" style="padding-left: 0.301em;"><span style="display: inline-block; position: relative; width: 1.134em; height: 0px;"><span style="position: absolute; clip: rect(2.979em 1000em 4.467em -999.997em); top: -3.985em; left: 0.003em;"><span class="mo" id="MathJax-Span-187" style="font-family: MathJax_Size1; vertical-align: 0.003em;">∫<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.122em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.521em; left: 0.717em;"><span class="mn" id="MathJax-Span-188" style="font-size: 70.7%; font-family: MathJax_Main;">1</span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -3.628em; left: 0.479em;"><span class="texatom" id="MathJax-Span-189"><span class="mrow" id="MathJax-Span-190"><span class="mn" id="MathJax-Span-191" style="font-size: 70.7%; font-family: MathJax_Main;">0</span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span></span></span><span class="mfrac" id="MathJax-Span-192" style="padding-left: 0.182em;"><span style="display: inline-block; position: relative; width: 2.741em; height: 0px; margin-right: 0.122em; margin-left: 0.122em;"><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.402em; left: 50%; margin-left: -0.533em;"><span class="mrow" id="MathJax-Span-193"><span class="mi" id="MathJax-Span-194" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-195" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.229em -999.997em); top: -3.568em; left: 50%; margin-left: -1.307em;"><span class="mrow" id="MathJax-Span-196"><span class="mi" id="MathJax-Span-197" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-198" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mo" id="MathJax-Span-199" style="font-size: 70.7%; font-family: MathJax_Main;">+</span><span class="mi" id="MathJax-Span-200" style="font-size: 70.7%; font-family: MathJax_Math-italic;">F<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-201" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(0.836em 1000em 1.253em -999.997em); top: -1.307em; left: 0.003em;"><span style="border-left-width: 2.741em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.25px; vertical-align: 0.003em;"></span><span style="display: inline-block; width: 0px; height: 1.074em;"></span></span></span></span><span class="mi" id="MathJax-Span-202" style="font-family: MathJax_Math-italic;">d<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span class="mrow" id="MathJax-Span-203" style="padding-left: 0.182em;"><span class="mo" id="MathJax-Span-204" style="vertical-align: 0.003em;"><span style="font-family: MathJax_Size1;">(</span></span><span class="mfrac" id="MathJax-Span-205"><span style="display: inline-block; position: relative; width: 1.134em; height: 0px; margin-right: 0.122em; margin-left: 0.122em;"><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -4.402em; left: 50%; margin-left: -0.533em;"><span class="mrow" id="MathJax-Span-206"><span class="mi" id="MathJax-Span-207" style="font-size: 70.7%; font-family: MathJax_Math-italic;">T<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span class="mi" id="MathJax-Span-208" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(3.336em 1000em 4.17em -999.997em); top: -3.568em; left: 50%; margin-left: -0.235em;"><span class="mi" id="MathJax-Span-209" style="font-size: 70.7%; font-family: MathJax_Math-italic;">P<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.063em;"></span></span><span style="display: inline-block; width: 0px; height: 3.991em;"></span></span><span style="position: absolute; clip: rect(0.836em 1000em 1.253em -999.997em); top: -1.307em; left: 0.003em;"><span style="border-left-width: 1.134em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.25px; vertical-align: 0.003em;"></span><span style="display: inline-block; width: 0px; height: 1.074em;"></span></span></span></span><span class="mo" id="MathJax-Span-210" style="vertical-align: 0.003em;"><span style="font-family: MathJax_Size1;">)</span></span></span></span><span style="display: inline-block; width: 0px; height: 2.801em;"></span></span></span><span style="border-left-width: 0.004em; border-left-style: solid; display: inline-block; overflow: hidden; width: 0px; height: 1.932em; vertical-align: -0.639em;"></span></span></nobr></span><script type="math/tex" id="MathJax-Element-8">AUPRC=\int^1_{0} \frac{TP}{TP+FP} d\left(\frac{TP}{P}\right)</script></td>
    </tr>
  </tbody>
</table>

### Examples
### python

The following code snippets illustrate how to load a sample dataset, train a binary classification algorithm on the data, and evaluate the performance of the algorithm by several binary evaluation metrics.

下面的代码段展示出了如何加载样本数据集，训练二元分类算法上的数据，并通过几个二进制评价指标评价该算法的性能。


	from pyspark.mllib.classification import LogisticRegressionWithLBFGS
	from pyspark.mllib.evaluation import BinaryClassificationMetrics
	from pyspark.mllib.regression import LabeledPoint
	from pyspark.mllib.util import MLUtils
	
	# Several of the methods available in scala are currently missing from pyspark
	
	# Load training data in LIBSVM format
	data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_binary_classification_data.txt")
	
	# Split data into training (60%) and test (40%)
	training, test = data.randomSplit([0.6, 0.4], seed = 11L)
	training.cache()
	
	# Run training algorithm to build the model
	model = LogisticRegressionWithLBFGS.train(training)
	
	# Compute raw scores on the test set
	predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))
	
	# Instantiate metrics object
	metrics = BinaryClassificationMetrics(predictionAndLabels)
	
	# Area under precision-recall curve
	print("Area under PR = %s" % metrics.areaUnderPR)
	
	# Area under ROC curve
	print("Area under ROC = %s" % metrics.areaUnderROC)
	
## Multiclass classification
A multiclass classification describes a classification problem where there are M>2 possible labels for each data point (the case where M=2 is the binary classification problem). For example, classifying handwriting samples to the digits 0 to 9, having 10 possible classes.

一种多类分类描述了分类问题，其中有针对每个数据点M>2个可能的标签。例如，分类笔迹样本的数字0到9，具有10个可能的类。

For multiclass metrics, the notion of positives and negatives is slightly different. Predictions and labels can still be positive or negative, but they must be considered under the context of a particular class. Each label and prediction take on the value of one of the multiple classes and so they are said to be positive for their particular class and negative for all other classes. So, a true positive occurs whenever the prediction and the label match, while a true negative occurs when neither the prediction nor the label take on the value of a given class. By this convention, there can be multiple true negatives for a given data sample. The extension of false negatives and false positives from the former definitions of positive and negative labels is straightforward.

对于多重分类指标，肯定和否定的概念是略有不同。预测和标签仍然可以是正或负，但是它们必须的特定类的上下文中加以考虑。每个标签和预测的值的代表了多个类中的一种。所以他们说他们的特定类是正的的和其它的类设置为负的。当标签值与预测值相匹配时，一个正确的正值将会产生。当标签值和年预测值与给定的值不相匹配时，一个真正的负值会产生。按照这种方法，对于给定的数据样本，可以有多个给定的真正的负值。错误负值的拓展，从之间定义的正值和负值标签中得到的错误正值都是很明确的。

### Label based metrics

Opposed to binary classification where there are only two possible labels, multiclass classification problems have many possible labels and so the concept of label-based metrics is introduced. Overall precision measures precision across all labels - the number of times any class was predicted correctly (true positives) normalized by the number of data points. Precision by label considers only one class, and measures the number of time a specific label was predicted correctly normalized by the number of times that label appears in the output.

与只有两个标签的二元分类方法不同的是，多重分类问题有更多可能的标签，所以以标签为基础的指标概念会被引入。整体精度测量所有标签的精度，任何类进行了预测的数据点的数量将会被正确预测。标签的精度仅考虑一个类，所测量的特定的时间标签将会被正确预测出来，通过标签在输出值中的出现次数。

**Available metrics**

### Examples
### python

The following code snippets illustrate how to load a sample dataset, train a multiclass classification algorithm on the data, and evaluate the performance of the algorithm by several multiclass classification evaluation metrics.

下面的代码片段说明了如何加载示例数据集，训练多类分类算法上的数据，并且通过多种多类分类评价指标评价算法的性能。

	from pyspark.mllib.classification import LogisticRegressionWithLBFGS
	from pyspark.mllib.util import MLUtils
	from pyspark.mllib.evaluation import MulticlassMetrics
	
	# Load training data in LIBSVM format
	data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_multiclass_classification_data.txt")
	
	# Split data into training (60%) and test (40%)
	training, test = data.randomSplit([0.6, 0.4], seed = 11L)
	training.cache()
	
	# Run training algorithm to build the model
	model = LogisticRegressionWithLBFGS.train(training, numClasses=3)
	
	# Compute raw scores on the test set
	predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))
	
	# Instantiate metrics object
	metrics = MulticlassMetrics(predictionAndLabels)
	
	# Overall statistics
	precision = metrics.precision()
	recall = metrics.recall()
	f1Score = metrics.fMeasure()
	print("Summary Stats")
	print("Precision = %s" % precision)
	print("Recall = %s" % recall)
	print("F1 Score = %s" % f1Score)
	
	# Statistics by class
	labels = data.map(lambda lp: lp.label).distinct().collect()
	for label in sorted(labels):
	    print("Class %s precision = %s" % (label, metrics.precision(label)))
	    print("Class %s recall = %s" % (label, metrics.recall(label)))
	    print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))
	
	# Weighted stats
	print("Weighted recall = %s" % metrics.weightedRecall)
	print("Weighted precision = %s" % metrics.weightedPrecision)
	print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
	print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
	print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)
	
## Multilabel classification
A multilabel classification problem involves mapping each sample in a dataset to a set of class labels. In this type of classification problem, the labels are not mutually exclusive. For example, when classifying a set of news articles into topics, a single article might be both science and politics.

多标签分类问题涉及到数据集中的每个样本映射到了一组类标签。在这种类型的分类问题，标签不是相互排斥的。例如，按照文章的主题来进行分类，一篇文章可能是科学和政治。

Because the labels are not mutually exclusive, the predictions and true labels are now vectors of label sets, rather than vectors of labels. Multilabel metrics, therefore, extend the fundamental ideas of precision, recall, etc. to operations on sets. For example, a true positive for a given class now occurs when that class exists in the predicted set and it exists in the true label set, for a specific data point.

因为标签不是相互排斥的，预测和真标签现在的标签集矢量，而不是标签的载体。多标记的指标，因此，延长的准确率，召回等的基本思路。例如，对于一个特定的数据点，一个给定类的一个真正的正值会发生，当在该类中存在预测设置和存在真标签集时。

**Available metrics**

### Examples

The following code snippets illustrate how to evaluate the performance of a multilabel classifer. The examples use the fake prediction and label data for multilabel classification that is shown below.

下面的代码段示出了如何评估多标签分类器的性能。实施例使用的多标记分类的假预测和标签数据如下所示：

Document predictions:

* doc 0 - predict 0, 1 - class 0, 2
* doc 1 - predict 0, 2 - class 0, 1
* doc 2 - predict none - class 0
* doc 3 - predict 2 - class 2
* doc 4 - predict 2, 0 - class 2, 0
* doc 5 - predict 0, 1, 2 - class 0, 1
* doc 6 - predict 1 - class 1, 2

Predicted classes:

* class 0 - doc 0, 1, 4, 5 (total 4)
* class 1 - doc 0, 5, 6 (total 3)
* class 2 - doc 1, 3, 4, 5 (total 4)

True classes:

* class 0 - doc 0, 1, 2, 4, 5 (total 5)
* class 1 - doc 1, 5, 6 (total 3)
* class 2 - doc 0, 3, 4, 6 (total 4)

### python

	from pyspark.mllib.evaluation import MultilabelMetrics
	
	scoreAndLabels = sc.parallelize([
	    ([0.0, 1.0], [0.0, 2.0]),
	    ([0.0, 2.0], [0.0, 1.0]),
	    ([], [0.0]),
	    ([2.0], [2.0]),
	    ([2.0, 0.0], [2.0, 0.0]),
	    ([0.0, 1.0, 2.0], [0.0, 1.0]),
	    ([1.0], [1.0, 2.0])])
	
	# Instantiate metrics object
	metrics = MultilabelMetrics(scoreAndLabels)
	
	# Summary stats
	print("Recall = %s" % metrics.recall())
	print("Precision = %s" % metrics.precision())
	print("F1 measure = %s" % metrics.f1Measure())
	print("Accuracy = %s" % metrics.accuracy)
	
	# Individual label stats
	labels = scoreAndLabels.flatMap(lambda x: x[1]).distinct().collect()
	for label in labels:
	    print("Class %s precision = %s" % (label, metrics.precision(label)))
	    print("Class %s recall = %s" % (label, metrics.recall(label)))
	    print("Class %s F1 Measure = %s" % (label, metrics.f1Measure(label)))
	
	# Micro stats
	print("Micro precision = %s" % metrics.microPrecision)
	print("Micro recall = %s" % metrics.microRecall)
	print("Micro F1 measure = %s" % metrics.microF1Measure)
	
	# Hamming loss
	print("Hamming loss = %s" % metrics.hammingLoss)
	
	# Subset accuracy
	print("Subset accuracy = %s" % metrics.subsetAccuracy)
	
## Ranking systems
The role of a ranking algorithm (often thought of as a recommender system) is to return to the user a set of relevant items or documents based on some training data. The definition of relevance may vary and is usually application specific. Ranking system metrics aim to quantify the effectiveness of these rankings or recommendations in various contexts. Some metrics compare a set of recommended documents to a ground truth set of relevant documents, while other metrics may incorporate numerical ratings explicitly.

排名算法（通常被认为是一个推荐系统）的作用是返回给用户一个基于一定的训练数据集的相关条款或文件。相关性的定义可以变化，并且通常是应用特有的。排名系统指标的目的是量化在各种情况下这些排名或建议的有效性。有些指标比较一组推荐文档地面真值集相关文件，而其他指标可以包含数字收视明确。

### Available metrics

A ranking system usually deals with a set of M users

U={u0,u1,...,uM−1}

Each user (ui) having a set of N ground truth relevant documents

Di={d0,d1,...,dN−1}

And a list of Q recommended documents, in order of decreasing relevance

Ri=[r0,r1,...,rQ−1]

The goal of the ranking system is to produce the most relevant set of documents for each user. The relevance of the sets and the effectiveness of the algorithms can be measured using the metrics listed below.

排名系统的目标是产生每个用户文档中最相关的集合。集合的相关性和算法的有效性可以使用下面列出的指标进行测定。

It is necessary to define a function which, provided a recommended document and a set of ground truth relevant documents, returns a relevance score for the recommended document.

有必要定义一个函数，它提供了一个建议的文件和一组基础事实有关的文件，为推荐文件返回一个相关分数。

relD(r)={10if r∈D,otherwise.

### Examples

The following code snippets illustrate how to load a sample dataset, train an alternating least squares recommendation model on the data, and evaluate the performance of the recommender by several ranking metrics. A brief summary of the methodology is provided below.

下面的代码段示出了如何加载样本数据集，训练交替的最小二乘推荐模型上的数据，并通过几个排名度量评价推荐器的性能。下面提供的方法提供了简要总结。

MovieLens ratings are on a scale of 1-5:

* 5: Must see
* 4: Will enjoy
* 3: It’s okay
* 2: Fairly bad
* 1: Awful

So we should not recommend a movie if the predicted rating is less than 3. To map ratings to confidence scores, we use:

所以，我们不应该推荐一部预测评级小于3的电影。为了映射评级数和信心分数，我们使用:

* 5 -> 2.5
* 4 -> 1.5
* 3 -> 0.5
* 2 -> -0.5
* 1 -> -1.5.

This mappings means unobserved entries are generally between It’s okay and Fairly bad. The semantics of 0 in this expanded world of non-positive weights are “the same as never having interacted at all.”

这种映射意味着不可观测的项目通常在应该推荐的和不应该推荐的之间。 在非正数权值这一扩大的世界的语义中，0意味着是“一样永远不必互动可言”

### python

	from pyspark.mllib.recommendation import ALS, Rating
	from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
	
	#  Read in the ratings data
	lines = sc.textFile("data/mllib/sample_movielens_data.txt")
	
	def parseLine(line):
	    fields = line.split("::")
	    return Rating(int(fields[0]), int(fields[1]), float(fields[2]) - 2.5)
	ratings = lines.map(lambda r: parseLine(r))
	
	# Train a model on to predict user-product ratings
	model = ALS.train(ratings, 10, 10, 0.01)
	
	# Get predicted ratings on all existing user-product pairs
	testData = ratings.map(lambda p: (p.user, p.product))
	predictions = model.predictAll(testData).map(lambda r: ((r.user, r.product), r.rating))
	
	ratingsTuple = ratings.map(lambda r: ((r.user, r.product), r.rating))
	scoreAndLabels = predictions.join(ratingsTuple).map(lambda tup: tup[1])
	
	# Instantiate regression metrics to compare predicted and actual ratings
	metrics = RegressionMetrics(scoreAndLabels)
	
	# Root mean sqaured error
	print("RMSE = %s" % metrics.rootMeanSquaredError)
	
	# R-squared
	print("R-squared = %s" % metrics.r2)
	
## Regression model evaluation
Regression analysis is used when predicting a continuous output variable from a number of independent variables.

当从一些自变量中预测连续输出变量时，回归分析会被使用。

### Available metrics

### Examples
### python

The following code snippets illustrate how to load a sample dataset, train a linear regression algorithm on the data, and evaluate the performance of the algorithm by several regression metrics.

下面的代码段示出了如何加载样本数据集，训练线性回归算法上的数据，并评估该算法的几个回归度量的性能。

	from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
	from pyspark.mllib.evaluation import RegressionMetrics
	from pyspark.mllib.linalg import DenseVector
	
	# Load and parse the data
	def parsePoint(line):
	    values = line.split()
	    return LabeledPoint(float(values[0]), DenseVector([float(x.split(':')[1]) for x in values[1:]]))
	
	data = sc.textFile("data/mllib/sample_linear_regression_data.txt")
	parsedData = data.map(parsePoint)
	
	# Build the model
	model = LinearRegressionWithSGD.train(parsedData)
	
	# Get predictions
	valuesAndPreds = parsedData.map(lambda p: (float(model.predict(p.features)), p.label))
	
	# Instantiate metrics object
	metrics = RegressionMetrics(valuesAndPreds)
	
	# Squared Error
	print("MSE = %s" % metrics.meanSquaredError)
	print("RMSE = %s" % metrics.rootMeanSquaredError)
	
	# R-squared
	print("R-squared = %s" % metrics.r2)
	
	# Mean absolute error
	print("MAE = %s" % metrics.meanAbsoluteError)
	
	# Explained variance
	print("Explained variance = %s" % metrics.explainedVariance)









