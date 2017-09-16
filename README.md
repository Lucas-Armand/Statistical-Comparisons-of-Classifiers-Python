# Statistical Comparisons of Classifiers in Python

It is a implamentation of famous Demšar's paper "Statistical comparisons of classifiers over multiple data sets" in Python.

In order to present an implementation of the technique, some methods and dataset were chosen.
The methods chosen were: RandoForest, MLP Neural Networks, SVM.
The datasets chosen were: iris, wine, breast cancer. 
In this example the comparison parameter will be AUC. Based on 5x2 crosvalidation result.

# Organization 
The "clsfCompare.py" file is the repository with three functions, which are proposed by Demsar as valid methods for comparing multiple methods in multiple data sets.

The file "examp.py" is an example that can be executed to instill the operation of the code.

# References

In this work we use the python libraries: sklearn (to import learning methods and databases), panda and numpy (for structuring, manipulation and visualization of data)

Demšar, Janez. "Statistical comparisons of classifiers over multiple data sets." Journal of Machine learning research 7.Jan (2006): 1-30.

Dietterich, Thomas G. "Approximate statistical tests for comparing supervised classification learning algorithms." Neural computation 10.7 (1998): 1895-1923.

Numpy - www.numpy.org/

Sklearn - scikit-learn.org

Panda - pandas.pydata.org/

Iris Dataset - https://archive.ics.uci.edu/ml/datasets/iris

Wine Dataset - http://archive.ics.uci.edu/ml/datasets/Wine?ref=datanews.io

Breast Cancer Dataset - https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
