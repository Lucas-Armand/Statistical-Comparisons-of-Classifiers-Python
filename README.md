# Statistical Comparisons of Classifiers in Python

It is a implamentation of famous Demšar's paper "Statistical comparisons of classifiers over multiple data sets" in Python. The libraries Scikit-learn, Pandas and Numpy are the main libraries used in this work.

In order to present an implementation of the technique, some methods and dataset were chosen.
The methods chosen were: RandoForest, MLP Neural Networks, SVM.
The datasets chosen were: iris, wine, breast cancer (all from UCI Machine Learning Repository). 
In this example the comparison parameter will be acurracy (based on 5x2 crosvalidation result).


# Organization 

The implementation, presented here, is divided into two codes: The "clfCompare.py" is the main file, in which two functions are proposed ("friedmanTest" and "nemeyinTest"). The second file, "example.py", is a demonstration of how an analysis could be done, its uses the "clfCompare.py" as a library.


# Utilization

To test the implementation simply download the files available in this git in a same folder. and run the file "example.py":

```
python example.py
```
To use the functions 'friedmanTest' and 'nemenymTeste' in some project, simply add the "clfCompare.py" file to your folder and import it into your code:

```python
import clfCompare
```



# Results Analysis

If you run the "example.py", you should see a report like that: 

```
 Iris (UCI) 
Apresentação dos dados (5 primeiras observações):
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2

Propriedades do DataSet:
      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
max            7.900000          4.400000           6.900000          2.500000
mean           5.843333          3.054000           3.758667          1.198667
min            4.300000          2.000000           1.000000          0.100000
std            0.828066          0.433594           1.764420          0.763161

 Wine (UCI) 
Apresentação dos dados (5 primeiras observações):
   alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \
0    14.23        1.71  2.43               15.6        127           2.80   
1    13.20        1.78  2.14               11.2        100           2.65   
2    13.16        2.36  2.67               18.6        101           2.80   
3    14.37        1.95  2.50               16.8        113           3.85   
4    13.24        2.59  2.87               21.0        118           2.80   

   flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \
0        3.06                  0.28             2.29             5.64  1.04   
1        2.76                  0.26             1.28             4.38  1.05   
2        3.24                  0.30             2.81             5.68  1.03   
3        3.49                  0.24             2.18             7.80  0.86   
4        2.69                  0.39             1.82             4.32  1.04   

   od280/od315_of_diluted_wines  proline  
0                          3.92     1065  
1                          3.40     1050  
2                          3.17     1185  
3                          3.45     1480  
4                          2.93      735  

Propriedades do DataSet:
        alcohol  malic_acid       ash  alcalinity_of_ash   magnesium  \
max   14.830000    5.800000  3.230000          30.000000  162.000000   
mean  13.000618    2.336348  2.366517          19.494944   99.741573   
min   11.030000    0.740000  1.360000          10.600000   70.000000   
std    0.811827    1.117146  0.274344           3.339564   14.282484   

      total_phenols  flavanoids  nonflavanoid_phenols  proanthocyanins  \
max        3.880000    5.080000              0.660000         3.580000   
mean       2.295112    2.029270              0.361854         1.590899   
min        0.980000    0.340000              0.130000         0.410000   
std        0.625851    0.998859              0.124453         0.572359   

      color_intensity       hue  od280/od315_of_diluted_wines      proline  
max         13.000000  1.710000                      4.000000  1680.000000  
mean         5.058090  0.957449                      2.611685   746.893258  
min          1.280000  0.480000                      1.270000   278.000000  
std          2.318286  0.228572                      0.709990   314.907474  

 Breast Cancer (UCI) 
Apresentação dos dados (5 primeiras observações):
   mean radius  mean texture  mean perimeter  mean area  mean smoothness  \
0        17.99         10.38          122.80     1001.0          0.11840   
1        20.57         17.77          132.90     1326.0          0.08474   
2        19.69         21.25          130.00     1203.0          0.10960   
3        11.42         20.38           77.58      386.1          0.14250   
4        20.29         14.34          135.10     1297.0          0.10030   

   mean compactness  mean concavity  mean concave points  mean symmetry  \
0           0.27760          0.3001              0.14710         0.2419   
1           0.07864          0.0869              0.07017         0.1812   
2           0.15990          0.1974              0.12790         0.2069   
3           0.28390          0.2414              0.10520         0.2597   
4           0.13280          0.1980              0.10430         0.1809   

   mean fractal dimension           ...             worst radius  \
0                 0.07871           ...                    25.38   
1                 0.05667           ...                    24.99   
2                 0.05999           ...                    23.57   
3                 0.09744           ...                    14.91   
4                 0.05883           ...                    22.54   

   worst texture  worst perimeter  worst area  worst smoothness  \
0          17.33           184.60      2019.0            0.1622   
1          23.41           158.80      1956.0            0.1238   
2          25.53           152.50      1709.0            0.1444   
3          26.50            98.87       567.7            0.2098   
4          16.67           152.20      1575.0            0.1374   

   worst compactness  worst concavity  worst concave points  worst symmetry  \
0             0.6656           0.7119                0.2654          0.4601   
1             0.1866           0.2416                0.1860          0.2750   
2             0.4245           0.4504                0.2430          0.3613   
3             0.8663           0.6869                0.2575          0.6638   
4             0.2050           0.4000                0.1625          0.2364   

   worst fractal dimension  
0                  0.11890  
1                  0.08902  
2                  0.08758  
3                  0.17300  
4                  0.07678  

[5 rows x 30 columns]

Propriedades do DataSet:
      mean radius  mean texture  mean perimeter    mean area  mean smoothness  \
max     28.110000     39.280000      188.500000  2501.000000         0.163400   
mean    14.127292     19.289649       91.969033   654.889104         0.096360   
min      6.981000      9.710000       43.790000   143.500000         0.052630   
std      3.524049      4.301036       24.298981   351.914129         0.014064   

      mean compactness  mean concavity  mean concave points  mean symmetry  \
max           0.345400        0.426800             0.201200       0.304000   
mean          0.104341        0.088799             0.048919       0.181162   
min           0.019380        0.000000             0.000000       0.106000   
std           0.052813        0.079720             0.038803       0.027414   

      mean fractal dimension           ...             worst radius  \
max                 0.097440           ...                36.040000   
mean                0.062798           ...                16.269190   
min                 0.049960           ...                 7.930000   
std                 0.007060           ...                 4.833242   

      worst texture  worst perimeter   worst area  worst smoothness  \
max       49.540000       251.200000  4254.000000          0.222600   
mean      25.677223       107.261213   880.583128          0.132369   
min       12.020000        50.410000   185.200000          0.071170   
std        6.146258        33.602542   569.356993          0.022832   

      worst compactness  worst concavity  worst concave points  \
max            1.058000         1.252000              0.291000   
mean           0.254265         0.272188              0.114606   
min            0.027290         0.000000              0.000000   
std            0.157336         0.208624              0.065732   

      worst symmetry  worst fractal dimension  
max         0.663800                 0.207500  
mean        0.290076                 0.083946  
min         0.156500                 0.055040  
std         0.061867                 0.018061  

[4 rows x 30 columns]

```
This first part is just a vizualization/presetantion of data.



```
Resultados de acurácia dos Métodos usados como referência, para o Dataset Íris (5x2 cross validation):

RandoForest (parâmetros = default): 

Score =  0.946666666667


Support Vector Machine (parâmetros = default): 

Score =  0.946666666667


Rede Neural MLP (parâmetros = default):

/home/armand/anaconda2/lib/python2.7/site-packages/sklearn/neural_network/multilayer_perceptron.py:564: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
Score =  0.966666666667


```
A presentation of the methods is made through a demonstration. The score showed was obtained through crossvalidation with a data set splited in half and its a mean of five replications. These results are for the dataset Iris. The parameters of each method were not changed (defualt).



```
Matriz Resultado dos Métodos para os DataSets:

               Random Forest       SVM  Redes Neurais MLP
iris                0.950667  0.946667           0.968000
wine                0.950833  0.404545           0.317500
breast cancer       0.944828  0.627415           0.611796


Data Frame Rankeado:
              Random Forest  Random Forest_rank       SVM  SVM_rank  Redes Neurais MLP  Redes Neurais MLP_rank  
iris               0.950667            2.000000  0.946667  3.000000              0.968                1.000000   
wine               0.950833            1.000000  0.404545  2.000000             0.3175                3.000000  
breast cancer      0.944828            1.000000  0.627415  2.000000           0.611796                3.000000    
rank_medio                             1.333333            2.333333                                   2.333333   

```
The process previously presented can be performed for multiples datasets and with the results is possible construct the results matrix (presented above). It is easy to notice that some methods performed significantly poorly for certain datasets. This may be related to a bad calibration of the initial parameters, but it is not part of the scope of this paper to discuss the calibration of the methods. Next, according to Demšar it is necessary to do the ranking of the results, which is presented in another matrix.

```
Resultados teste de Friedman:
 X²_f = 2.66666666667
F_fS = 1.5

Resultado teste de Nemenyi:
CD = 1.1455

```


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
