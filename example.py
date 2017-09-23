#!/usr/bin/python
# -*- coding: utf-8 -*-

# COMPARAÇÃO DE MULTIPLOS METODOS DE APRENDIZAGEM EM MULTIPLOS DATASETS
# Métodos : RandoForest, SVM, e RN
# DataSets: iris, wine, breast_cancer_boston (todos da UCI)

# Arquivo criado para apresentação na matéria
# " Tópicos Especiais em Inteligẽncia Artificial".
# Autor: Lucas Armand Souza Assis de Oliveira
# Prof.: Gerson Zaverucha

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import scipy.stats as ss
import math
from clfCompare import friedmanTest
from clfCompare import nemenyiTest

def dataVisualize(data= load_iris()):


    # Criando um dataframe com as datas de entrada:
    df = pd.DataFrame(data.data, columns=data.feature_names)


    # Apresenta dos dados e divisão entre teste e trainamento:
    print 'Apresentação dos dados (5 primeiras observações):'
    print df.head()
    print ''
    print 'Propriedades do DataSet:'
    s_mean = df.mean()
    s_std = df.std()
    s_max = df.max()
    s_min = df.min()

    df_prop= pd.DataFrame({'max':s_max,'min':s_min,'mean':s_mean,'std':s_std}).transpose()
    print df_prop


def randForest(data=load_iris(), report=False):

    # Criando um dataframe com as datas de entrada:
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Adicionando os dados que objetivamos prever ao dataframe:
    df['target'] = pd.Categorical.from_codes(data.target, data.target_names)

    # Reservando os o nome das propriedades do dataset:
    features = data.feature_names

    # Traduzindo as classes em fatores numéricos como 1,2,3:
    y = pd.factorize(df['target'])[0]

    # Definindo o classificador Random Forest na configuração default:
    clf = RandomForestClassifier()
    if report: print clf

    # Realizar uma comparação do tipo 5x2 cross validation (Dietterich (1988))
    score = np.mean([cross_val_score(clf,df[features],y,cv=2) for i in range(5)])

    return score

def sVM(data=load_iris(), perc_train=.75, report=False):

    # Criando um dataframe com as datas de entrada:
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Adicionando os dados que objetivamos prever ao dataframe:
    df['target'] = pd.Categorical.from_codes(data.target, data.target_names)

    # Reservando os o nome das propriedades do dataset:
    features = data.feature_names

    # Traduzindo as classes em fatores numéricos como 1,2,3:
    y = pd.factorize(df['target'])[0]

    # Definindo o classificador Random Forest na configuração default:
    clf = SVC()
    if report: print clf

    # Realizar uma comparação do tipo 5x2 cross validation (Dietterich (1988))
    score = np.mean([cross_val_score(clf,df[features],y,cv=2) for i in range(5)])

    return score


def mLP(data=load_iris(), perc_train=.75, report=False):

    # Criando um dataframe com as datas de entrada:
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Adicionando os dados que objetivamos prever ao dataframe:
    df['target'] = pd.Categorical.from_codes(data.target, data.target_names)

    # Reservando os o nome das propriedades do dataset:
    features = data.feature_names

    # Traduzindo as classes em fatores numéricos como 1,2,3:
    y = pd.factorize(df['target'])[0]

    # Definindo o classificador Random Forest na configuração default:
    clf = MLPClassifier()
    if report: print clf

    # Realizar uma comparação do tipo 5x2 cross validation (Dietterich (1988))
    score = np.mean([cross_val_score(clf,df[features],y,cv=2) for i in range(5)])

    return score


def clfMatrizExemp (matriz=None):
    # Se matriz igual 'None' isso siginifica que todos os valores da
    # gerado mais uma vez.
    if matriz is None:
        rf_iris = randForest(load_iris())
        svm_iris = sVM(load_iris())
        mlp_iris = mLP(load_iris())
        rf_wine = randForest(load_wine())
        svm_wine = sVM(load_wine())
        mlp_wine = mLP(load_wine())
        rf_breast_cancer = randForest(load_breast_cancer())
        svm_breast_cancer = sVM(load_breast_cancer())
        mlp_breast_cancer = mLP(load_breast_cancer())
        matriz = [[rf_iris,svm_iris,mlp_iris],
                  [rf_wine,svm_wine,mlp_wine],
                  [rf_breast_cancer,svm_breast_cancer,mlp_breast_cancer]]
    df = pd.DataFrame(data = matriz, index = ['iris','wine','breast cancer'], columns = ['Random Forest', 'SVM', 'Redes Neurais MLP'])
    return df

if __name__ == '__main__':
    rpt = False # Alterar essa variável fara com que os parâmetros
                # dos métodos sejam apresentados
    print 'Apresentação das bases de dados usadas como referência:'
    print ''
    print ' Iris (UCI) '
    dataVisualize(load_iris())
    print ''
    print ' Wine (UCI) '
    dataVisualize(load_wine())
    print ''
    print ' Breast Cancer (UCI) '
    dataVisualize(load_breast_cancer())
    print ''
    print ''
    print ''

    print 'Resultados de acurácia dos métodos usados como referência, para o Dataset Íris (5x2 cross validation):'
    print ''
    print 'RandoForest (parâmetros = default): '
    print ''
    rf= randForest(report=rpt)
    print 'Score = ',rf
    print ''
    print ''
    print 'Support Vector Machine (parâmetros = default): '
    print ''
    svm = sVM(report=rpt)
    print 'Score = ',svm
    print ''
    print ''
    print 'Rede Neural MLP (parâmetros = default):'
    print ''
    mlp= mLP(report=rpt)
    print 'Score = ',mlp
    print ''
    print ''
    print ''
    print 'Matriz Resultado dos Métodos para os DataSets:'
    print ''
    df = clfMatrizExemp()
    print df
    print ''
    print ''
    friedmanTest(df, report=True)
    nemenyiTest(df, report=True)
# REFERENCES:
# www.chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html
