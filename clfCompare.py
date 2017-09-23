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

def ranking (df,report=False):

    # testenado se "df" é realmente um data frame:
    if type(df) != pd.core.frame.DataFrame:
        df = pd.DataFrame(df)

    # ss.rankdata calcula o rank de um vetor (do menor para o maior)
    rank = []
    for line_result in df.itertuples():
        rank.append(ss.rankdata(line_result[1:]))

    # Aqui nós inverteremos o rank (para ser do maior para o menor)
    M = max(rank[0])+1
    new_rank =[]
    for line_rank in rank:
        new_rank.append([M-i for i in line_rank])

    # Contrução do novo DataFrame:
    new_matriz = []
    for df_row,rank_row  in zip(df.itertuples(),new_rank):
	row = []
	for i in range(1,len(df_row)):
		row.append(df_row[i])
		row.append(rank_row[i-1])
	new_matriz.append(row)

    new_columns = []
    for col_name in df.columns:
	new_columns.append(col_name)
	new_columns.append(col_name+'_rank')


    # Agora vamos calcular o rankmédio de cada método:
    count = 0
    media_rank = []
    for line_rank in new_rank:
        if count == 0:
            count +=1
            media_rank = line_rank
        else:
            count+=1
            for j in range(len(line_rank)):
                media_rank[j] += line_rank[j]
    media_rank = [float(i)/count for i in media_rank]
    new_med_rnk = []
    for med_rnk in media_rank:
	new_med_rnk.append('')
	new_med_rnk.append(med_rnk)

    # Adicionar os rankings finais a matriz
    new_matriz.append(new_med_rnk)

    # E atualizar os index
    new_index = df.index.append(pd.Series(['rank_medio']))

    # Contruindo DataFrame rankeado:
    new_df = pd.DataFrame(data = new_matriz, index = new_index, columns = new_columns)
    new_df.rank = True

    if report == True:
        print 'Data Frame Rankeado:'
        print new_df
        print ''

    return new_df

def friedmanTest (df, report=False):

    if df.rank != True:
        print 'Data frame será rankeado'
        print ''
        print report
        df = ranking(df,report)
    elif report == True:
        print 'DataFrame já rankeado'
        print df

    # # # Implementação do teste de Fredman:

    # k =  Número de algorítimos:
    k = len(df.columns)/2 		# É necessário descontar as colunas de rank

    # N = Número de datasets:
    N = len(df)

    # Inicialmente devemos acessar a ultima linha do DataFrame (médias dos ranks)
    last_row = df[:][-1:].squeeze()

    # E isolar só os valores de rankmédio dos valores nulos (''):
    media_rank = last_row[last_row!='']

    m_r_quadrado = media_rank**2
    X_quadrado_F = (12*N/(k*(k+1)))*(sum(m_r_quadrado) - (k*((k+1)**2))/4)
    F_f = (N-1)*X_quadrado_F/(N*(k-1)-X_quadrado_F)

    if report == True:
        # Report resultado:
        print ''
        print 'Resultados teste de Friedman:'
        print ' X²_f = '+str(X_quadrado_F)
        print 'F_fS = '+str(F_f)
        print ''
    return F_f

def nemenyiTest (df, alfa=0.10, report=False):

    if df.rank != True:
        print 'Data frame será rankeado'
        print ''
        print report
        df = ranking(df,report)
    elif report == True:
        print 'DataFrame já rankeado'
        print df

    # k =  Número de algorítimos:
    k = len(df.columns)/2 		# É necessário descontar as colunas de rank

    # N = Número de datasets:
    N = len(df)

    # Nemenyi Test:
    # Calculo da distancia critica:

    q={}
    q[0.05] = {2:1.960, 3:2.343, 4:2.569, 5:2.728, 6:2.850, 7:2.949, 8:3.031, 9:3.102, 10:3.164}
    q[0.10] = {2:1.645, 3:2.052, 4:2.291, 5:2.459, 6:2.589, 7:2.693, 8:2.780, 9:2.855, 10:2.920}

    CD = q[alfa][N]*math.sqrt((float(k)*(k-1))/(6*N))

    if report == True:
        # Report CD
        print 'Resultado teste de Nemenyi:'
        print 'CD = '+ str(CD)
        print ''
    return CD

# REFERENCES:
# www.chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html
