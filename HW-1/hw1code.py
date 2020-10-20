import matplotlib.pyplot as plt
import csv
import math
import numpy as np
from numpy import percentile
import seaborn as sns
import pandas as pd
import statistics
import scipy
from sklearn.neighbors import KNeighborsClassifier
from scipy import linalg as LA2
from pprint import pprint

# read normal sample csv file
numbers = []
with open('NormalSample.csv', 'r') as f:
    data = csv.DictReader(f)
    for row in data:
        numbers.append(row)
# turn all values into numbers, not string
numbers = [dict([a, float(x)] for a,x in b.items()) for b in numbers]
    
def question_1():
    # get largest integer less than min, smallest integer greater than max
    vals = [x['x'] for x in numbers]
    print('max:', max(vals), 'min:', min(vals))
    int_max = math.ceil(max(vals))
    int_min = math.floor(min(vals))
    pl_range = int_max - int_min
    print('Integer Max:', int_max, 'Integer Min:', int_min)
    questions = [{'number': '1.b', 'h': 0.25},
                 {'number': '1.c', 'h': 0.5},
                 {'number': '1.d', 'h': 1},
                 {'number': '1.e', 'h': 2}]
    # range / h = bins
    # plots for each question
    over_35 = [x for x in vals if x > 35]
    print(over_35)
    for q in questions:
        num_bins = int((int_max - int_min) // q['h'])
        nums = len(np.arange(int_min, int_max + q['h'], q['h']))
        # show normal histogram
        normal = plt.hist(vals, bins = np.arange(int_min, int_max + q['h'], q['h']))
        plt.title(q['number'])
        plt.xlabel('x')
        plt.ylabel('total')
        plt.show()
        fig = plt.figure()
        density = plt.hist(vals, bins = np.arange(int_min, int_max + q['h'], q['h']), density = True)
        # get midpoint for coord
        midpoints = [m + (q['h']/2) for m in np.arange(int_min, int_max, q['h'])]
        print([(mid, pde) for mid, pde in zip(midpoints, density[0])])
        plt.close(fig)
        # show histogram with pde and densities
        plot = sns.distplot(vals, hist = True, kde = True,
                            bins = np.arange(int_min, int_max + q['h'], q['h']))
        plot.set(xlabel = 'x', ylabel = 'density')
        plot.set_title(q['number'])
        plt.show()
    
def question_2():
    # box plot for each group
    groups = set([x['group'] for x in numbers])
    groupings = {}
    # box plot for all x
    vals = [x['x'] for x in numbers]
    # get five number summary
    print('All x summary')
    quartiles = percentile(vals, [25, 75])
    median = statistics.median(vals)
    print(f'Min: {min(vals)}')
    print(f'Q1: {quartiles[0]}')
    print(f'Median: {median}')
    print(f'Q3: {quartiles[1]}')
    print(f'Max: {max(vals)}')
    # calculate 1.5 IQR for outliers
    iqr = quartiles[1] - quartiles[0]
    lower_lim = quartiles[0] - (1.5 * iqr)
    upper_lim = quartiles[1] + (1.5 * iqr)
    print(f'1.5 IQR whiskers = ({lower_lim}, {upper_lim})')
    groupings['all'] = vals
    outliers = [x for x in vals if x < lower_lim or x > upper_lim]
    print(f'{len(outliers)} outliers = ', np.sort(outliers))
    print('------------------------------------')
    # box plot for each group and 5 number summaries
    for g in groups:
        print(f'{g} summary')
        vals_g = [x['x'] for x in numbers if x['group'] == g]
        groupings[g] = vals_g
        quartiles = percentile(vals_g, [25, 75])
        median = statistics.median(vals_g)
        print(f'Min: {min(vals)}')
        print(f'Q1: {quartiles[0]}')
        print(f'Median: {median}')
        print(f'Q3: {quartiles[1]}')
        print(f'Max: {max(vals)}')
        # calculate 1.5 IQR for outliers
        iqr = quartiles[1] - quartiles[0]
        lower_lim = quartiles[0] - (1.5 * iqr)
        upper_lim = quartiles[1] + (1.5 * iqr)
        print(f'1.5 IQR whiskers = ({lower_lim}, {upper_lim})')
        outliers = [x for x in vals_g if x < lower_lim or x > upper_lim]
        print(f'{len(outliers)} outliers = ', np.sort(outliers))
        print('------------------------------------')
    # make plots for each group -- all, 1, 2
    fig, ax = plt.subplots()
    ax.boxplot(groupings.values())
    ax.set_xticklabels(groupings.keys())
    ax.set_xlabel('Group')
    ax.set_ylabel('x')
    ax.set_title('2')

def question_3():
    data = []
    with open('Fraud.csv', 'r', encoding = 'utf-8-sig') as f:
        cs = csv.DictReader(f)
        for row in cs:
            data.append(row)
    # turn values into int
    data = [dict([a, float(x)] for a,x in b.items()) for b in data]
    # get total and fraud
    fraud = [x for x in data if x['FRAUD'] == 1]
    not_fraud = [x for x in data if x['FRAUD'] == 0]
    print(f'Total: {len(data)}, Fraud: {len(fraud)}, Not Fraud: {len(not_fraud)}')
    # for each interval var, box plot
    itv_vars = {'TOTAL_SPEND': 'Total $', 'DOCTOR_VISITS': 'Number of Visits', 
                'NUM_CLAIMS': 'Number of claims', 'MEMBER_DURATION': 'Membership Duration', 
                'OPTOM_PRESC': 'Number of optical exams', 'NUM_MEMBERS': 'Number of members covered'}
    for var, info in itv_vars.items():
        fraud_v = [x[var] for x in fraud]
        not_fraud_v = [x[var] for x in not_fraud]
        plt.boxplot([fraud_v, not_fraud_v], 
                    labels = ['Fraud', 'Not Fraud'],
                    sym = 'k.', vert = False)
        plt.title(var)
        plt.xlabel('Result')
        plt.ylabel(info)
        plt.show()
    # interval vars to matrix
    matrix = [list(x.values())[2:] for x in data]
    x = np.matrix(matrix)
    xtx = x.transpose() * x
    # get eigenvalues
    eigen_val, eigen_vec = np.linalg.eigh(xtx)
    print('Eigenvalues: ', eigen_val)
    print(f'{len([x > 1 for x in eigen_val])} dimensions can be used as they all have eigenvalues > 1')
    # get transformation matrix and prove orthonormal
    transf_m = eigen_vec * np.linalg.inv(np.sqrt(np.diagflat(eigen_val)))
    print('Transformation matrix :', transf_m)
    # transformed x
    transf_x = x * transf_m
    # check if identity matrix
    result = transf_x.transpose() * transf_x
    #print(result)
    # orthonormalize
    orth_x = LA2.orth(x)
    check = orth_x.transpose().dot(orth_x)
    # should be identity matrix
    print(check)
    
    # nearest neighbors
    neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
    # target
    target = [list(x.values())[1] for x in data]
    neighbors = neigh.fit(transf_x, target)
    print(neighbors.score(transf_x, target))
    
    # observation
    obsv = np.matrix([[7500, 15, 3, 127, 2, 2]])
    transf_obsv = obsv * transf_m
    obsv_neigh = neighbors.kneighbors(transf_obsv)
    # print the distance and the element # of the neighbors
    obsv_neighbors = [(distance, element) 
                      for distance, element in zip(list(obsv_neigh[0][0]), 
                                                   list(obsv_neigh[1][0]))]
    print(obsv_neighbors)
    # print the neighbors values
    for neighb in obsv_neighbors:
        print(data[neighb[1]])
        
    # print probability of fraud
    print(neighbors.predict_proba(transf_obsv))
    # get 1, higher than fraud % which was like 0.19
    # so fraud

def question_4():
    data = [[7.7, -37, 4],
            [9.5, -38, 1],
            [3.0, -34, 2],
            [9.1, -75, 1],
            [2.2, -31, 2],
            [4.8, -7, 4],
            [5.5, -6, 3],
            [10, -61, 1],
            [4.2, -23, 2],
            [1.6, -54, 1]]
    # euclidean = math.sqrt((current[0] - check[0]) ** 2 + (current[1] - check[1]) ** 2)
    # manhattan = abs(current[0] - check[0]) + abs(current[1] - check[1])
    # chebyshev = max(abs(current[0] - check[0]), abs(current[1] - check[1]))
    for i_nbr in range(1, 10):
        k = i_nbr
        step1 = []
        i = 0
        for current in data:
            step1.append([])
            step1[i] = []
            for check in data:
                # replace distance for different stuff
                distance = max(abs(current[0] - check[0]), abs(current[1] - check[1]))
                step1[i].append(np.round(distance, decimals = 4))
            i += 1
        #pprint(step1)
        step2 = []
        for row in step1:
            array = np.array(row)
            temp = array.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(array)) + 1
            step2.append(list(ranks))
        #pprint(step2)
        step3 = []
        for i in range(len(step2)):
            nbrs_y = [data[x][2] for x in range(len(step2)) if step2[i][x] <= k]
            y_guess = sum(nbrs_y) / len(nbrs_y)
            error = np.round(data[i][2] - y_guess, decimals = 3)
            squared = np.round(error ** 2, decimals = 3)
            #print(data[i][2], y_guess, error, squared)
            step3.append([data[i][2], y_guess, error, squared])
        #pprint(step3)
        sqr_err = [x[3] for x in step3]
        root_avg_sqr_err = math.sqrt(sum(sqr_err) / len(sqr_err))
        print(k, root_avg_sqr_err)
    

def question_5():
    inputVar = ['Variance_Image', 'Skewness_Image', 'Kurtosis_Image', 'Entropy_image']
    targetVar = 'Authenticity'
    bankNote = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',
                               sep = ',', header = None)
    bankNote = bankNote.rename(columns = {0: inputVar[0], 1: inputVar[1], 2: inputVar[2],
                                          3: inputVar[3], 4: targetVar})
    # to 4, cuz all vars including entropy image
    X = bankNote[inputVar[0:4]]
    y = bankNote[targetVar]
    nObs = len(y)
    misClassRate_orig = pd.DataFrame(columns = ['k', 'Misclassification Rate'])
    kMax = 10
    for i in range(kMax):
        k = i +1
        nbrs = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute', metric = 'euclidean')
        model = nbrs.fit(X, y)
        yPredictProb = model.predict_proba(X)
        yPredict = min(y) + np.argmax(yPredictProb, axis = 1)
        MCE = np.sum(np.where(y == yPredict, 0 , 1)) / nObs
        misClassRate_orig.loc[i] = [k, MCE]
    # without orthonormalizing
    print('Original')
    print(misClassRate_orig)
    print('--------------------')
    # orthonormalizing
    X = np.matrix(bankNote[inputVar[0:4]].values)
    xtx = X.transpose() * X
    evals, evecs = np.linalg.eigh(xtx)
    transf = evecs * np.linalg.inv(np.sqrt(np.diagflat(evals)))
    transf_x = X * transf
    misClassRate_orth = pd.DataFrame(columns = ['k', 'Misclassification Rate'])
    for i in range(kMax):
        k = i + 1
        nbrs = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute', metric = 'euclidean')
        model = nbrs.fit(transf_x, y)
        yPredictProb = model.predict_proba(transf_x)
        yPredict = min(y) + np.argmax(yPredictProb, axis = 1)
        MCE = np.sum(np.where(y == yPredict, 0 , 1)) / nObs
        misClassRate_orth.loc[i] = [k, MCE]        
    print('Orthonormalize')
    print(misClassRate_orth)

if __name__ == '__main__':
    #question_1()
    #question_2()
    #question_3()
    #question_4()
    zquestion_5()