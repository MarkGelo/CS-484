import pandas as pd
import statsmodels.api as stats
import sympy
import scipy
import numpy
from sklearn import metrics
from sklearn.model_selection import train_test_split
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import sklearn.tree as tree
import graphviz

def create_interaction(in_df1, in_df2):
    name1 = in_df1.columns
    name2 = in_df2.columns
    out_df = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            out_df[outName] = in_df1[col1] * in_df2[col2]
    return (out_df)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)
    
    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreate the estimates of the full parameters
    workParams = pd.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

def question_19():
    data = pd.read_csv("policy_2001.csv")
    data_train, data_test = train_test_split(data, test_size = 0.33, random_state = 20201014, stratify = data['CLAIM_FLAG'])
    
    y = data_train["CLAIM_FLAG"].astype('category')
    #designX = pd.DataFrame(y.where(y.isnull(), 1))
    designX = data_train[["MVR_PTS"]]
    designX = designX.join(data_train[["BLUEBOOK_1000"]])
    designX = designX.join(data_train[["TRAVTIME"]])
    designX = stats.add_constant(designX, prepend=True)
    LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')
    print("*"*50)
    
    # intercept + CREDIT SCORE BAND
    print("CREDIT SCORE BAND")
    csb = pd.get_dummies(data_train[['CREDIT_SCORE_BAND']].astype('category'))
    designX = csb
    designX = designX.join(data_train["MVR_PTS"])
    designX = designX.join(data_train["BLUEBOOK_1000"])
    designX = designX.join(data_train["TRAVTIME"])
    designX = stats.add_constant(designX, prepend=True)
    LLK_c, DF_c, fullParams_c = build_mnlogit (designX, y, debug = 'N')
    testDev = 2 * (LLK_c - LLK0)
    testDF = DF_c - DF0
    testPValue_GS = scipy.stats.chi2.sf(testDev, testDF)
    print('Significance = ', testPValue_GS)
    print("*"*50)
    """
    # intercept + 
    print("BLUEBOOK_1000")
    designX = data_train[["BLUEBOOK_1000"]]
    designX = designX.join(data_train["MVR_PTS"])
    designX = stats.add_constant(designX, prepend=True)
    LLK_c, DF_c, fullParams_c = build_mnlogit (designX, y, debug = 'N')
    testDev = 2 * (LLK_c - LLK0)
    testDF = DF_c - DF0
    testPValue_GS = scipy.stats.chi2.sf(testDev, testDF)
    print('Significance = ', testPValue_GS)
    print("*"*50)
    """
    # intercept + 
    print("CUST_LOYALTY")
    designX = data_train[["CUST_LOYALTY"]]
    designX = designX.join(data_train["MVR_PTS"])
    designX = designX.join(data_train["BLUEBOOK_1000"])
    designX = designX.join(data_train["TRAVTIME"])
    designX = stats.add_constant(designX, prepend=True)
    LLK_c, DF_c, fullParams_c = build_mnlogit (designX, y, debug = 'N')
    testDev = 2 * (LLK_c - LLK0)
    testDF = DF_c - DF0
    testPValue_GS = scipy.stats.chi2.sf(testDev, testDF)
    print('Significance = ', testPValue_GS)
    print("*"*50)
    """
    # intercept + 
    print("MVR_PTS")
    designX = stats.add_constant(data_train["MVR_PTS"], prepend=True)
    LLK_c, DF_c, fullParams_c = build_mnlogit (designX, y, debug = 'N')
    testDev = 2 * (LLK_c - LLK0)
    testDF = DF_c - DF0
    testPValue_GS = scipy.stats.chi2.sf(testDev, testDF)
    print('Significance = ', testPValue_GS)
    print("*"*50)
    """
    # intercept + 
    print("TIF")
    designX = data_train[["TIF"]]
    designX = designX.join(data_train["MVR_PTS"])
    designX = designX.join(data_train["BLUEBOOK_1000"])
    designX = designX.join(data_train["TRAVTIME"])
    designX = stats.add_constant(designX, prepend=True)
    LLK_c, DF_c, fullParams_c = build_mnlogit (designX, y, debug = 'N')
    testDev = 2 * (LLK_c - LLK0)
    testDF = DF_c - DF0
    testPValue_GS = scipy.stats.chi2.sf(testDev, testDF)
    print('Significance = ', testPValue_GS)
    print("*"*50)
    """
    # intercept + 
    print("TRAVTIME")
    designX = data_train[["TRAVTIME"]]
    designX = designX.join(data_train["MVR_PTS"])
    designX = designX.join(data_train["BLUEBOOK_1000"])
    designX = stats.add_constant(designX, prepend=True)
    LLK_c, DF_c, fullParams_c = build_mnlogit (designX, y, debug = 'N')
    testDev = 2 * (LLK_c - LLK0)
    testDF = DF_c - DF0
    testPValue_GS = scipy.stats.chi2.sf(testDev, testDF)
    print('Significance = ', testPValue_GS)
    print("*"*50)
    """
    
def question_20():
    # MVR_PTS, BLUEBOOK_1000, TRAVTIME
    data = pd.read_csv("policy_2001.csv")
    data_train, data_test = train_test_split(data, test_size = 0.33, random_state = 20201014, stratify = data['CLAIM_FLAG'])
    
    y = data_train["CLAIM_FLAG"].astype('category')
    designX = data_train[["MVR_PTS"]]
    designX = designX.join(data_train[["BLUEBOOK_1000"]])
    designX = designX.join(data_train[["TRAVTIME"]])
    designX = stats.add_constant(designX, prepend=True)
    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(designX.values).rref()
    X = designX.iloc[:, list(inds)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    print("*"*50)
    X_test = stats.add_constant(data_test[["MVR_PTS", "BLUEBOOK_1000", "TRAVTIME"]], prepend = True)
    #print(X_test)
    y_pred_prob = thisFit.predict(X_test)
    #print(y_pred_prob[[1]])
    y_pred = pd.to_numeric(y_pred_prob.idxmax(axis=1))
    #acc = metrics.accuracy_score(data_test["CLAIM_FLAG"], y_pred)
    #print(acc)
    #print(data_test["CLAIM_FLAG"],"\n",  y_pred)
    lr_auc = metrics.roc_auc_score(data_test["CLAIM_FLAG"], y_pred)
    print(lr_auc)

def question_17_18():
    trainData = pd.read_csv('ChicagoCompletedPotHole.csv',
                       delimiter=',', usecols=['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION',
    'LATITUDE','LONGITUDE'])
    
    trainData['N_POTHOLES_FILLED_ON_BLOCK'] = numpy.log(trainData['N_POTHOLES_FILLED_ON_BLOCK'])
    trainData['N_DAYS_FOR_COMPLETION'] = numpy.log(1 + trainData['N_DAYS_FOR_COMPLETION'])
    
    nClusters = numpy.zeros(10)
    Elbow = numpy.zeros(10)
    Silhouette = numpy.zeros(10)
    TotalWCSS = numpy.zeros(10)
    Inertia = numpy.zeros(10)
    KClusters = 1
    nCars = trainData.shape[0]
    
    for c in range(10):
       KClusters += 1
       nClusters[c] = KClusters
    
       kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20201014).fit(trainData)
    
       # The Inertia value is the within cluster sum of squares deviation from the centroid
       Inertia[c] = kmeans.inertia_
       
       if (KClusters > 1):
           Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
    
       WCSS = numpy.zeros(KClusters)
       nC = numpy.zeros(KClusters)
    
       for i in range(nCars):
          k = kmeans.labels_[i]
          nC[k] += 1
          diff = trainData.iloc[i,] - kmeans.cluster_centers_[k]
          WCSS[k] += diff.dot(diff)
    
       Elbow[c] = 0
       for k in range(KClusters):
          Elbow[c] += WCSS[k] / nC[k]
          TotalWCSS[c] += WCSS[k]
    """
       print("Cluster Assignment:", kmeans.labels_)
       for k in range(KClusters):
          print("Cluster ", k)
          print("Centroid = ", kmeans.cluster_centers_[k])
          print("Size = ", nC[k])
          print("Within Sum of Squares = ", WCSS[k])
          print(" ")
    """
          
    print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
    for c in range(14):
       print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
             .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))
    
    # Part (b)
    plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
    plt.grid(True)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Elbow Value")
    plt.title("Elbow Method")
    plt.show()
    
    plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
    plt.grid(True)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Value")
    plt.title("Silhouette method")
    plt.show()
    
    for k in range(2, 10):
        kmeans_model = cluster.KMeans(n_clusters = k, random_state = 20201014).fit(trainData)
        labels = kmeans_model.labels_
        print(k, metrics.calinski_harabasz_score(trainData, labels))

def question_16():
    d = {"X": [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4], 
         "Y": [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1]}
    test_d = {"X": [0,1,2,3,4],
              "Y": [1,0,1,0,1]}
    train_data = pd.DataFrame(data = d)
    test_data = pd.DataFrame(data = test_d)
    y = train_data["Y"].astype('category')
    designX = train_data[["X"]]
    designX = stats.add_constant(designX, prepend=True)
    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(designX.values).rref()
    X = designX.iloc[:, list(inds)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    print(thisFit.summary())
    X_test = stats.add_constant(test_data[["X"]])
    y_pred_prob = thisFit.predict(X_test)
    #y_pred = pd.to_numeric(y_pred_prob.idxmax(axis=1))
    print(y_pred_prob)
    
def question_5():
    data = pd.read_csv("2020S2MT_Q5.csv")
    vals = data["x"]
    int_max = int(max(vals))
    int_min = int(min(vals))
    normal = plt.hist(vals, bins = numpy.arange(int_min, int_max + 5.0, 5.0))
    plt.show()
    density = plt.hist(vals, bins = numpy.arange(int_min, int_max + 5.0, 5.0), density = True)
    plt.show()
    # get midpoint for coord
    midpoints = [m + (5.0/2) for m in numpy.arange(int_min, int_max, 5.0)]
    print([(mid, pde) for mid, pde in zip(midpoints, density[0])])
    #print(sum(density[0]))
    # show histogram with pde and densities
    plot = sns.distplot(vals, hist = True, kde = True,
                        bins = numpy.arange(int_min, int_max + 5.0, 5.0))
    plot.set(xlabel = 'x', ylabel = 'density')
    plt.show()
    in_range = [x for x in vals if x < 100 and x >= 95]
    print(in_range)
    print(len(in_range), "values in bin midpoint = 97.5 -- out of total 100 values")
    
def question_13_14_15():
    d = {"Feature": [], "Target": []}
    d["Feature"].extend(["I"]*65)
    d["Target"].extend([1]*65)
    d["Feature"].extend(["I"]*304)
    d["Target"].extend([2]*304)
    d["Feature"].extend(["I"]*530)
    d["Target"].extend([3]*530)
    d["Feature"].extend(["I"]*487)
    d["Target"].extend([4]*487)
    d["Feature"].extend(["I"]*140)
    d["Target"].extend([5]*140)
    
    d["Feature"].extend(["II"]*490)
    d["Target"].extend([1]*74)
    d["Target"].extend([2]*185)
    d["Target"].extend([3]*160)
    d["Target"].extend([4]*55)
    d["Target"].extend([5]*16)
    
    d["Feature"].extend(["III"]*2002)
    d["Target"].extend([1]*33)
    d["Target"].extend([2]*228)
    d["Target"].extend([3]*623)
    d["Target"].extend([4]*755)
    d["Target"].extend([5]*363)
    
    d["Feature"].extend(["IV"]*982)
    d["Target"].extend([1]*90)
    d["Target"].extend([2]*290)
    d["Target"].extend([3]*349)
    d["Target"].extend([4]*213)
    d["Target"].extend([5]*40)
    data = pd.DataFrame(data = d)
    y = data["Target"].astype('category')
    # 1 is A, 2 is B and so on
    #print(y)
    
    # intercept only
    designX = pd.DataFrame(y.where(y.isnull(), 1))
    LLK0, DF0, fullParams0 = build_mnlogit(designX, y, debug = 'Y')
    
    # intercept + feature
    print("---------Intercept + feature-------------")
    fea = pd.get_dummies(data[["Feature"]].astype('category'))
    designX = fea
    designX = stats.add_constant(designX, prepend=True)
    LLK_c, DF_c, fullParams_c = build_mnlogit (designX, y, debug = 'Y')
    testDev = 2 * (LLK_c - LLK0)
    testDF = DF_c - DF0
    testPValue_GS = scipy.stats.chi2.sf(testDev, testDF)
    print('Deviance Chi=Square Test')
    print("Number of Free Parameters =", DF_c)
    print("Model Log-Likelihood Value =", LLK_c)
    print('Deviance test Statistic = ', testDev)
    print('  Degreee of Freedom = ', testDF)
    print('        Significance = ', testPValue_GS)
    print("*"*50)
    
def test():
    d = {"Target": []}
    d["Target"].extend([1]*262)
    d["Target"].extend([2]*1007)
    d["Target"].extend([3]*1662)
    d["Target"].extend([4]*1510)
    d["Target"].extend([5]*559)
    data = pd.DataFrame(data = d)
    y = data["Target"].astype('category')
    # 1 is I, 2 is II, and so on
    #print(y)
    
    # intercept only
    designX = pd.DataFrame(y.where(y.isnull(), 3)) # which number to choose
    # different numbers result in diff coef
    LLK0, DF0, fullParams0 = build_mnlogit(designX, y, debug = 'Y')


if __name__ == "__main__":
    #test()
    #question_13_14_15()
    #question_5()
    #question_16()
    #question_17_18()
    #question_19()
    #question_20()
    
    
    
    
    
