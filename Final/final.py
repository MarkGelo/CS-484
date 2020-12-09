import pandas as pd
import statsmodels.api as stats
import sympy
import scipy
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import sklearn.tree as tree
import graphviz
import math
import sklearn.svm as svm
import sklearn.neural_network as nn
import sklearn.ensemble as ensemble

# Define a function that performs the Chi-square test
def ChiSquareTest (
    xCat,           # input categorical feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):

    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))
    print("CROSSTAB")
    print(obsCount)
    if (debug == 'Y'):
        print('Observed Count:\n', obsCount)
        print('Column Total:\n', cTotal)
        print('Row Total:\n', rTotal)
        print('Overall Total:\n', nTotal)
        print('Expected Count:\n', expCount)
        print('\n')
       
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
    chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = np.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig, cramerV)

def split(data, split = 1, typ = 'EO', subset = None): # entropy ordinal splitting
    # splits according to the parameter
    if typ == 'EO':
        data['LE_Split'] = (data.iloc[:,0] <= split)
    elif typ == 'EN':
        data['LE_Split'] = data.iloc[:, 0].apply(lambda x: True if x in subset else False)
    cross_table = pd.crosstab(index=data['LE_Split'], 
                              columns=data.iloc[:, 1], margins=True, 
                              dropna=True)
    n_rows = cross_table.shape[0]
    n_col = cross_table.shape[1]
    t_entropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_col):
            proportion = cross_table.iloc[i_row, i_column] / cross_table.iloc[i_row, (n_col - 1)]
            if proportion > 0:
                row_entropy -= proportion * np.log2(proportion)
        t_entropy += row_entropy * cross_table.iloc[i_row, (n_col - 1)]
    t_entropy = t_entropy / cross_table.iloc[(n_rows - 1), (n_col - 1)]
    return cross_table, t_entropy

def FindMinEO(data, intervals): # finding mininum entropy ordinal
    min_entropy = 999999999.99999999999
    min_interval = None
    min_table = None
    # tries out all splits possible and finds the min
    for i in range(intervals[0], intervals[len(intervals) - 1]):
        cur_table, cur_entropy = split(data, i + 0.5, typ = 'EO')
        if cur_entropy < min_entropy:
            min_entropy = cur_entropy
            min_interval = i + 0.5
            min_table = cur_table
    return min_table, min_entropy, min_interval

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
    workParams = pd.DataFrame(np.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

def question_7():
    # make up data
    d = {"Row": [], "Column": []}
    d['Row'].extend(['A']*(4340 + 5403 + 2456 + 353))
    d['Column'].extend([1]*4340)
    d['Column'].extend([2]*5403)
    d['Column'].extend([3]*2456)
    d['Column'].extend([4]*353)
    d['Row'].extend(['B']*(8095 + 16156 + 10798 + 2371))
    d['Column'].extend([1]*8095)
    d['Column'].extend([2]*16156)
    d['Column'].extend([3]*10798)
    d['Column'].extend([4]*2371)
    d['Row'].extend(['C']*(4761 + 14154 + 14103 + 4597))
    d['Column'].extend([1]*4761)
    d['Column'].extend([2]*14154)
    d['Column'].extend([3]*14103)
    d['Column'].extend([4]*4597)
    d['Row'].extend(['D']*(813 + 3636 + 5307 + 2657))
    d['Column'].extend([1]*813)
    d['Column'].extend([2]*3636)
    d['Column'].extend([3]*5307)
    d['Column'].extend([4]*2657)
    data = pd.DataFrame(data = d)
    chisqstat, chisqdf, chisqsig, cramerV = ChiSquareTest(data['Row'],
                                                          data['Column'])
    print("CramerV:", cramerV)
    
def question_11_12_13():
    d = {"Vehicle Age": [], "Claim Indicator": []}
    d["Vehicle Age"].extend([1]*(1731 + 846)) # 1 to 3
    d["Claim Indicator"].extend(["No"]*1731)
    d["Claim Indicator"].extend(["Yes"]*846)
    d["Vehicle Age"].extend([2]*(1246 + 490)) # 4 to 7
    d["Claim Indicator"].extend(["No"]*1246)
    d["Claim Indicator"].extend(["Yes"]*490)
    d["Vehicle Age"].extend([3]*(1412 + 543)) # 8 to 10
    d["Claim Indicator"].extend(["No"]*1412)
    d["Claim Indicator"].extend(["Yes"]*543)
    d["Vehicle Age"].extend([4]*(2700 + 690)) # 11 and above
    d["Claim Indicator"].extend(["No"]*2700)
    d["Claim Indicator"].extend(["Yes"]*690)
    data = pd.DataFrame(data = d)
    p_no = data.groupby('Claim Indicator').size()['No']/data.shape[0]
    p_yes = data.groupby('Claim Indicator').size()['Yes']/data.shape[0]
    root_entropy = -((p_yes * math.log2(p_yes)) + (p_no * math.log2(p_no)))
    print("QUESTION 11")
    print("Root Entropy:", root_entropy)
    print("----------------------------")
    print("QUESTION 12")
    age_split = FindMinEO(data[['Vehicle Age', 'Claim Indicator']],
                                       [1, 2, 3, 4])
    print("First Layer Split:")
    print("Vehicle Age:", age_split[1])
    print(f"""Vehicle Age for first split with entropy of {age_split[1]} 
          with a split at {age_split[2]}""")
    print("----------------------")
    
    print("QUESTION 13")
    print("Entropy Reduction:", root_entropy - age_split[1])
    
def question_14():
    train_data = pd.read_csv("WineQuality_Train.csv")
    test_data = pd.read_csv("WineQuality_Test.csv")
    y = train_data["quality_grp"]
    #designX = pd.DataFrame(y.where(y.isnull(), 1))
    designX = train_data[["alcohol"]]
    designX = designX.join(train_data["citric_acid"])
    designX = designX.join(train_data["free_sulfur_dioxide"])
    designX = designX.join(train_data["residual_sugar"])
    designX = designX.join(train_data["sulphates"])
    designX = stats.add_constant(designX, prepend=True)
    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(designX.values).rref()
    X = designX.iloc[:, list(inds)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 10000, tol = 1e-8)
    print(thisFit.summary())
    X_test = stats.add_constant(test_data[["alcohol", "citric_acid", "free_sulfur_dioxide",
                                           "residual_sugar", "sulphates"]])
    y_pred_prob = thisFit.predict(X_test)
    #y_pred = pd.to_numeric(y_pred_prob.idxmax(axis=1))
    print(y_pred_prob)
    threshold = 0.1961733010776
    pred_data = []
    y_pred_prob = y_pred_prob.applymap(lambda x: 1 if x >= threshold else 0)
    for x in y_pred_prob[1]:
        pred_data.append(x)
    actu_data = test_data["quality_grp"].tolist()
    y_actu = pd.Series(actu_data, name = 'Actual')
    y_pred = pd.Series(pred_data, name = 'Predicted')
    cfn1 = pd.crosstab(y_actu, y_pred)
    print(cfn1)
    accuracy = metrics.accuracy_score(y_actu, y_pred)
    misclassification = 1 - accuracy
    print(f"Accuracy: {accuracy}")
    print(f"Misclassification Rate: {misclassification}")

def question_15():
    train_data = pd.read_csv("WineQuality_Train.csv")
    test_data = pd.read_csv("WineQuality_Test.csv")
    svmmodel = svm.SVC(kernel = "linear",
                       random_state = 20201202)
    thisfit = svmmodel.fit(train_data[["alcohol", "citric_acid", "free_sulfur_dioxide",
                                           "residual_sugar", "sulphates"]], 
                           train_data["quality_grp"])
    y_pred = thisfit.predict(test_data[["alcohol", "citric_acid", "free_sulfur_dioxide",
                                           "residual_sugar", "sulphates"]])
    print(y_pred)
    actu_data = test_data["quality_grp"].tolist()
    y_actu = pd.Series(actu_data, name = 'Actual')
    y_pred = pd.Series(y_pred, name = 'Predicted')
    cfn1 = pd.crosstab(y_actu, y_pred)
    print(cfn1)
    accuracy = metrics.accuracy_score(y_actu, y_pred)
    misclassification = 1 - accuracy
    print(f"Accuracy: {accuracy}")
    print(f"Misclassification Rate: {misclassification}")

def question_16():
    train_data = pd.read_csv("WineQuality_Train.csv")
    test_data = pd.read_csv("WineQuality_Test.csv")
    b_df = pd.DataFrame(columns = ["Activation Function", 
                                   "# Layers", 
                                   "# neurons/layer",
                                   "# Iterations",
                                   "Loss",
                                   "Misclassication Rate",
                                   "Out Activation"])
    temp_b_df = pd.DataFrame(columns = ["Activation Function",
                                   "# Layers", 
                                   "# neurons/layer",
                                   "# Iterations",
                                   "Loss",
                                   "Misclassication Rate",
                                   "Out Activation"])
    act = ["relu"]
    threshold = 0.1961733010776
    for actf in act:
        #print(actf)
        for nl in np.arange(1, 11):
            for npl in np.arange(5, 11):
                nnobj = nn.MLPClassifier(hidden_layer_sizes = (npl,)*nl,
                                         activation = actf,
                                         verbose = False,
                                         solver = "lbfgs",
                                         learning_rate_init = 0.1,
                                         max_iter = 5000,
                                         random_state = 20201202)
                nnfit = nnobj.fit(train_data[["alcohol", "citric_acid", "free_sulfur_dioxide",
                                           "residual_sugar", "sulphates"]], 
                           train_data["quality_grp"])
                y_pred_prob = nnobj.predict_proba(test_data[["alcohol", "citric_acid", "free_sulfur_dioxide",
                                           "residual_sugar", "sulphates"]])
                y_pred = np.where(y_pred_prob[:,1] >= threshold, 1, 0)
                misclass = 1 - (metrics.accuracy_score(test_data["quality_grp"], y_pred))
                temp_b_df = temp_b_df.append(pd.DataFrame(
                    [[nnobj.activation, nl, npl, nnobj.n_iter_, 
                      nnobj.loss_, misclass, nnobj.out_activation_]],
                    columns = ["Activation Function",
                                   "# Layers", 
                                   "# neurons/layer",
                                   "# Iterations",
                                   "Loss",
                                   "Misclassication Rate",
                                   "Out Activation"]))
        # get min
        b_df = b_df.append(temp_b_df[temp_b_df.Loss == temp_b_df.Loss.min()])
        temp_b_df = pd.DataFrame(columns = temp_b_df.columns)
    print(b_df)

def question_10():
    thresholds = [0.2, 0.3, 0.45, 0.5, 0.55, 0.4, 0.7, 0.1]
    actu = [0,0,1,1,1,0,0,1,1,0]
    pred_prob = [0.2, 0.3, 0.45, 0.5, 0.55, 0.4, 0.45, 0.7, 0.7, 0.1]
    for threshold in thresholds:
        pred = []
        for pred_pr in pred_prob:
            if pred_pr >= threshold:
                pred.append(1)
            else:
                pred.append(0)
        y_actu = pd.Series(actu, name = 'Actual')
        y_pred = pd.Series(pred, name = 'Predicted')  
        cfn1 = pd.crosstab(y_actu, y_pred)
        #print(cfn1)
        tn = cfn1[0][0]
        fn = cfn1[0][1]
        fp = cfn1[1][0]
        tp = cfn1[1][1]
        # tp = 5
        # fp = 5
        # fn = 0
        # f1s = 5/(5+0.5(5)) = 5/7.5
        print(type(fp))
        f1s = tp / (tp + 0.5*(fp + fn))
        print(threshold, ":", f1s)

def question_5():
    data = pd.read_csv("FinalQ5.csv")
    y = data["Late4Work"]
    csb = pd.get_dummies(data[['TransportMode']].astype('category'))
    designX = csb
    designX = designX.join(data["CommuteMile"])
    #print(data["CommuteMile"])
    #ct = create_interaction(csb, pd.get_dummies(data[['CommuteMile']].astype('int64')))
    #designX = designX.join(ct)
    designX = stats.add_constant(designX, prepend=True)
    reduced_form, inds = sympy.Matrix(designX.values).rref()
    X = designX.iloc[:, list(inds)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 10000, tol = 1e-8)
    print(thisFit.summary())

if __name__ == "__main__":
    #question_7()
    #question_11_12_13()
    #question_14()
    #question_15()
    #question_16()
    #question_10()
    question_5()
    pass
    
    
    
    
    
