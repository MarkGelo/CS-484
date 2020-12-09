import numpy
import pandas as pd
pd.options.display.max_columns = 1000
import scipy
import statsmodels.api as stats
import sklearn.naive_bayes as naive_bayes

# functions from class
# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y):

    # Find the non-redundant columns in the design matrix fullX
    nFullParam = fullX.shape[1]
    XtX = numpy.transpose(fullX).dot(fullX)
    invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim = nFullParam, inputM = XtX, tol = 1e-8)

    # Build a multinomial logistic model
    X = fullX.iloc[:, list(nonAliasParam)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method = 'newton', maxiter = 100, gtol = 1e-8, full_output = True, disp = True)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    # The number of free parameters
    nYCat = thisFit.J
    thisDF = len(nonAliasParam) * (nYCat - 1)

    # Return model statistics
    return (thisLLK, thisDF, thisParameter, thisFit, aliasParam)

def SWEEPOperator (pDim, inputM, tol):
    # pDim: dimension of matrix inputM, integer greater than one
    # inputM: a square and symmetric matrix, numpy array
    # tol: singularity tolerance, positive real

    aliasParam = []
    nonAliasParam = []

    A = numpy.copy(inputM)
    diagA = numpy.diagonal(inputM)

    for k in range(pDim):
        Akk = A[k,k]
        if (Akk >= (tol * diagA[k])):
            nonAliasParam.append(k)
            ANext = A - numpy.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / Akk
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:, k] = 0.0 * A[:, k]
            ANext[k, :] = ANext[:, k]
        A = ANext
    return (A, aliasParam, nonAliasParam)

def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pd.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

# Define a function that performs the Chi-square test
def ChiSquareTest (
    xCat,           # input categorical feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):

    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))
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
    cramerV = numpy.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig, cramerV)

def question_1():
    purchasell = pd.read_csv('Purchase_Likelihood.csv')
    purchasell = purchasell.dropna()
    y = purchasell['insurance'].astype('category')
    gs = pd.get_dummies(purchasell[['group_size']].astype('category'))
    ho = pd.get_dummies(purchasell[['homeowner']].astype('category'))
    mc = pd.get_dummies(purchasell[['married_couple']].astype('category'))
    
    devianceTable = pd.DataFrame()
    
    # intercept only
    designX = pd.DataFrame(y.where(y.isnull(), 1))
    LLK0, DF0, fullParams0, thisFit, nonAlias = build_mnlogit(designX, y)
    devianceTable = devianceTable.append([[0, 'Intercept', DF0, LLK0, None, None, None]])
    
    # + group_size
    designX = stats.add_constant(gs, prepend=True)
    LLK1, DF1, fullParams1, thisFit, nonAlias = build_mnlogit (designX, y)
    testDev = 2.0 * (LLK1 - LLK0)
    testDF = DF1 - DF0
    testPValue = scipy.stats.chi2.sf(testDev, testDF)
    fe1 = -(numpy.log10(testPValue))
    devianceTable = devianceTable.append([[1, 'gs',
                                           DF1, LLK1, testDev, testDF, testPValue]])
    # + homeowner
    designX = gs
    designX = designX.join(ho)
    designX = stats.add_constant(designX, prepend = True)
    LLK1, DF1, fullParams1, thisFit, nonAlias = build_mnlogit (designX, y)
    testDev = 2.0 * (LLK1 - LLK0)
    testDF = DF1 - DF0
    testPValue = scipy.stats.chi2.sf(testDev, testDF)
    fe2 = -(numpy.log10(testPValue))
    devianceTable = devianceTable.append([[2, 'gs + ho',
                                           DF1, LLK1, testDev, testDF, testPValue]])
    # + married_couple
    designX = gs
    designX = designX.join(ho)
    designX = designX.join(mc)
    designX = stats.add_constant(designX, prepend = True)
    LLK1, DF1, fullParams1, thisFit, nonAlias = build_mnlogit (designX, y)
    testDev = 2.0 * (LLK1 - LLK0)
    testDF = DF1 - DF0
    testPValue = scipy.stats.chi2.sf(testDev, testDF)
    fe3 = -(numpy.log10(testPValue))
    devianceTable = devianceTable.append([[3, 'gs + ho + mc',
                                           DF1, LLK1, testDev, testDF, testPValue]])
    # + group_size * homeowner
    designX = gs
    designX = designX.join(ho)
    designX = designX.join(mc)
    # gs*ho
    gsho = create_interaction(gs, ho)
    designX = designX.join(gsho)
    designX = stats.add_constant(designX, prepend = True)
    LLK1, DF1, fullParams1, thisFit, nonAlias = build_mnlogit (designX, y)
    testDev = 2.0 * (LLK1 - LLK0)
    testDF = DF1 - DF0
    testPValue = scipy.stats.chi2.sf(testDev, testDF)
    fe4 = -(numpy.log10(testPValue))
    devianceTable = devianceTable.append([[4, 'gs + ho + mc + gs*ho',
                                           DF1, LLK1, testDev, testDF, testPValue]])
    # + group_size * married_couple
    designX = gs
    designX = designX.join(ho)
    designX = designX.join(mc)
    # gs*ho
    gsho = create_interaction(gs, ho)
    designX = designX.join(gsho)
    # gs*mc
    gsmc = create_interaction(gs, mc)
    designX = designX.join(gsmc)
    designX = stats.add_constant(designX, prepend = True)
    LLK1, DF1, fullParams1, thisFit, nonAlias = build_mnlogit (designX, y)
    testDev = 2.0 * (LLK1 - LLK0)
    testDF = DF1 - DF0
    testPValue = scipy.stats.chi2.sf(testDev, testDF)
    fe5 = -(numpy.log10(testPValue))
    devianceTable = devianceTable.append([[5, 'gs + ho + mc + gs*ho + gs*mc',
                                           DF1, LLK1, testDev, testDF, testPValue]])
    # + homeowner * married_couple
    designX = gs
    designX = designX.join(ho)
    designX = designX.join(mc)
    # gs*ho
    gsho = create_interaction(gs, ho)
    designX = designX.join(gsho)
    # gs*mc
    gsmc = create_interaction(gs, mc)
    designX = designX.join(gsmc)
    # ho*mc
    homc = create_interaction(ho, mc)
    designX = designX.join(homc)
    designX = stats.add_constant(designX, prepend = True)
    LLK1, DF1, fullParams1, thisFit, alias = build_mnlogit (designX, y)
    testDev = 2.0 * (LLK1 - LLK0)
    testDF = DF1 - DF0
    testPValue = scipy.stats.chi2.sf(testDev, testDF)
    fe6 = -(numpy.log10(testPValue))
    devianceTable = devianceTable.append([[6, 'gs + ho + mc + gs*ho + gs*mc + ho*mc',
                                           DF1, LLK1, testDev, testDF, testPValue]])
    
    devianceTable = devianceTable.rename(columns = {0:'Sequence', 1:'Model Specification',
                                                2:'Number of Free Parameters', 3:'Log-Likelihood',
                                                4:'Deviance', 5:'Degree of Freedom', 6:'Significance'})
    print("----Part a ----")
    # or return the nonalias in function -- check
    #non_alias_params = []
    #for col in nonAlias.columns:
    #    non_alias_params.append(col)
    #print(non_alias_params)
    # bruh, wants the ALIAS, not the nonalias
    X = designX.iloc[:, list(alias)]
    alias_col = []
    for col in X.columns:
        alias_col.append(col)
    print(alias_col)
    
    print("---- Part b----")
    print(testDF, "Degrees of freedom")
    
    print("---- Part c----")
    print(devianceTable)
    
    print("---- Part d----")
    print('gs:', fe1)
    print('gs + ho:', fe2)
    print('gs + ho + mc:', fe3)
    print('gs + ho + mc + gs*ho:', fe4)
    print('gs + ho + mc + gs*ho + gs*mc:', fe5)
    print('gs + ho + mc + gs*ho + gs*mc + ho*mc:', fe6)
    
def question_2():
    purchasell = pd.read_csv('Purchase_Likelihood.csv')
    purchasell = purchasell.dropna()
    
    print("----Part a----")
    frequen = purchasell.groupby('insurance').size()
    atable = pd.DataFrame(columns = ['freq', 'prob'])
    atable['freq'] = frequen
    atable['prob'] = atable['freq']/purchasell.shape[0]
    print(atable)
    
    print("----Part b----")
    RowWithColumn(purchasell['insurance'], purchasell['group_size'], 'ROW')

    print("----Part c----")
    RowWithColumn(purchasell['insurance'], purchasell['homeowner'])
    
    print("----Part d----")
    RowWithColumn(purchasell['insurance'], purchasell['married_couple'])
    
    print('----Part e----')
    chisqstat, chisqdf, chisqsig, cramerV = ChiSquareTest(purchasell['group_size'],
                                                          purchasell['insurance'])
    print('Cramer for group_size:', cramerV)
    return
    chisqstat, chisqdf, chisqsig, cramerV = ChiSquareTest(purchasell['homeowner'],
                                                          purchasell['insurance'])
    print('Cramer for homeowner:', cramerV)
    chisqstat, chisqdf, chisqsig, cramerV = ChiSquareTest(purchasell['married_couple'],
                                                          purchasell['insurance'])
    print('Cramer for married_couple:', cramerV)
    
    print('----Part f----')
    # hmm bernoulliNB or multinomialNB
    # no smoothing so alpha = 0
    x = purchasell[['group_size', 'homeowner', 'married_couple']].astype('category')
    y = purchasell['insurance'].astype('category')
    # alpha too small if made it to 0 -- so made it to 1.0e-10
    classifier = naive_bayes.MultinomialNB(alpha = 1.0e-10).fit(x, y)
    gs = [1,2,3,4]
    ho = [0,1]
    mc = [0,1]
    test = []
    for g in gs:
        for h in ho:
            for m in mc:
                test.append([g, h, m])
    test = pd.DataFrame(test, columns = ['group_size', 'homeowner', 'married_couple'])
    test = test[['group_size', 'homeowner', 'married_couple']].astype('category')
    test_prob = classifier.predict_proba(test)
    testprob = pd.DataFrame(test_prob, columns = ['Prob(insurance=0)',
                                                  'Prob(insurance=1)',
                                                  'Prob(insurance=2)'])
    t = pd.concat([test, testprob], axis = 1)
    print(t)
    
    print('----Part g----')
    t['Prob(insurance=1)/Prob(insurance=0)'] = t['Prob(insurance=1)']/t['Prob(insurance=0)']
    print(t)

if __name__ == '__main__':
    #question_1()
    question_2()
    