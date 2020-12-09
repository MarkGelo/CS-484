import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.metrics as metrics
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

spiral = pd.read_csv('SpiralWithCluster.csv', 
                     delimiter = ",", 
                     usecols = ["x", "y", "SpectralCluster"])
x_train = spiral[['x', 'y']]
y_train = spiral["SpectralCluster"]

def question_1():
    print("----Part a----")
    total = spiral.shape[0]
    total_1 = spiral[spiral["SpectralCluster"] == 1].shape[0]
    print(f"{(total_1/total)*100}% of observations have SpectralCluster = 1")
    
    print("----Part b,c,d----")
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
    act = ["identity", "logistic", "relu", "tanh"]
    threshold = total_1/total
    for actf in act:
        #print(actf)
        for nl in np.arange(1, 6):
            for npl in np.arange(1, 11):
                nnobj = nn.MLPClassifier(hidden_layer_sizes = (npl,)*nl,
                                         activation = actf,
                                         verbose = False,
                                         solver = "lbfgs",
                                         learning_rate_init = 0.1,
                                         max_iter = 10000,
                                         random_state = 20200408)
                nnfit = nnobj.fit(x_train, y_train)
                y_pred = np.where(nnobj.predict_proba(x_train)[:,1] >= threshold,
                                  1, 0)
                misclass = 1 - (metrics.accuracy_score(y_train, y_pred))
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
    
    print("----Part e----")
    nnobj = nn.MLPClassifier(hidden_layer_sizes = (7,)*4,
                                         activation = "relu",
                                         verbose = False,
                                         solver = "lbfgs",
                                         learning_rate_init = 0.1,
                                         max_iter = 10000,
                                         random_state = 20200408)
    thisFit = nnobj.fit(x_train, y_train)
    y_pred = np.where(nnobj.predict_proba(x_train)[:,1] >= threshold,
                                  1, 0)
    spiral['Predicted'] = y_pred
    reds = spiral[spiral['Predicted'] == 0]
    plt.scatter(reds['x'], reds['y'], c="red", label = 0)
    blues = spiral[spiral['Predicted'] == 1]
    plt.scatter(blues['x'], blues['y'], c = "blue", label = 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Part e MLP - relu, 4 layers with 7 neurons each')
    plt.grid(True)
    plt.legend(title = "Predicted")
    plt.show()
    print("Just a graph")
    
def question_2():
    print("----Part a----")
    svmmodel = svm.SVC(kernel = "linear", 
                       decision_function_shape = "ovr",
                       random_state = 20200408)
    thisfit = svmmodel.fit(x_train, y_train)
    y_pred = thisfit.predict(x_train)
    spiral['Predicted'] = y_pred
    
    print("Intercept:", thisfit.intercept_)
    print("Coefficients:", thisfit.coef_)
    
    print("----Part b----")
    misclass  = 1 - metrics.accuracy_score(y_train, y_pred)
    print('Misclassification:', misclass)
    
    print("----Part c----")
    inter = thisfit.intercept_
    co1 = thisfit.coef_[0][0]
    co2 = thisfit.coef_[0][1]
    x = np.linspace(-5, 5)
    y = (-inter - co1*np.linspace(-5, 5))/co2
    reds = spiral[spiral['Predicted'] == 0]
    plt.scatter(reds['x'], reds['y'], c="red", label = 0)
    blues = spiral[spiral['Predicted'] == 1]
    plt.scatter(blues['x'], blues['y'], c = "blue", label = 1)
    plt.plot(x, y, 
             color = 'black', linestyle = 'dotted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Part c - SVM + hyperplane')
    plt.grid(True)
    plt.legend(title = "Predicted")
    plt.show()
    print("Just a graph")
    
    print("----Part d----")
    # Convert to the polar coordinates -- from profs code
    spiral['radius'] = np.sqrt(spiral['x']**2 + spiral['y']**2)
    spiral['theta'] = np.arctan2(spiral['y'], spiral['x'])
    def customArcTan (z):
        theta = np.where(z < 0.0, 2.0*np.pi+z, z)
        return (theta)
    spiral['theta'] = spiral['theta'].apply(customArcTan)
    reds = spiral[spiral['SpectralCluster'] == 0]
    plt.scatter(reds['radius'], reds['theta'], c="red", label = 0)
    blues = spiral[spiral['SpectralCluster'] == 1]
    plt.scatter(blues['radius'], blues['theta'], c = "blue", label = 1)
    plt.xlabel('radius')
    plt.ylabel('theta')
    plt.title('Part d - SVM + Polar')
    plt.grid(True)
    plt.legend(title = "Class")
    plt.show()
    print("Just a graph")
    
    print("----Part e----")
    spiral['group'] = 3
    spiral.loc[spiral['radius'] <= 2.5, 'group'] = 2
    spiral.loc[(spiral['radius'] < 3) & (spiral['theta'] >= 2), 
               'group'] = 2
    spiral.loc[(spiral['radius'] < 3.5) & (spiral['theta'] >= 3), 
               'group'] = 2
    spiral.loc[(spiral['radius'] < 4) & (spiral['theta'] >= 4), 
               'group'] = 2
    spiral.loc[(spiral['radius'] < 2) & (spiral['theta'] >= 3), 
               'group'] = 1
    spiral.loc[(spiral['radius'] < 2.5) & (spiral['theta'] >= 4), 
               'group'] = 1
    spiral.loc[(spiral['radius'] < 3) & (spiral['theta'] >= 5), 
               'group'] = 1
    spiral.loc[(spiral['radius'] < 1.5) & (spiral['theta'] >= 6), 
               'group'] = 0
    reds = spiral[spiral['group'] == 0]
    plt.scatter(reds['radius'], reds['theta'], c="red", label = 0)
    blues = spiral[spiral['group'] == 1]
    plt.scatter(blues['radius'], blues['theta'], c = "blue", label = 1)
    green = spiral[spiral['group'] == 2]
    plt.scatter(green['radius'], green['theta'], c = "green", label = 2)
    black = spiral[spiral['group'] == 3]
    plt.scatter(black['radius'], black['theta'], c = "black", label = 3)
    plt.xlabel('radius')
    plt.ylabel('theta')
    plt.title('Part e - SVM + Polar + new Group')
    plt.grid(True)
    plt.legend(title = "Class")
    plt.show()
    print("Just a graph")
    
    print("----Part f----")
    eqs = []
    for gr in [(0,1), (1,2), (2,3)]:
        temp_spiral = spiral[spiral['group'].isin(gr)]
        tx_train = temp_spiral[['radius', 'theta']]
        ty_train = temp_spiral["group"]
        svmmodel = svm.SVC(kernel = "linear", 
                           decision_function_shape = "ovr",
                           random_state = 20200408)
        thisfit = svmmodel.fit(tx_train, ty_train)
        y_pred = thisfit.predict(tx_train)
        
        print(f"SVM {gr}")
        print("Intercept:", thisfit.intercept_)
        print("Coefficients:", thisfit.coef_)
        inter = thisfit.intercept_
        co1 = thisfit.coef_[0][0]
        co2 = thisfit.coef_[0][1]
        x = np.linspace(0, 4.5)
        y = (-inter - co1*x)/co2
        eqs.append((x, y))
        print('\n')
        
    print("----Part g----")
    reds = spiral[spiral['group'] == 0]
    plt.scatter(reds['radius'], reds['theta'], c="red", label = 0)
    blues = spiral[spiral['group'] == 1]
    plt.scatter(blues['radius'], blues['theta'], c = "blue", label = 1)
    green = spiral[spiral['group'] == 2]
    plt.scatter(green['radius'], green['theta'], c = "green", label = 2)
    black = spiral[spiral['group'] == 3]
    plt.scatter(black['radius'], black['theta'], c = "black", label = 3)
    for eq in eqs:
        plt.plot(eq[0], eq[1], 
             color = 'black', linestyle = 'dotted')
    plt.xlabel('radius')
    plt.ylabel('theta')
    plt.title('Part g - SVM + Polar + new Group + hyperplane')
    plt.grid(True)
    plt.legend(title = "Class")
    plt.show()
    print("Just a graph")
    
    print("----Part h----")
    ceqs = []
    for eq in eqs:
        ceqs.append((eq[0]*np.cos(eq[1]), eq[0]*np.sin(eq[1])))
    reds = spiral[spiral['SpectralCluster'] == 0]
    plt.scatter(reds['x'], reds['y'], c="red", label = 0)
    blues = spiral[spiral['SpectralCluster'] == 1]
    plt.scatter(blues['x'], blues['y'], c = "blue", label = 1)
    for eq in ceqs:
        plt.plot(eq[0], eq[1], 
             color = 'black', linestyle = 'dotted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Part h - SVM + Polar-newGroup-hyperplane-cartesian')
    plt.grid(True)
    plt.legend(title = "Class")
    plt.show()
    print("Just a graph")

if __name__ == "__main__":
    #question_1()
    question_2()
    