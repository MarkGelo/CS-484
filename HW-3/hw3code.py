import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.metrics as metrics
from itertools import combinations
pd.options.mode.chained_assignment = None  # default='warn'

train_perc = 0.7
test_perc = 0.3
claim_history = pd.read_csv('claim_history.csv',
                            delimiter=',')
train_data, test_data = train_test_split(claim_history, 
                                         test_size = test_perc, 
                                         random_state = 60616, 
                                         stratify = claim_history["CAR_USE"])
# education ordinal
train_data['EDUCATION'] = train_data['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})
test_data['EDUCATION'] = test_data['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

def question_4():
    print("----Part a----")
    train_data_inf = train_data.groupby('CAR_USE').size()
    # 2 - commercial and private
    train_comm = train_data_inf[0]
    train_comm_prob = train_comm/train_data.shape[0]
    train_pri = train_data_inf[1]
    train_pri_prob = train_pri/train_data.shape[0]
    print("Training Data")
    print(f"Commercial: {train_comm}, {train_comm_prob}")
    print(f"Private: {train_pri}, {train_pri_prob}")
    
    print("----Part b----")
    test_data_inf = test_data.groupby('CAR_USE').size()
    test_pri = test_data_inf[1]
    p_pri = (train_pri + test_pri) / (train_data.shape[0] + test_data.shape[0])
    p_pri_test = test_data_inf[1] / test_data.shape[0]
    p_testing_given_pri = (p_pri_test * test_perc) / p_pri
    print("Probability:", p_testing_given_pri)

# functions for q5 and q6
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

def FindMinEN(data, sett):
    subset_map = {}
    # tries out all combinations
    for i in range(1, (int(len(sett) / 2)) + 1):
        subsets = combinations(sett, i)
        for ss in subsets:
            remaining = tuple()
            for ele in sett:
                if ele not in ss:
                    remaining += (ele,)
            if subset_map.get(remaining) == None:
                subset_map[ss] = remaining
    min_entropy = 99999999999.99999999999
    min_subset1 = min_subset2 = min_table = None
    for subsett in subset_map:
        table, entropy = split(data, typ = 'EN', subset = subsett)
        if entropy < min_entropy:
            min_entropy = entropy
            min_subset1 = subsett
            min_subset2 = subset_map.get(subsett)
            min_table = table
    return min_table, min_entropy, min_subset1, min_subset2

def question_5():
    print("----Part a----")
    p_com_train = train_data.groupby('CAR_USE').size()['Commercial'] / train_data.shape[0]
    p_pri_train = train_data.groupby('CAR_USE').size()['Private'] / train_data.shape[0]
    root_entropy = -((p_com_train * math.log2(p_com_train)) + (p_pri_train * math.log2(p_pri_train)))
    print(f'Entropy of root node : {root_entropy}')
    
    print("----Part b and c and d----")
    '''
    classifier_tree = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = 2)
    X = train_data.drop('CAR_USE', axis = 1)
    Y = train_data[['CAR_USE']][:]
    X = X[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']]
    
    Encode = LabelEncoder()
    for i in list(X.columns.values):
            X[str(i)] = Encode.fit_transform(X[str(i)])
    dec_tree = classifier_tree.fit(X, Y)
    tree.plot_tree(dec_tree)
    '''
    # trying out all vars and finding the var with min entropy
    education_split = FindMinEO(train_data[['EDUCATION', 'CAR_USE']],
                                       [0, 1, 2, 3, 4])
    car_type_split = FindMinEN(train_data[['CAR_TYPE', 'CAR_USE']], 
                                           ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 
                                            'Sports Car', 'Van'])
    occupation_split = FindMinEN(train_data[['OCCUPATION', 'CAR_USE']],
                                 ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 
                                  'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])
    print('----First layer Split----')
    print(f"Education: {education_split[1]}")
    print(f"Car Type: {car_type_split[1]}")
    print(f"Occupation: {occupation_split[1]}")
    # occupation lowest entropy
    print(f"""Occupation for first split with entropy of {occupation_split[1]} and branches 
          with values of {occupation_split[2]} and {occupation_split[3]}""")
    
    # left split -- find min again
    # split data by the splits -- 2 3 in occupation split
    train_left_split = train_data[train_data['OCCUPATION'].isin(occupation_split[2])]
    left_educ_split = FindMinEO(train_left_split[['EDUCATION', 'CAR_USE']],
                                [0, 1, 2, 3, 4])
    left_car_split = FindMinEN(train_left_split[['CAR_TYPE', 'CAR_USE']], 
                                           ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 
                                            'Sports Car', 'Van'])
    left_oc_split = FindMinEN(train_left_split[['OCCUPATION', 'CAR_USE']],
                                 ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 
                                  'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])
    print("----Left Split----")
    print(f"Education: {left_educ_split[1]}")
    print(f"Car Type: {left_car_split[1]}")
    print(f"Occupation: {left_oc_split[1]}")
    # left split is Education with min
    print(f"""Education has minimum with entropy of {left_educ_split[1]} with 
          a split at {left_educ_split[2]} which results in (Below High School) and 
          (High School, Bachelors, Masters, Doctors)""")
    # right split)
    train_right_split = train_data[train_data['OCCUPATION'].isin(occupation_split[3])]
    right_educ_split = FindMinEO(train_right_split[['EDUCATION', 'CAR_USE']],
                                [0, 1, 2, 3, 4])
    right_car_split = FindMinEN(train_right_split[['CAR_TYPE', 'CAR_USE']], 
                                           ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 
                                            'Sports Car', 'Van'])
    right_oc_split = FindMinEN(train_right_split[['OCCUPATION', 'CAR_USE']],
                                 ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 
                                  'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])
    print("----Right Split----")
    print(f"Education: {right_educ_split[1]}")
    print(f"Car Type: {right_car_split[1]}")
    print(f"Occupation: {right_oc_split[1]}")
    # left split is Education with min
    print(f"""Car Type has minimum with entropy of {right_car_split[1]} with 
          {right_car_split[2]} and {right_car_split[3]}""")
    print("-"*10)
    # leave 1 - left left
    leave1_data = train_left_split[train_left_split['EDUCATION'] <= left_educ_split[1]]
    print("Leave 1")
    print(f"{occupation_split[2]} -> (Below High School)")
    print(f"Total Count: {leave1_data.shape[0]}")
    print(f"Commercial: {leave1_data.groupby('CAR_USE').size()['Commercial']}")
    print(f"Private: {leave1_data.groupby('CAR_USE').size()['Private']}")
    print()
    # leave 2 - left right
    leave2_data = train_left_split[train_left_split['EDUCATION'] > left_educ_split[1]]
    print("Leave 2")
    print(f"{occupation_split[2]} -> (High School, Bachelors, Masters, Doctors)")
    print(f"Total Count: {leave2_data.shape[0]}")
    print(f"Commercial: {leave2_data.groupby('CAR_USE').size()['Commercial']}")
    print(f"Private: {leave2_data.groupby('CAR_USE').size()['Private']}")
    print()
    # leave 3 - right left
    leave3_data = train_right_split[train_right_split['CAR_TYPE'].isin(right_car_split[2])]
    print("Leave 3")
    print(f"{occupation_split[3]} -> {right_car_split[2]}")
    print(f"Total Count: {leave3_data.shape[0]}")
    print(f"Commercial: {leave3_data.groupby('CAR_USE').size()['Commercial']}")
    print(f"Private: {leave3_data.groupby('CAR_USE').size()['Private']}")
    print()
    # leave 4 - right right
    leave4_data = train_right_split[train_right_split['CAR_TYPE'].isin(right_car_split[3])]
    print("Leave 4")
    print(f"{occupation_split[3]} -> {right_car_split[3]}")
    print(f"Total Count: {leave4_data.shape[0]}")
    print(f"Commercial: {leave4_data.groupby('CAR_USE').size()['Commercial']}")
    print(f"Private: {leave4_data.groupby('CAR_USE').size()['Private']}")
    print()
    
    print("----Part e and f----")
    threshold = train_data.groupby('CAR_USE').size()['Commercial']/train_data.shape[0]
    # decision tree
    p_data = np.ndarray(shape = (len(test_data), 2), dtype = float)
    idx = 0
    for i, row in test_data.iterrows():
        if row['OCCUPATION'] in ['Blue Collar', 'Student', 'Unknown']: # left
            if row['EDUCATION'] <= 0.5: # left
                # values from count of comm, pri in leaves
                p_com = leave1_data.groupby('CAR_USE').size()['Commercial'] / leave1_data.shape[0]
                p_pri = leave1_data.groupby('CAR_USE').size()['Private'] / leave1_data.shape[0]
                p_data[idx] = [p_com, p_pri]
            else: # right
                p_com = leave2_data.groupby('CAR_USE').size()['Commercial'] / leave2_data.shape[0]
                p_pri = leave2_data.groupby('CAR_USE').size()['Private'] / leave2_data.shape[0]
                p_data[idx] = [p_com, p_pri]
        else: # right
            if row['CAR_TYPE'] in ['Minivan', 'SUV', 'Sports Car']: # left
                p_com = leave3_data.groupby('CAR_USE').size()['Commercial'] / leave3_data.shape[0]
                p_pri = leave3_data.groupby('CAR_USE').size()['Private'] / leave3_data.shape[0]
                p_data[idx] = [p_com, p_pri]
            else: # right
                p_com = leave4_data.groupby('CAR_USE').size()['Commercial'] / leave4_data.shape[0]
                p_pri = leave4_data.groupby('CAR_USE').size()['Private'] / leave4_data.shape[0]
                p_data[idx] = [p_com, p_pri]
        idx += 1
    # commercial is event value
    p_data_f = p_data[:, 0]
    falsep, truep, thresholds = metrics.roc_curve(test_data[['CAR_USE']], 
                                                  p_data_f, pos_label='Commercial')
    # KS plot
    points = [x if x < 1 else np.nan for x in thresholds]
    plt.plot(points, truep, marker='o', label='True Positive', color='green')
    plt.plot(points, falsep, marker='o', label='False Positive', color='red')
    plt.title("KS Chart")
    plt.grid(True)
    plt.xlabel("Probability Threshold")
    plt.ylabel("Positive Rate")
    plt.legend(loc = 'upper right', shadow = True)
    plt.show()
    
    # find greatest difference and get threshold
    ksStatistic = 0
    ksThreshold = 0
    for i in range(len(thresholds)):
        if truep[i]-falsep[i]>ksStatistic:
            ksStatistic = truep[i]-falsep[i]
            ksThreshold = thresholds[i]
    print("Kolmogorov-Smirnov statistic:", ksStatistic)
    print("Event probability cutoff value:", ksThreshold)
    p_data_f = p_data[:, 0]
    pred_data = []
    for prob in p_data_f:
        if prob > threshold:
            pred_data.append('Commercial')
        else:
            pred_data.append('Private')
    actu_data = test_data['CAR_USE'].tolist()
    #cfn = metrics.confusion_matrix(test_data[['CAR_USE']], pred_data)
    y_actu = pd.Series(actu_data, name = 'Actual')
    y_pred = pd.Series(pred_data, name = 'Predicted')
    cfn1 = pd.crosstab(y_actu, y_pred)
    #print(cfn)
    print(cfn1)
    
def question_6():
    print("----Part a----")
    ks = 0.7369561479553026
    threshold = ks
    # decision tree
    p_data = np.ndarray(shape = (len(test_data), 2), dtype = float)
    idx = 0
    for i, row in test_data.iterrows():
        if row['OCCUPATION'] in ['Blue Collar', 'Student', 'Unknown']: # left
            if row['EDUCATION'] <= 0.5: # left
                # values from count of comm, pri in leaves
                # didnt want to copy leaves stuff, so just got the values and put in here
                p_com = 0.2681660899653979
                p_pri = 0.7318339100346021
                p_data[idx] = [p_com, p_pri]
            else: # right
                p_com = 0.8396584440227703
                p_pri = 0.1603415559772296
                p_data[idx] = [p_com, p_pri]
        else: # right
            if row['CAR_TYPE'] in ['Minivan', 'SUV', 'Sports Car']: # left
                p_com = 0.008372093023255815
                p_pri = 0.9916279069767442
                p_data[idx] = [p_com, p_pri]
            else: # right
                p_com = 0.5384615384615384
                p_pri = 0.46153846153846156
                p_data[idx] = [p_com, p_pri]
        idx += 1
    p_data_f = p_data[:, 0]
    pred_data = []
    for prob in p_data_f:
        if prob > threshold:
            pred_data.append('Commercial')
        else:
            pred_data.append('Private')
    actu_data = test_data['CAR_USE'].tolist()
    #cfn = metrics.confusion_matrix(test_data[['CAR_USE']], pred_data)
    y_actu = pd.Series(actu_data, name = 'Actual')
    y_pred = pd.Series(pred_data, name = 'Predicted')
    cfn1 = pd.crosstab(y_actu, y_pred)
    #print(cfn)
    print(cfn1)
    accuracy = metrics.accuracy_score(y_actu, y_pred)
    misclassification = 1 - accuracy
    print(f"Accuracy: {accuracy}")
    print(f"Misclassification Rate: {misclassification}")
    
    print("----Part b----")
    rase = 0
    for i in range(len(actu_data)):
        if actu_data[i] == 'Commercial':
            rase += (1 - p_data_f[i]) ** 2
        else:
            rase += (0 - p_data_f[i]) ** 2
    rase = np.sqrt(rase / len(actu_data))
    print(f"RASE: {rase}")
    
    print("----Part c----")
    test_y = test_data['CAR_USE']
    y_true = [1 if x == 'Commercial' else 0 for x in test_y]
    y_score = p_data_f
    auc = metrics.roc_auc_score(y_true, y_score)
    print("AUC: {}".format(auc))
    
    print("----Part d----")
    gini = auc * 2 - 1
    print(f"Gini Coefficient: {gini}")
    
    print("----Part e----")
    events = []
    non_events = []
    idx = 0
    for x in test_y:
        if x == 'Commercial':
            events.append(p_data_f[idx])
        else:
            non_events.append(p_data_f[idx])
        idx += 1
    concordant = 0 
    discordant = 0
    tied = 0
    for i in events:
        for j in non_events:
            if i < j:
                discordant += 1
            elif i > j:
                concordant += 1
            else:
                tied += 1
    #print(concordant, discordant, tied)
    #auc_ant = 0.5 + 0.5 * (concordant - discordant) / (discordant + concordant + tied)
    #print(auc_ant)
    goodman_kruskal = (concordant - discordant) / (concordant + discordant)
    print("Goodman Kruskal Gamma:", goodman_kruskal)    
    
    print("----Part f----")
    print("just a graph")
    # slides has this
    fpr, tpr, thresholds = metrics.roc_curve(test_data[['CAR_USE']], p_data_f, 
                                             pos_label = 'Commercial')
    # dummy coord
    oneMinusSpecificity = np.append([0], fpr)
    sensitivity = np.append([0], tpr)
    oneMinusSpecificity = np.append(oneMinusSpecificity, [1]) # ?
    sensitivity = np.append(sensitivity, [1])
    
    plt.figure(figsize = (6, 6))
    plt.plot(oneMinusSpecificity, sensitivity, marker = 'o')
    plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
    plt.grid(True)
    plt.title("ROC")
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    question_4()
    question_5()
    question_6()