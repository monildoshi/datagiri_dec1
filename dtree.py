# Step 1 - Import required libraries

from __future__ import division # for python 2 only
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
from datetime import datetime

# Step 2 - Load the data
data_df = pd.read_csv('loan_application.csv')
sample_df = pd.read_csv('manual_calculation.csv')

# Step 3 - Perform preprocessing steps

def relevant_data(data_df):
    """
    Returns only the required columns from the data set
    :param data_df: raw pandas data frame
    :return: pandas data frame with relevant columns
    """
    data_df = data_df.drop('Application_ID', axis=1)
    return data_df

def cat2int(data_df):
    """
    Converts categorical values in to discret numeric values
    :param data_df: raw data frame
    :return: data frame with categorical converted to numerics
    """

    data_df['Dependents'] = data_df['Dependents'].map(
        lambda x: 4 if x == '3+' else int(x))

    data_df['Gender'] = data_df['Gender'].map(lambda x: 0 if x == 'No' else 1)

    data_df['Education'] = data_df['Education'].map(
        lambda x: 0 if x == 'Not Graduate' else 1)

    data_df['Married'] = data_df['Married'].map(
        lambda x: 0 if x == 'No' else 1)

    data_df['Property_Area'] = data_df['Property_Area'].map(
        lambda x: 0 if x == 'Urban' else 1 if x == 'Semiurban' else 2)

    data_df['Income'] = data_df['Income'].map(
        lambda x: 0 if x == 'low' else 1 if x == 'medium' else 2)

    data_df['Self_Employed'] = data_df['Self_Employed'].map(
        lambda x: 0 if x == 'No' else 1)

    return data_df

def get_x_y(data_df):
    """
    Returns X and y i.e. predictors and target variale from data set
    :param data_df: raw data frame
    :return: 2 pandas data frames
    """

    X = data_df.drop('Application_Status', axis=1)
    y = data_df.loc[:, 'Application_Status']

    return X, y

# Step 4 - Perform manual Gini calculation

def get_probablities(data_df, variable):
    """
    Provides probablities for Y and N outcomes for a given variable
    :param
        variable: Column name / Predictors
        data_df: raw pandas dataframe
    :return: pandas dataframe with count, probabilities, squared probabilities
             and probabilities multiplied by its log
    """

    # Count of every Y and N for variable's different values
    count = pd.DataFrame(data_df.groupby([variable, 'Application_Status'])[
                             'Application_Status'].count())
    count.columns = ['count']

    # Count of every Y and N for the whole subset
    target_count = pd.DataFrame(data_df.groupby(variable)[
                                    'Application_Status'].count())
    target_count.columns = ['target_count']
    target_count['target_weight'] = target_count['target_count'].map(
        lambda x: x / target_count['target_count'].sum())

    count = count.merge(target_count, left_index=True, right_index=True,
                        how='left')

    # Probability of every Y and N for variable's different values
    prob = pd.DataFrame(data_df.groupby([variable, 'Application_Status'])[
                            'Application_Status'].count()).groupby(level=0).\
        apply(lambda x: x / float(x.sum())).round(3)
    prob.columns = ['prob']

    # Merging these 2 dataframes
    result_df = count.merge(prob, left_index=True, right_index=True)

    result_df['sqrd_prob'] = result_df['prob'].map(lambda x: x**2)

    result_df['log_prob'] = result_df['prob'].map(lambda x: x*np.log2(x))

    # Calculate Gini Index for individual variable's outcomes
    gini_resp = pd.DataFrame(result_df.groupby(level=0).
                             apply(lambda x: 1 - float(x.sqrd_prob.sum())))
    gini_resp.columns = ['gini_respective']

    result_df = result_df.merge(gini_resp, left_index=True, right_index=True,
                                how='left')

    # Calculate Entropy for individual variable's outcomes
    entropy_resp = pd.DataFrame(result_df.groupby(level=0).
                                apply(lambda x: -1*float(x.log_prob.sum())))
    entropy_resp.columns = ['entropy_resp']

    result_df = result_df.merge(entropy_resp, left_index=True,
                                right_index=True, how='left')

    return result_df.round(3)


def get_gini_index(data_df):
    """
    Provides Gini Index for every variable except for unique App ID and target
    :param data_df: your test / train dataset
    :return: pandas dataframe with gini score for each variable
    """

    # Initiate a list to save the results
    gini_ls = []

    # Iterate over every columns and get its probabilities
    for col in ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Credit_History', 'Property_Area', 'Income']:
        data = get_probablities(data_df, col)

        # Retaining only required columns
        data = data[['target_weight', 'gini_respective']].drop_duplicates()

        # Calculate Gini for the variable and append results to list
        gini = pd.DataFrame(data['target_weight']*data['gini_respective']). \
            sum()[0]
        gini_ls.append((col, gini))

    res_df = pd.DataFrame(gini_ls, columns=['Variable', 'Gini']).round(2)

    return res_df.sort_values('Gini')

# Step 5 - Decision Trees using Scikit-Learn

def run_dtree(data_df, method='gini', max_depth=None, min_samples_leaf=1):
    """
    Provides predictions made by decision tree and prints the accuracy score
    on test dataset
    :param method: criterion for split. Options:
                   gini
                   entropy
    :return: numpay array of predictions
    """

    X, y = get_x_y(data_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=7)

    clf = tree.DecisionTreeClassifier(criterion=method, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print('Your accuracy is {}'.format(accuracy_score(y_test, y_pred)))

    return round((accuracy_score(y_test, y_pred)), 2)


if __name__ == "__main__":
    app_df = relevant_data(data_df)
    app_df = cat2int(app_df)

    print(test)
    '''
    method_ls = ['gini', 'entropy']
    max_depth_ls = [None, 3, 4]
    min_sample_ls = [1, 10, 20]

    params_ls = []
    for i in product(method_ls, max_depth_ls, min_sample_ls):
        params_ls.append(list(i))

    res_ls = []

    run = 1
    for params in params_ls:
        run = run
        method = params[0]
        max_depth = params[1]
        min_sample = params[2]
        print('Running experiment # {} using {} as splitting criterion and ' \
              'with max depth of {} & min sample leaf of {}'.format(run,
                method, max_depth, min_sample))
        accuracy = run_dtree(app_df, method=method, max_depth=max_depth,
                             min_samples_leaf=min_sample)

        res_ls.append((run, method, max_depth, min_sample, accuracy))
        run +=1

    fin_df = pd.DataFrame(res_ls, columns=['Run No.', 'Mehtod', 'Max Depth',
                                           'Min Samples Leaf', 'Accuracy'])

    print(fin_df.head())'''