"""
This module cointains functions to train and
test models for churn prediction, besides it generates
classification report and feature importance plots.

Author: Marcos Aurélio Hermogenes Boriola
Date: August 10, 2022
"""

import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    plt.hist(df['Churn'])
    plt.axes().set_xlabel('Churn')
    plt.axes().set_ylabel('Num. of clients')
    plt.tight_layout()
    plt.savefig('./images/eda/churn_hist.png')

    plt.figure(figsize=(20, 10))
    plt.hist(df['Customer_Age'])
    plt.axes().set_xlabel('Client age')
    plt.axes().set_ylabel('Num. of clients')
    plt.tight_layout()
    plt.savefig('./images/eda/age_hist.png')

    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.axes().set_xlabel('Marital status')
    plt.axes().set_ylabel('Num. of clients (%)')
    plt.axes().set_yticklabels([0, 10, 20, 30, 40, 50])
    plt.tight_layout()
    plt.savefig('./images/eda/marital_hist.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.tight_layout()
    plt.savefig('./images/eda/total_trans_ct_hist.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.tight_layout()
    plt.savefig('./images/eda/features_correlation.png')

    plt.figure(figsize=(20, 10))
    sns.boxplot(x='Customer_Age', y='Avg_Utilization_Ratio', data=df)
    plt.tight_layout()
    plt.savefig('./images/eda/age_utilization_boxplot.png')

    plt.figure(figsize=(20, 10))
    sns.barplot(x='Income_Category', y='Avg_Utilization_Ratio', data=df)
    plt.tight_layout()
    plt.savefig('./images/eda/income_utilization_bar.png')


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        col_list = []
        col_groups = df.groupby(col).mean()['Churn']

        for val in df[col]:
            col_list.append(col_groups[val])

        df[col + '_Churn'] = col_list

    return df


def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: [optional] string of response name, this argument could be
                        used for naming variables or index y column

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    if response:
        y = df[response]
    else:
        y = df['Churn']

    category_cols = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    df = encoder_helper(df, category_cols)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    classifiers = [{'name': 'Logistic Regression',
                    'train_preds': y_train_preds_lr,
                    'test_preds': y_test_preds_lr},
                   {'name': 'Random Forest',
                    'train_preds': y_train_preds_rf,
                    'test_preds': y_test_preds_rf}]
    for classifier in classifiers:
        plt.figure(figsize=(5, 5))
        plt.text(0.01, 1.25, str(f'{classifier["name"]} Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, classifier['test_preds'])),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(f'{classifier["name"]} Test'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, classifier['train_preds'])),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.tight_layout()
        img_name = classifier["name"].lower().replace(" ", "_")
        plt.savefig(f'./images/results/{img_name}_classification_report.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000, n_jobs=-1)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        './images/results/rf_feature_importance.png')

    # plot roc curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig('./images/results/lr_roc_curve.png')

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_,
                   X_test, y_test,
                   ax=ax, alpha=0.8)
    plt.savefig('./images/results/rf_roc_curve.png')
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/rf_lr_roc_curve.png')
    plt.clf()

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('./images/results/rf_shap_values.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':
    df_ = import_data('./data/bank_data.csv')
    df_['Churn'] = df_['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(df_)
    dataset  = perform_feature_engineering(df_, 'Churn')
    train_models(*dataset)
