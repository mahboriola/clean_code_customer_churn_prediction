"""
This module is used to test each function
from churn_library and log every test step.

Author: Marcos Aurélio Hermogenes Boriola
Date: August 10, 2022
"""

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:  # test the dataset
        df = cls.import_data('./data/bank_data.csv')
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        perform_eda(df)
        logging.info('Testing perform_eda: SUCCESS')
    except KeyError as err:
        logging.error(
            'Testing perform_eda: Some keys were not found in the dataset')
        raise err

    try:  # test if the files were created
        file_list = os.listdir('./images/eda/')
        assert len(file_list) > 0
    except AssertionError as err:
        logging.error(
            'Testing perform_eda: EDA plots not found in the output folder')
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df = cls.import_data('./data/bank_data.csv')
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        category_cols = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        df = encoder_helper(df, category_cols)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper: The dataframe doesn\'t appear to have rows and columns')
        raise err

    try:
        assert all(
            item in df.columns for item in [
                col + '_Churn' for col in category_cols])
        logging.info('Testing encoder_helper: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper: Churn-related columns were not created')
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df = cls.import_data('./data/bank_data.csv')
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, 'Churn')
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info('Testing perform_feature_engineering: SUCCESS')
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: " \
            "Some train and/or test dataframe doesn't appear to have rows and columns")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df = cls.import_data('./data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df, 'Churn')
    train_models(X_train, X_test, y_train, y_test)
    logging.info('Testing train_models: SUCCESS')

    try:
        file_list = os.listdir('./models/')
        assert len(file_list) > 0
    except AssertionError as err:
        logging.error(
            'Testing train_models: Trained model files were not found')
        raise err

    try:
        file_list = os.listdir('./images/results')
        assert len(file_list) > 0
    except AssertionError as err:
        logging.error(
            'Testing train_models: Training and validation plots were not found')
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
