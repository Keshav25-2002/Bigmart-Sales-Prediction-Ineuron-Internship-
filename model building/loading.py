import pandas as pd



import warnings
warnings.filterwarnings('ignore')



class Loading_raw:
    """Class for loading raw data from local source
    """
    
    def load_train(self):
        """
        Methode: load_train

        input: None
        output: return train_data as pandas DataFrame
        on fail: return None and log error

        version: 1.0

        """
        try:
            #read data from csv
            train_data = pd.read_csv('Train.csv')
            return train_data
        except Exception as e:
           raise e
    
    def load_test(self):
        """
        Methode: load_test

        input: None
        output : return test_data as pandas DataFrame
        on fail: return None and log error

        version: 1.0
        """
        try:
            #read data from csv
            test_data = pd.read_csv('Test.csv')
            return test_data
        except Exception as e:
            raise e

