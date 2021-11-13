import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
ll=LabelEncoder()
from newfeatures import NewFeatures

class Encoding:
    '''Class For Encoding the data
    '''
    def __init__(self):
       
        self.df_obj = NewFeatures()
    
    def TrainEncoding(self):
        '''Meathod for encoding the 
        train data
        input: None
        output: return encoded train_data as pandas DataFrame'''
        train = self.df_obj.TrainNewfeatures()
        for col in train.columns[train.dtypes=='object'].drop('Item_Identifier','Outlet_Identifier'):
            train[col]=ll.fit_transform(train[col])
        train = train.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)
        return train

    def TestEncoding(self):
        '''Meathod for encoding the Test data
        input: None
        output: return encoded test_data as pandas DataFrame
        '''
        test = self.df_obj.TestNewfeatures()
        for col in test.columns[test.dtypes=='object'].drop('Item_Identifier','Outlet_Identifier'):
            test[col]=ll.fit_transform(test[col])
        test = test.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)
        return test
