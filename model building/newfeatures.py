import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings


from loading import Loading_raw
from missing_values import Missing_Value

class NewFeatures:
    '''Class creating for New Features
    '''
    def __init__(self):
       
        self.df_obj = Missing_Value()
    

    def TrainNewfeatures(self):
        '''Meathod for adding new features deriving
        from already existed features

        input: None
        
        output: return train_data as pandas DataFrame
        '''

        train = self.df_obj.trainmissingvalues()
        def hello(s):
            if s<=67.5:
                return 0
            elif (s>67.5) & (s<=134.5):
                return 1
            elif (s>134.5) & (s<=201.1):
                return 2
            else:
                return 3
        train['MRP_bins']=train['Item_MRP'].apply(hello)
        train['new_out']=train['Outlet_Identifier'].str.split('0').str.get(1).astype('int')
        train['total']=2021-train['Outlet_Establishment_Year']
        train['new_item']=train['Item_Identifier'].str[-2:].astype('int')
        train['Item_category']=train['Item_Identifier'].str[:2]
        return train

    def TestNewfeatures(self):
        '''Meathod for adding new features deriving
        from already existed features

        input: None
        
        output: return test_data as pandas DataFrame
        '''
        test = self.df_obj.testmissingvalues()
        test['new_item']=test['Item_Identifier'].str[-2:].astype('int')
        def hello(s):
            if s<=67.5:
                return 0
            elif (s>67.5) & (s<=134.5):
                return 1
            elif (s>134.5) & (s<=201.1):
                return 2
            else:
                return 3    
        test['MRP_bins']=test['Item_MRP'].apply(hello)
        test['new_out']=test['Outlet_Identifier'].str.split('0').str.get(1).astype('int')
        test['total']=2021-test['Outlet_Establishment_Year']
        test['Item_category']=test['Item_Identifier'].str[:2]
        return test

