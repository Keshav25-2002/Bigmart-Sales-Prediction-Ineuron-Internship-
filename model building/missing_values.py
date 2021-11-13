import numpy as np
import pandas as pd
from loading import Loading_raw


class Missing_Value:
    """Class for treating missing values
    """
    def __init__(self):
       
        self.df_object = Loading_raw()

    def trainmissingvalues(self):
        '''Meathod for dealing with missing values


        input: None
        
        output: return train_data as pandas DataFrame
        '''
        
        train = self.df_object.load_train()
        train['Item_Weight'].fillna(0,inplace = True)
        
        ind = train[train['Item_Weight'] == 0].index
        for i in ind:
            it=train.iloc[i,0]
            train.iloc[i,1]=np.mean(train[train['Item_Identifier']==it]['Item_Weight'])
        
        train['Item_Weight'].replace(0,np.mean(train['Item_Weight']),inplace=True)
        train['Outlet_Size'].fillna('Missing',inplace=True)
        train['Outlet_Size'].replace('Missing','Small',inplace=True)
        
        
        train['Item_Fat_Content'].replace(['Low Fat','Regular','LF','reg','low fat'],['LF','REG','LF','REG','LF'],inplace=True)
        ind=train[train['Item_Visibility']==0].index
        
        for i in ind:
            it=train.iloc[i,1]
            train.iloc[i,3]=np.mean(train[train['Item_Weight']==it]['Item_Visibility'])
        
        return train
        
        
      


    def testmissingvalues(self):
        '''Meathod for dealing with missing values


        input: None
        
        output: return test_data as pandas DataFrame
        '''
        
        test=self.df_object.load_test()
        test['Item_Weight'].fillna(0,inplace=True)
        
        ind=test[test['Item_Weight']==0].index
        
        for i in ind:
            it=test.iloc[i,0]
            test.iloc[i,1]=np.mean(test[test['Item_Identifier']==it]['Item_Weight'])
        
        test['Item_Weight'].replace(0,np.mean(test['Item_Weight']),inplace=True)
        
        test['Outlet_Size'].fillna('Missing',inplace=True)
        
        test['Outlet_Size'].replace('Missing','Small',inplace=True)
        
        
        test['Item_Fat_Content'].replace(['Low Fat','Regular','LF','reg','low fat'],['LF','REG','LF','REG','LF'],inplace=True)
        
        ind=test[test['Item_Visibility']==0].index
        
        for i in ind:
            it=test.iloc[i,1]
            test.iloc[i,3]=np.mean(test[test['Item_Weight']==it]['Item_Visibility'])
        
        
        return test


    