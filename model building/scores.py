from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
import numpy as np





class Score():
    
    def adj_r2(self,x,y,r2):
        """
        Methode: adj_r2
        Description: Caculate the adjusted r2 score
        Input: x, y, r2
        Output: adj_r2
        on falure: log error

        Version: 1.0
        """
        try:
            train_adj_r2 = 1 - (1-r2)*(len(y)-1)/(len(y)-x.shape[1]-1)
            return train_adj_r2
        except Exception as e:
            print(e)
            
    
    
    def evaluation_r2_score(self,act,pred):
        """
        Methode: evaluation_r2_score
        Description: Caculate the r2 score
        Input: actual values, predicted values
        Output: r2
        on falure: log error

        Version: 1.0
        """
        try:
            r2_sc = r2_score(act,pred)
            return r2_sc
        except Exception as e:
            print(e)
    
    
    def mae(self,act,pred):
        """
        Methode : mae
        Description: Caculate the mean absolute error
        Input: actual values, predicted values
        Output: mae
        on falure: log error

        Version: 1.0
        """
        try:
            mae = mean_absolute_error(act,pred)
            return mae
        except Exception as e:
            print(e)

    
   
   
    def rmse(self,act,pred):
        """
        Methode: rmse
        Description: Caculate the root mean squared error
        Input: actual values, predicted values
        Output: rmse
        on falure: log error

        Version: 1.0
        """
        try:
            mse = mean_squared_error(act,pred)
            rmse = np.sqrt(mse)
            return rmse
        except Exception as e:
            print(e)
    
    
    
    def cv_score(self,obj,X,Y):
        """
        Methode: cv_score
        Description: Caculate the cross validation score
        Input: obj, X, Y
        Output: cv_score
        on falure: log error

        Version: 1.0
        """
        try:
            cv_Score = cross_val_score(obj,X,Y,cv = 10,n_jobs = -1)
            return round(np.mean(cv_Score),2) # returning mean of cv_score
        except Exception as e:
            print(e)