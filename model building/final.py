import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from model_training import Models

class Model_Selection:
    '''Class for testing differnt algorithms
    '''
    def __init__(self):
       self.model= Models()
       self.output={}
    

    def Model_Selection(self):
        '''Meathod for testing different algorithms

        input: None

        Output: rmse values of differnt models
        '''
        #here we use different machine learning algorithms
        ex=ExtraTreesRegressor(n_estimators=700,max_depth=6,min_samples_split=28, min_samples_leaf=50,n_jobs=-1)
        gg=GradientBoostingRegressor(n_estimators=700, min_samples_leaf=55)
        lg=lgb.LGBMRegressor(max_depth=20,n_estimators=100)
        final=VotingRegressor([('a',lg),('c',ex),('b',gg)],weights=[1,1,2])
        all_models = [ex,gg,lg,final]
        model_names = ["ex","gg","lg","final"]
        for mdl,mdl2 in zip(all_models,model_names):
            
            #self.model.models(mdl)
            self.output.update(self.model.models(mdl,mdl2))
        return self.output


model= Model_Selection()
values=model.Model_Selection()
print(values)
