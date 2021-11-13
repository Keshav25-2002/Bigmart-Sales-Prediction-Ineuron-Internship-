from sklearn.model_selection import train_test_split
from encoding import Encoding
from scores import Score

class Models:
    '''Class for model building and Training
    '''
    def __init__(self):
       
        self.Train = Encoding()
        self.dict_model={}
        self.obj_score=Score()

    def models(self,model,name):
        '''Meathod for automatic running of
        different algorithms
        input: Algorithm name as model

        output: rmse value
        '''
        train = self.Train.TrainEncoding()
        test =self.Train.TestEncoding()
        X = train.drop(columns = 'Item_Outlet_Sales')
        Y = train['Item_Outlet_Sales']
        
        x_train ,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.09,random_state=101)
        
        bagged_train_pred = 0
        bagged_test_pred = 0
        no_of_bags = 3

        for i, (trees, depth, seed) in enumerate([[500, 6, 21], [500, 7, 42], [1000, 8, 84]], start = 1):
            print('bag {} of bags {}'.format(i, no_of_bags))
            print('Model training for estimators {}, depth {},random state {}'.format(trees, depth, seed))
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            bagged_test_pred += y_pred
        bagged_test_pred = bagged_test_pred/no_of_bags
      
        y_pred=(abs(y_pred))
        r2=model.score(x_train,y_train)
        train_adj_r2=self.obj_score.adj_r2(x_train,y_train,r2)
        test_adj_r2=self.obj_score.adj_r2(x_test,y_test,r2)
        y_pred = model.predict(x_test)
        evaluation_score=self.obj_score.evaluation_r2_score(y_test,y_pred)
        mae=self.obj_score.mae(y_test,y_pred)
        rmse=self.obj_score.rmse(y_test,y_pred)
        cv_score=self.obj_score.cv_score(model,X,Y)
        self.dict_model[name]=[name,(evaluation_score+rmse)/2]
        return self.dict_model


           

    