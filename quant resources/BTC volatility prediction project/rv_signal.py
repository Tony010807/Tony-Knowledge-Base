import numpy as np
import pandas as pd
import os

from dateutil.relativedelta import relativedelta

data_path = "/Users/qianhonglin/Desktop/Quant full time/Digital Currency vol prediction" # 文件夹所在位置
os.chdir(data_path)

from data_module import DataModule, DataPreprocessing
from gbdt_module import RVForecasting_GBDT
from utils import get_features_importance, cal_r2, cal_corr
import warnings
warnings.filterwarnings("ignore")


# predict object，the next day's volatility
HORIZONS = ["24h"]

# delay
TRAINING_DELAY_DAYS = 3

# train dataset length
PERIODS = 720

# predict label
TARGET = 'rv_24'

# path
ROOT_DIR = '/Users/qianhonglin/Desktop/Quant full time/Digital Currency vol prediction/DC_Features_20190401_20221224.parquet'

# basic feature
COL_NAMES = ['open', 'high', 'low', 'close', 'vwap', 'twap', 'awap', 'volume','tradeval', 'ntrade', 'ntrade_up', 'ntrade_down', 'ntrade_flat', 'close_ask', 'close_bid', 'close_asksize', 'close_bidsize', 'close_spread', 'nquote']

# drop feature，these features are not stable
DROP_COLUMNS = ['open', 'high', 'low', 'close', 'vwap', 'twap', 'awap', 'volume',
    'tradeval', 'ntrade', 'ntrade_up', 'ntrade_down', 'ntrade_flat','close_ask', 'close_bid', 'close_asksize', 'close_bidsize']

# parameters
PARAMS = {
        'learning_rate': 0.0893197957786196,  # learning rate
        'max_depth': 6, # max tree depth
        'lambda_l1': 0.6598662930525132, # regularization
        'lambda_l2': 0.048533564817128005, # regularization
        'objective': 'rmse', # min rmse
        'boosting': 'gbdt', 
        'verbosity': -1, # whether output dairy
        'n_jobs':-1,  # multiprocessing number, default:-1
        'n_iters': 500 # iterates number
    }


class RVSignal:
    
    def __init__(self, train_size, horizon, importance_plot=False) -> None:
        self.importance_plot = importance_plot # different feature's contribution
        self.train_size = train_size # training set： 720
        self.horizon = horizon # future 24h
        self.pred_module = RVForecasting_GBDT(self.horizon[0], TRAINING_DELAY_DAYS) # model
       
    def rolling_train_pred(self, dataset: pd.DataFrame):
        
        begin_time, end_time = dataset.index[0], dataset.index[-1] # 2019-06-01，2022-12-01
        
        first_model_cutoff = begin_time.normalize() + pd.Timedelta(days=self.train_size) # 2021-05-21
        if first_model_cutoff.day != 1: # If it is not the 1st of every month, then it will be pushed back.
            first_model_cutoff += pd.offsets.MonthBegin() # Move date to first day of next month，2021-06-01
        
        last_model_cutoff = end_time.normalize() - pd.Timedelta(days = TRAINING_DELAY_DAYS) # 2022-11-28
        if last_model_cutoff.day != 1: # If it is not the 1st of every month, then it will be pushed forward.
            last_model_cutoff -= pd.offsets.MonthBegin() # 2022-11-01

        current_model_cutoff = first_model_cutoff # 2021-06-01
        predictions = []
        models = []
        while current_model_cutoff <= last_model_cutoff:
            
            #start & end index
            train_data_start = current_model_cutoff - pd.Timedelta(days = self.train_size) # first time：2019-06-12
            train_data_end = current_model_cutoff - pd.Timedelta(days = 1) # due to 24h rv calculation，first time：2021-05-31
            train_data_mask = (dataset.index < train_data_end) & (dataset.index >= train_data_start)
            
            train_data = dataset.loc[train_data_mask]

            test_data_start = current_model_cutoff + pd.Timedelta(days = TRAINING_DELAY_DAYS) # first time：2021-06-04
            test_data_end = test_data_start + relativedelta(months = 1) # first time：2021-07-04
            test_data_mask = (dataset.index >= test_data_start) & (dataset.index < test_data_end)

            test_data = dataset.loc[test_data_mask]
            
            train_X, train_y = train_data.drop(TARGET, axis=1), train_data[TARGET].astype('float64')
            test_X = test_data.drop(TARGET, axis=1)
            
            # make sure "train_X's corresponding train_y" and "train_y" doesn't have empty value
            train_X, train_y = train_X[np.isfinite(train_y).values].copy(), train_y[np.isfinite(train_y).values]
           
            
            
            # check if model is exist
            if self.pred_module.is_model_existed(current_model_cutoff): 
                model = self.pred_module.load_model(current_model_cutoff)
                
            else:
                model = self.pred_module.train(train_X, train_y, PARAMS)
                self.pred_module.save_model(model, current_model_cutoff)
                
            # predict (1)    
            if current_model_cutoff == first_model_cutoff: # Predict the model in-sample data, before test_data_start
                front_data_mask = dataset.index < test_data_start # we need consider the model before test_data_start
                front_data = dataset.loc[front_data_mask]
                front_X = front_data.drop(TARGET, axis=1)
                front_prediction = self.pred_module.predict(front_X, model)
                predictions.append(front_prediction)

            # predict (2)    
            if len(test_X) > 0: # Normal out-of-sample data prediction
                prediction = self.pred_module.predict(test_X, model)
                predictions.append(prediction)
            
            
            current_model_cutoff += relativedelta(months = 1) # rolling next month
            models.append(model)
        
        predictions = pd.concat(predictions)
        if self.importance_plot:
            get_features_importance(models, importance_type='gain')
            
        return predictions # Forecast volatility results for the next day
    
    
def main():
    
    freq = '1h'
    begin_time, end_time = '20190601', '20221201' # start time，end time
    
    # BTC, ETH -> DataModule
    dm = [DataModule(ROOT_DIR, freq, base) for base in ['BTC', 'ETH']]
    datasets = dict(zip(['BTC', 'ETH'], [dm_i.get_features(starttime=begin_time, endtime=end_time, columns=COL_NAMES) for dm_i in dm]))
    dp = DataPreprocessing(datasets=datasets, freq=freq)
    data = dp.data_collation(drop_columns=DROP_COLUMNS)
    
    rv_signal = RVSignal(PERIODS, HORIZONS, importance_plot=True)
    res = rv_signal.rolling_train_pred(data)
    res.to_csv('/Users/qianhonglin/Desktop/Quant full time/Digital Currency vol prediction/rv_forecasting_result.csv')
    
    ic = cal_corr(data, res)
    r2 = cal_r2(data, res)
    print(f'Prediction IC: {ic}')
    print(f'Prediction R2: {r2}')

    return res
    
    

if __name__ == '__main__':
    main()

 