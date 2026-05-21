import os
import datetime
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

from loguru import logger
from pathlib import Path
from dateutil.relativedelta import relativedelta

from typing import Dict

Cache_path = '/Users/qianhonglin/Desktop/Quant full time/Digital Currency vol prediction/ModelCache'

RE_BASE_ENCODING = {0:'BTC', 1:'ETH'}

class RVForecasting_GBDT:
    
    def __init__(self, horizon: str, lag:int, root_dir:str=Cache_path, percentage_error:bool = True):
        self.root_dir = Path(root_dir) # path of model saving
        self.horizon = horizon # 24h
        self.lag = lag # delay
        self.model = None
        self.model_folder = os.path.join(root_dir, f'RV_{self.horizon}_Models') # save model file
        self.model_prefix = "RVLight" #prefix
        self.percentage_error = percentage_error # weighted loss function
    
    
    def get_model_name(self, model_cutoff:datetime.datetime)->str:
        model_name = model_cutoff.strftime("%Y%m%d%H%M%S")
        if self.model_prefix:
            model_name = self.model_prefix + "_" + model_name
        return model_name
    
    
    def get_model_path(self, model_cutoff:datetime.datetime)->str:
        model_name = self.get_model_name(model_cutoff)
        model_path = os.path.join(self.model_folder, model_name)
        return model_path
    
    
    def save_model(self, model:lgb.Booster, model_cutoff:datetime)->None:
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        model_path = self.get_model_path(model_cutoff)
        model_str = model.model_to_string()
        with open(model_path, "wb") as f:
            pickle.dump(model_str, f)  
    
    
    def is_model_existed(self, model_cutoff:datetime.datetime)->bool: # model_cutoff: using current_model_cutoff in rv_signal.py
        model_path = self.get_model_path(model_cutoff)
        return os.path.exists(model_path)
    
    
    def load_model(self, model_cutoff)->lgb.Booster:
        model_path = self.get_model_path(model_cutoff)
        with open(model_path, 'rb') as f:
            model_str = pickle.load(f)
        model = lgb.Booster(model_str = model_str)
        return model

    
    def FindModel(self, date:datetime.datetime):
        pred_start = (date.normalize() - pd.offsets.MonthEnd(1) + pd.Timedelta(days=self.lag)).strftime('%Y%m%d')
        pred_end = (date.normalize() + pd.offsets.MonthEnd(0) + pd.Timedelta(days=self.lag)).strftime('%Y%m%d')
        model_name = f'RV_{self.horizon}_{pred_start}_{pred_end}'
        querry = self.model_path.glob(f'{model_name}.txt')
        try:
            next(querry)
        except StopIteration:
            logger.error(f"No model called {model_name}")
            return False
        else:
            return True

        
    def LoadModel(self, date:datetime.datetime): 
        pred_start = (date.normalize() - pd.offsets.MonthEnd(1) + pd.Timedelta(days=self.lag)).strftime('%Y%m%d')
        pred_end = (date.normalize() + pd.offsets.MonthEnd(0) + pd.Timedelta(days=self.lag)).strftime('%Y%m%d')
        model_name = f'RV_{self.horizon}_{pred_start}_{pred_end}'
        model = lgb.Booster(model_file=self.model_path.joinpath(f'{model_name}.txt'))
        self.model = model
    
    # X（特征矩阵，类型为numpy数组），y（目标变量，类型为numpy数组），以及params（一个包含训练参数的字典）。
    def train(self, X:np.array, y:np.array, params:Dict):
        
        # 这行代码根据self.percentage_error的值计算样本权重。如果self.percentage_error为真，则使用1 / (np.square(y) + 0.001)来计算每个样本的权重；这通常用于处理百分比误差。如果self.percentage_error为假，则所有样本的权重被设置为1。
        train_weights = 1 / (np.square(y) + 0.001) if self.percentage_error else [1.0] * len(y)
        
        # 这行代码创建了一个LightGBM的Dataset对象，使用提供的特征矩阵X、目标变量y、计算出的样本权重train_weights，并指定了一个名为base的分类特征。
        train_dataset = lgb.Dataset(X, y, weight = train_weights, categorical_feature=['base'])
        lgb_params = params.copy()
        lgb_params.pop("n_iters") # 从参数字典中移除键n_iters。n_iters可能是用于指定训练迭代次数的参数，但它不是LightGBM直接使用的参数，所以在调用LightGBM的train方法前将其移除。
        model = lgb.train(
            params = lgb_params,
            num_boost_round = params['n_iters'],
            train_set = train_dataset,
            valid_sets = [train_dataset], # 指定验证数据集列表，在这个例子中，训练数据集同时被用作验证数据集。
            callbacks = [lgb.log_evaluation(250)], #  设置回调函数，这里指定每250轮迭代后记录一次评估日志。
            categorical_feature = ["base"] # 
        )
        return model
    
   
    
    def predict(self, data:np.array, model:lgb.Booster):
        res = model.predict(data)
        
        # 这一行首先通过np.vstack将预测结果和data中的"base"列的值堆叠成一个新的二维数组，然后转换成pd.DataFrame。这个DataFrame有两列：'predictions'和'base'，并且使用data的索引作为DataFrame的索引。
        res = pd.DataFrame(np.vstack([res, data["base"].values]).T, columns=['predictions', 'base'], index = data.index)
        # 这一行将"base"列设置为索引的一部分，append=True表示"base"将作为一个新的层级添加到现有的索引上，而不是替换它。
        res = res.set_index("base", append = True)
        
        #使用unstack方法将数据重塑，这里的目标是将"base"索引层级的不同值转换为列，每个列对应"base"的一个唯一值，而行则对应原始的索引。"predictions"列的值被分布到这些新的列中。
        res = res["predictions"].unstack()
        res.columns = [RE_BASE_ENCODING[int(x)] for x in res.columns] #这一行重新编码列名。假设RE_BASE_ENCODING是一个字典，用于将"base"的原始编码（现在作为列名）映射到新的编码或名称。int(x)是必要的，因为列名（来自"base"的值）可能是数字。
        return res 