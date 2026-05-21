import optuna
import numpy as np
import os

from sklearn.model_selection import train_test_split

data_path = "/Users/qianhonglin/Desktop/Quant full time/Digital Currency vol prediction" # 文件夹所在位置
os.chdir(data_path)

from gbdt_module import RVForecasting_GBDT
from data_module import DataModule, DataPreprocessing

import warnings
warnings.filterwarnings("ignore")


# 准备数据（X和y是你的特征和标签数据）

HORIZONS = ['24h']

TARGET = 'rv_24'

# 数据路径
ROOT_DIR = '/Users/qianhonglin/Desktop/Quant full time/Digital Currency vol prediction/DC_Features_20190401_20221224.parquet'

# 基础特征
COL_NAMES = ['open', 'high', 'low', 'close', 'vwap', 'twap', 'awap', 'volume','tradeval', 'ntrade', 'ntrade_up', 'ntrade_down', 'ntrade_flat', 'close_ask', 'close_bid', 'close_asksize', 'close_bidsize', 'close_spread', 'nquote']

# 需要丢弃的特征
DROP_COLUMNS = ['open', 'high', 'low', 'close', 'vwap', 'twap', 'awap', 'volume',
    'tradeval', 'ntrade', 'ntrade_up', 'ntrade_down', 'ntrade_flat','close_ask', 'close_bid', 'close_asksize', 'close_bidsize']

TRAINING_DELAY_DAYS = 3

# 划分训练集和验证集
freq = '1h'
begin_time, end_time = '20190601', '20220601'
dm = [DataModule(ROOT_DIR, freq, base) for base in ['BTC', 'ETH']]
datasets = dict(zip(['BTC', 'ETH'], [dm_i.get_features(starttime=begin_time, endtime=end_time, columns=COL_NAMES) for dm_i in dm]))
dp = DataPreprocessing(datasets=datasets, freq=freq)
data = dp.data_collation(drop_columns=DROP_COLUMNS)
X, y = data.drop(TARGET, axis=1), data[TARGET].astype('float64')
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义优化目标函数
def objective(trial):
    # 定义超参数搜索空间
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
        'objective': 'rmse',  # 固定为'rmse'
        'boosting': 'gbdt',   # 固定为'gbdt'
        'verbosity': -1,      # 固定为-1
        'n_jobs': -1,         # 固定为-1
        'n_iters': 500        # 固定为500
    }

    # 创建LightGBM回归模型
    rvf = RVForecasting_GBDT('24h', TRAINING_DELAY_DAYS)
    model = rvf.train(X_train, y_train, params)
    predictions = model.predict(X_valid)
    corr = np.corrcoef(y_valid[np.isfinite(y_valid.values)], predictions[np.isfinite(y_valid.values)])[0][1] #越大越好
    return 1 - corr #'minimize' loss

# 创建Optuna Study对象
study = optuna.create_study(direction='minimize', study_name='optim_search')
    
# 执行超参数优化
study.optimize(objective, n_trials=10)

# 输出最佳参数配置
best_params = study.best_params
best_loss = study.best_value

print("最佳参数配置：", best_params)
print("最佳IC：", 1 - best_loss)

optuna.visualization.plot_param_importances(study)