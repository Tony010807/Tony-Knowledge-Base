

import numpy as np
import pandas as pd
import os
from typing import Dict, List
from loguru import logger

data_path = "/Users/qianhonglin/Desktop/Quant full time/Digital Currency vol prediction" # 文件夹所在位置
os.chdir(data_path)

from features import Features
from utils import load_data, smoothing_Z_score

BASE_ENCODING = {'BTC':0, 'ETH':1}

class DataModule:
    def __init__(self, source, freq:str, base:str='BTC') -> None:
        self.freq = freq
        self.base = base
        self.source = source
    
    def get_data(self, starttime, endtime, columns):
        df = load_data(start_date=starttime, end_date=endtime, base=self.base.lower(), path=self.source)
        df = df.sort_values(by=['date','base'], ascending=True)
        df.set_index(['date', 'base'], drop=True, inplace=True)
        df = df[columns]
        return df
    
    def get_features(self, starttime, endtime, columns, add_target=True):
        feat = Features()
        df = self.get_data(starttime, endtime, columns)
        df['BookSkew'] = feat.BookSkew(df)
        df['DayofWeek'] = feat.DayofWeek(df)
        df['TimeFromNight'] = feat.TimeFromMidnight(df)
        df['HighOrLow_24'] = feat.HighOrLow(df, period=24)
        df['HighOrLow_3'] = feat.HighOrLow(df, period=3)
        df['HighOrLow_24_max'] = feat.HighOrLow(df, period=24, is_max=True)
        df['HighOrLow_3_max'] = feat.HighOrLow(df, period=3, is_max=True)
        df['relative_spread'] = feat.relative_spread(df)
        df['TT_24'] = feat.TT(df, period=24)
        df['TT_3'] = feat.TT(df, period=3)
        df['TTNet_24'] = feat.TTNet(df, period=24)
        df['TTNet_3'] = feat.TTNet(df, period=3)
        df['TTNetNum_24'] = feat.TTNetNum(df, period=24)
        df['TTNetNum_3'] = feat.TTNetNum(df, period=3)
        df['TTNetNumPct_24'] = feat.TTNetNumPct(df, period=24)
        df['TTNetNumPct_3'] = feat.TTNetNumPct(df, period=3)
        df['TTNum_24'] = feat.TTNum(df, period=24)
        df['TTNum_3'] = feat.TTNum(df, period=3)
        df['TTToLevelRatio'] = feat.TTToLevelRatio(df)
        df['VolumeChangeRatio'] = feat.VolumeChangeRatio(df)
        df['open_norm'] = feat.price_norm(df, column='open', windows=7)
        df['close_norm'] = feat.price_norm(df, column='close', windows=7)
        df['low_norm'] = feat.price_norm(df, column='low', windows=7)
        df['high_norm'] = feat.price_norm(df, column='high', windows=7)
        df['vwap_norm'] = feat.price_norm(df, column='vwap', windows=7)
        df['twap_norm'] = feat.price_norm(df, column='twap', windows=7)
        df['awap_norm'] = feat.price_norm(df, column='awap', windows=7)
        df['log_return'] = feat.log_return(df)
        df['rv_24before'] = feat.rv(df, window=24, scale_window=365 * 24)
        df["rv_1wbefore"] = feat.rv(df, window=24 * 7, scale_window=365 * 24)
        df["rv_1mbefore"] = feat.rv(df, window=24 * 30, scale_window=365 * 24)
        df['relative_trade_volume'] = feat.relative_trade_volume(df)
        df['relative_tradeval'] = feat.relative_tradeval(df)
        df['relative_ntrade'] = feat.relative_ntrade(df)
        df['relative_high_ratio'] = feat.relative_high_ratio(df)
        df['relative_low_ratio'] = feat.relative_low_ratio(df)
        df['corr_high_rank_volume'] = feat.corr_high_rank_volume(df)
        df['corr_std_high_vol'] = feat.corr_std_high_vol(df)
        df['price_diff_ratio'] = feat.price_diff_ratio(df)
        df['corr_high_low'] = feat.corr_high_low(df)
        df['super_high_low'] = feat.super_high_low(df)
        if add_target:
            df['rv_24'] = feat.rv_24(df, window=24, scale_window=365 * 24)
        return df
    
    
    
    
class DataPreprocessing:
    def __init__(self, datasets: Dict, freq:str) -> None:
        self.dataset_names = datasets.keys()
        self.datasets = datasets
        self.freq = freq
    
    def _outlier_check(self, data:pd.DataFrame, sz:int=5, tr:int=3):
        data = smoothing_Z_score(data, sz=sz, tr=tr)
        return data
    
    def _missing_check(self, data:pd.DataFrame, dataset_name):
        stime = lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
        begin, end = stime(data.index[0][0]), stime(data.index[-1][0])
        expected_data_row_number = len(list(pd.date_range(begin, end, freq = self.freq)))
        missing_rows = expected_data_row_number - data.shape[0]
        if missing_rows > 0:
            logger.warning("Missing %d rows for %s" % (missing_rows, dataset_name))
    
    def _data_filter(self, dataset_name):
        data = self.datasets[dataset_name].copy()
        data = self._outlier_check(data)
        self._missing_check(data, dataset_name)
        data = data.reset_index(level='base')
        data['base'] = data['base'].apply(lambda x: BASE_ENCODING[x.upper()])
        data = data.asfreq(self.freq)
        return data
        
    def data_collation(self, drop_columns=None):
        DatasetCollections = []
        for db_name in self.dataset_names:
            dbb_data = self._data_filter(db_name)
            dbb_data = dbb_data.reset_index(drop=False)
            DatasetCollections.append(dbb_data)
        data = pd.concat(DatasetCollections)
        if isinstance(drop_columns, list):
            for cl in drop_columns:
                if cl in data.columns:
                    data = data.drop(cl, axis=1)
        return data.sort_values(['date', 'base']).set_index('date')
    