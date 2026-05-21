import pandas as pd
import numpy as np

class Features:
    @staticmethod
    def price_norm(data:pd.DataFrame, column:str, windows) -> np.array:
        # 用于归一化价格量纲，表示相对价格大小
        assert np.isin([column], data.columns).all()
        _min_periods = (windows-1)*24
        return data[column]/data[column].rolling(windows*24, min_periods=_min_periods).mean()

    @staticmethod
    def log_return(data:pd.DataFrame) -> np.array:
        # 计算对数收益率
        assert np.isin(['close'], data.columns).all()
        _lr = np.log(data['close']).diff()
        return _lr.values

    @staticmethod
    def rv(data:pd.DataFrame, window:float, scale_window:float, shift:int=0) -> np.array:
        # 计算波动率
        assert np.isin(['close'], data.columns).all()
        _min_periods = int(0.8 * window) # rolling winsows中包含的最小的元素数量
        _rv = np.sqrt(np.square(np.log(data['close']).diff()).rolling(window = int(window), min_periods = _min_periods).mean() * int(scale_window)) # scale_window 365*24 处理到年化水平
        _rv =  _rv.shift(shift) # shift 预测的长度
        return _rv.values

    @staticmethod
    def rv_24(data:pd.DataFrame, window:float, scale_window:float) -> np.array:
        assert np.isin(['close'], data.columns).all()
        _rv_24 = Features.rv(data, window, scale_window, shift=-24)
        return _rv_24

    @staticmethod
    def BookSkew(data: pd.DataFrame) -> np.array:
        """
        计算盘口偏度（skew）指标，该指标用于分析市场的买卖压力分布。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_asksize' 和 'close_bidsize' 列。

        返回:
        np.array: 包含计算的盘口偏度值的数组。

        逻辑:
        1. 首先，使用断言确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列。
        2. 然后，计算盘口偏度（skew）指标，它用于描述卖单和买单的相对强度。
        3. 最后，返回包含盘口偏度值的数组。

        相关信息:
        - 盘口偏度是金融分析中的一个重要指标，用于衡量市场上买卖双方的力量对比。

        示例用法:
        data = pd.DataFrame(...)
        skew_values = YourClass.BookSkew(data)
        """
        assert np.isin(['close_asksize', 'close_bidsize'], data.columns).all()
        
        # 计算盘口偏度（skew）
        _skew = data.close_bidsize / (data.close_asksize + data.close_bidsize)
        
        return _skew.values



    @staticmethod
    def DayofWeek(data: pd.DataFrame):
        """
        计算DataFrame的日期索引对应的每个日期是一周的哪一天（星期几）。

        参数:
        data (pd.DataFrame): 包含日期索引的DataFrame。

        返回:
        np.array: 包含每个日期的星期几的数组。

        逻辑:
        1. 首先，将DataFrame的索引转换为日期时间格式，以便处理日期信息。
        2. 然后，使用Pandas的`weekday`属性获取每个日期的星期几（0表示星期一，1表示星期二，...，6表示星期日）。
        3. 最后，返回包含星期几信息的数组。

        示例用法:
        data = pd.DataFrame(...)
        dow = YourClass.DayofWeek(data)
        """
        index_ = pd.to_datetime(data.index.get_level_values(0))
        _dow = index_.weekday
        return _dow.values


    @staticmethod
    def TimeFromMidnight(data: pd.DataFrame):
        """
        计算DataFrame日期索引中每个时间点距午夜的时间（以小数天为单位）。

        参数:
        data (pd.DataFrame): 包含日期索引的DataFrame。

        返回:
        np.array: 包含每个时间点距午夜的时间的数组（以小数天为单位）。

        逻辑:
        1. 首先，将DataFrame的索引转换为日期时间格式，以便处理日期和时间信息。
        2. 然后，计算每个时间点距午夜的时间，将结果转换为小数天。
        3. 最后，返回包含时间信息的数组。

        示例用法:
        data = pd.DataFrame(...)
        time_since_midnight = YourClass.TimeFromMidnight(data)
        """
        index_ = pd.to_datetime(data.index.get_level_values(0))
        
        # 计算每个时间点距午夜的时间
        _time_since = (index_ - index_.normalize()).seconds / 86400
        
        return _time_since.values

    
    @staticmethod
    def HighOrLow(data: pd.DataFrame, period: int, is_max: bool = False):
        """
        计算DataFrame中市场价格相对于一定时间内最高价或最低价的涨幅或跌幅。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_ask' 和 'close_bid' 列。
        period (int): 用于计算最高价或最低价的时间窗口大小。
        is_max (bool, 可选): 如果为True，则计算相对于最高价的涨幅；如果为False（默认），则计算相对于最低价的跌幅。

        返回:
        np.array: 包含相对于最高价或最低价的涨幅或跌幅的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'close_ask' 和 'close_bid' 两列。
        2. 计算市场价格的中间价（(close_ask + close_bid) / 2）。
        3. 根据 'is_max' 参数，选择计算最高价或最低价的滚动窗口值。
        4. 计算相对于最高价或最低价的涨幅或跌幅。
        5. 返回包含涨幅或跌幅信息的数组。

        示例用法:
        data = pd.DataFrame(...)
        # 计算相对于最高价的涨幅
        high_rise = YourClass.HighOrLow(data, period=10, is_max=True)
        # 计算相对于最低价的跌幅
        low_fall = YourClass.HighOrLow(data, period=10, is_max=False)
        """
        assert np.isin(['close_ask', 'close_bid'], data.columns).all()
        
        # 计算市场价格的中间价
        _mp = (data.close_ask + data.close_bid) / 2
        
        # 根据 'is_max' 参数，选择计算最高价或最低价的滚动窗口值
        if is_max:
            _val = _mp.rolling(period).max()
        else:
            _val = _mp.rolling(period).min() 
        
        # 计算相对于最高价或最低价的涨幅或跌幅
        result = ((_val - _mp) / _mp).values
        
        return result


    @staticmethod
    def TT(data: pd.DataFrame, period: int):
        """
        计算DataFrame中市场订单总量的滚动加和。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_asksize' 和 'close_bidsize' 列。
        period (int): 用于计算滚动加和的时间窗口大小。

        返回:
        np.array: 包含滚动加和后的市场订单总量的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列。
        2. 计算市场订单总量，即 'close_bidsize' 和 'close_asksize' 的总和。
        3. 使用滚动窗口计算总订单量在给定时间窗口内的累积和。
        4. 返回包含累积和的数组。

        示例用法:
        data = pd.DataFrame(...)
        total_order_size = YourClass.TT(data, period=10)
        """
        
        # 确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列
        assert np.isin(['close_asksize', 'close_bidsize'], data.columns).all()
        
        # 计算市场订单总量
        _size = data.close_bidsize + data.close_asksize
        
        # 使用滚动窗口计算总订单量在给定时间窗口内的累积和
        result = _size.rolling(period).sum().values
    
        return result


    @staticmethod
    def TTNet(data: pd.DataFrame, period: int):
        """
        计算DataFrame中市场订单净量的滚动加和。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_asksize' 和 'close_bidsize' 列。
        period (int): 用于计算滚动加和的时间窗口大小。

        返回:
        np.array: 包含滚动加和后的市场订单净量的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列。
        2. 计算市场订单净量，即 'close_bidsize' 减去 'close_asksize'。
        3. 使用滚动窗口计算订单净量在给定时间窗口内的累积和。
        4. 返回包含累积和的数组。

        示例用法:
        data = pd.DataFrame(...)
        net_order_size = YourClass.TTNet(data, period=10)
        """
        
        # 确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列
        assert np.isin(['close_asksize', 'close_bidsize'], data.columns).all()
        
        # 计算市场订单净量
        _res = data.close_bidsize - data.close_asksize
        
        # 使用滚动窗口计算订单净量在给定时间窗口内的累积和
        result = _res.rolling(period).sum().values
        
        return result


    @staticmethod
    def TTNetNum(data: pd.DataFrame, period: int):
        """
        计算DataFrame中市场订单净量的滚动加和，每个时间点的正负号代表买入（+1）或卖出（-1）。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_asksize' 和 'close_bidsize' 列。
        period (int): 用于计算滚动加和的时间窗口大小。

        返回:
        np.array: 包含滚动加和后的市场订单净量的数组，每个时间点的正负号代表买入（+1）或卖出（-1）。

        逻辑:
        1. 首先，确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列。
        2. 计算市场订单净量，即 'close_bidsize' 减去 'close_asksize'。
        3. 使用Lambda函数将每个净量的正负号表示为+1（买入）或-1（卖出）。
        4. 使用滚动窗口计算订单净量在给定时间窗口内的累积和，同时保留正负号信息。
        5. 返回包含累积和的数组，每个时间点的正负号代表买入（+1）或卖出（-1）。

        示例用法:
        data = pd.DataFrame(...)
        net_order_size = YourClass.TTNetNum(data, period=10)
        """
    
        # 确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列
        assert np.isin(['close_asksize', 'close_bidsize'], data.columns).all()
        
        # 计算市场订单净量
        _res = (data.close_bidsize - data.close_asksize).apply(lambda x: np.sign(x))
        
        # 使用滚动窗口计算订单净量在给定时间窗口内的累积和，同时保留正负号信息
        result = _res.rolling(period).sum().values
        
        return result


    @staticmethod
    def TTNetNumPct(data: pd.DataFrame, period: int):
        """
        计算DataFrame中市场订单净量的滚动加和，每个时间点的正负号代表买入（+1）或卖出（-1），并将正负号表示为百分比形式。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_asksize' 和 'close_bidsize' 列。
        period (int): 用于计算滚动加和的时间窗口大小。

        返回:
        np.array: 包含滚动加和后的市场订单净量的数组，每个时间点的正负号代表买入（1.0）或卖出（0.0）。

        逻辑:
        1. 首先，确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列。
        2. 计算市场订单净量，即 'close_bidsize' 减去 'close_asksize'。
        3. 使用Lambda函数将每个净量的正负号表示为0.0（卖出）或1.0（买入）。
        4. 使用滚动窗口计算订单净量在给定时间窗口内的累积和，并将正负号表示为0.0（卖出）或1.0（买入）。
        5. 返回包含累积和的数组，每个时间点的正负号代表买入（1.0）或卖出（0.0）。

        示例用法:
        data = pd.DataFrame(...)
        net_order_size = YourClass.TTNetNumPct(data, period=10)
        """
        
        # 确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列
        assert np.isin(['close_asksize', 'close_bidsize'], data.columns).all()
        
        # 计算市场订单净量
        _res = (data.close_bidsize - data.close_asksize).apply(lambda x: 0.5 * np.sign(x) + 0.5)
        
        # 使用滚动窗口计算订单净量在给定时间窗口内的累积和，并将正负号表示为0.0（卖出）或1.0（买入）
        result = _res.rolling(period).sum().values
        
        return result


    @staticmethod
    def TTNum(data: pd.DataFrame, period: int):
        """
        计算DataFrame中市场订单净量的滚动加和，每个时间点的正负号代表买入（+1）或卖出（-1），并将正负号表示为绝对值。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_asksize' 和 'close_bidsize' 列。
        period (int): 用于计算滚动加和的时间窗口大小。

        返回:
        np.array: 包含滚动加和后的市场订单净量的数组，每个时间点的正负号都为1。

        逻辑:
        1. 首先，确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列。
        2. 计算市场订单净量，即 'close_bidsize' 减去 'close_asksize'。
        3. 使用Lambda函数将每个净量的正负号表示为1，表示无论是买入还是卖出，都将其视为正数。
        4. 使用滚动窗口计算订单净量在给定时间窗口内的累积和，并将正负号保持为1。
        5. 返回包含累积和的数组，每个时间点的正负号都为1。

        示例用法:
        data = pd.DataFrame(...)
        net_order_size = YourClass.TTNum(data, period=10)
        """
    
        # 确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列
        assert np.isin(['close_asksize', 'close_bidsize'], data.columns).all()
        
        # 计算市场订单净量
        _res = (data.close_bidsize - data.close_asksize).apply(lambda x: np.abs(np.sign(x)))
        
        # 使用滚动窗口计算订单净量在给定时间窗口内的累积和，并将正负号保持为1
        result = _res.rolling(period).sum().values
        
        return result


    @staticmethod
    def TTToLevelRatio(data: pd.DataFrame):
        """
        计算DataFrame中市场订单净量（买入净量 - 卖出净量）在每个时间点相对于总净量的比率。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_asksize' 和 'close_bidsize' 列。

        返回:
        np.array: 包含市场订单净量在每个时间点相对于总净量的比率的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列。
        2. 计算市场的买入净量和卖出净量，使用累积的方式计算，正数表示买入，负数表示卖出。
        3. 使用这两个累积值计算市场订单净量在每个时间点相对于总净量的比率。
        4. 返回包含比率信息的数组。

        示例用法:
        data = pd.DataFrame(...)
        order_ratio = YourClass.TTToLevelRatio(data)
        """
        
        # 确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列
        assert np.isin(['close_asksize', 'close_bidsize'], data.columns).all()
        
        # 计算市场的买入净量和卖出净量，使用累积的方式计算，正数表示买入，负数表示卖出
        _bids = np.cumsum(np.signbit(data.close_bidsize - data.close_asksize) * (data.close_bidsize - data.close_asksize))
        _asks = np.cumsum(np.signbit(data.close_asksize - data.close_bidsize) * (data.close_asksize - data.close_bidsize))
        
        # 使用这两个累积值计算市场订单净量在每个时间点相对于总净量的比率
        result = (_bids / (_asks + _bids)).values
        
        return result


    @staticmethod
    def VolumeChangeRatio(data: pd.DataFrame):
        """
        计算DataFrame中市场订单的成交量变化比率。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_asksize' 和 'close_bidsize' 列。

        返回:
        np.array: 包含市场订单的成交量变化比率的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列。
        2. 根据 'close_bidsize' 是否大于 'close_asksize'，将数据分为买单和卖单。
        3. 计算买单和卖单的成交量变化，忽略微小的变化（小于1e-5）。
        4. 计算每个时间点的成交量变化比率，根据买单和卖单的不同来选择对应的成交量变化。
        5. 返回包含成交量变化比率的数组。

        示例用法:
        data = pd.DataFrame(...)
        volume_ratio = YourClass.VolumeChangeRatio(data)
        """
    
        # 确保 'data' 包含 'close_asksize' 和 'close_bidsize' 两列
        assert np.isin(['close_asksize', 'close_bidsize'], data.columns).all()
        
        # 根据 'close_bidsize' 是否大于 'close_asksize'，将数据分为买单和卖单
        is_bid = data.close_bidsize > data.close_asksize
        is_ask = data.close_bidsize <= data.close_asksize
        
        # 计算买单和卖单的成交量变化，忽略微小的变化（小于1e-5）
        _bidsize = ((data.close_bidsize.diff() > 1e-5) * data.close_bidsize +
                    (data.close_bidsize.diff() < -1e-5) * data.close_bidsize.shift(1) +
                    (data.close_bidsize.diff().abs() <= 1e-5) * data.close_bidsize.diff().abs())

        _asksize = ((data.close_asksize.diff() > 1e-5) * data.close_asksize +
                    (data.close_asksize.diff() < -1e-5) * data.close_asksize.shift(1) +
                    (data.close_asksize.diff().abs() <= 1e-5) * data.close_asksize.diff().abs())
        
        # 计算每个时间点的成交量变化比率，根据买单和卖单的不同来选择对应的成交量变化
        result = (is_bid * _bidsize + is_ask * _asksize).values
        
        return result

    
    @staticmethod
    def vwap_balance(data: pd.DataFrame, is_abs=True):
        """
        计算DataFrame中的VWAP（成交量加权平均价）买单和卖单之间的差值。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'vwap_buy' 和 'vwap_sell' 列。
        is_abs (bool, 可选): 控制是否返回差值的绝对值。默认为 True，返回绝对值。

        返回:
        np.array: 包含VWAP买单和卖单之间的差值或绝对差值的数组，根据 'is_abs' 参数而定。

        逻辑:
        1. 首先，确保 'data' 包含 'vwap_buy' 和 'vwap_sell' 两列。
        2. 计算VWAP买单和卖单之间的差值。
        3. 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值。
        4. 返回包含差值或绝对差值的数组。

        示例用法:
        data = pd.DataFrame(...)
        vwap_diff = YourClass.vwap_balance(data)
        或
        vwap_abs_diff = YourClass.vwap_balance(data, is_abs=True)
        """
        
        # 确保 'data' 包含 'vwap_buy' 和 'vwap_sell' 两列
        assert np.isin(['vwap_buy', 'vwap_sell'], data.columns).all()
        
        # 计算VWAP买单和卖单之间的差值
        _res = data['vwap_buy'] - data['vwap_sell']
        
        # 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值
        result = abs(_res) if is_abs else _res
        
        return result

    
    @staticmethod
    def volume_imbalance(data: pd.DataFrame, is_abs=True):
        """
        计算DataFrame中的买单和卖单成交量之间的差值。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'volume_buy' 和 'volume_sell' 列。
        is_abs (bool, 可选): 控制是否返回差值的绝对值。默认为 True，返回绝对值。

        返回:
        np.array: 包含买单和卖单成交量之间的差值或绝对差值的数组，根据 'is_abs' 参数而定。

        逻辑:
        1. 首先，确保 'data' 包含 'volume_buy' 和 'volume_sell' 两列。
        2. 计算买单成交量和卖单成交量之间的差值。
        3. 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值。
        4. 返回包含差值或绝对差值的数组。

        示例用法:
        data = pd.DataFrame(...)
        volume_diff = YourClass.volume_imbalance(data)
        或
        volume_abs_diff = YourClass.volume_imbalance(data, is_abs=True)
        """
    
        # 确保 'data' 包含 'volume_buy' 和 'volume_sell' 两列
        assert np.isin(['volume_buy', 'volume_sell'], data.columns).all()
        
        # 计算买单成交量和卖单成交量之间的差值
        _res = data['volume_buy'] - data['volume_sell']
        
        # 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值
        result = abs(_res) if is_abs else _res
        
        return result


    @staticmethod
    def tradeval_imbalance(data: pd.DataFrame, is_abs=True):
        """
        计算DataFrame中的买单和卖单成交额之间的差值。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'tradeval_buy' 和 'tradeval_sell' 列。
        is_abs (bool, 可选): 控制是否返回差值的绝对值。默认为 True，返回绝对值。

        返回:
        np.array: 包含买单和卖单成交额之间的差值或绝对差值的数组，根据 'is_abs' 参数而定。

        逻辑:
        1. 首先，确保 'data' 包含 'tradeval_buy' 和 'tradeval_sell' 两列。
        2. 计算买单成交成交额和卖单成交额值之间的差值。
        3. 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值。
        4. 返回包含差值或绝对差值的数组。

        示例用法:
        data = pd.DataFrame(...)
        tradeval_diff = YourClass.tradeval_imbalance(data)
        或
        tradeval_abs_diff = YourClass.tradeval_imbalance(data, is_abs=True)
        """
    
        # 确保 'data' 包含 'tradeval_buy' 和 'tradeval_sell' 两列
        assert np.isin(['tradeval_buy', 'tradeval_sell'], data.columns).all()
        
        # 计算买单成交额和卖单成交额之间的差值
        _res = data['tradeval_buy'] - data['tradeval_sell']
        
        # 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值
        result = abs(_res) if is_abs else _res
        
        return result


    @staticmethod
    def ntrade_imbalance(data: pd.DataFrame, is_abs=True):
        """
        计算DataFrame中的买单和卖单成交笔数之间的差值。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'ntrade_buy' 和 'ntrade_sell' 列。
        is_abs (bool, 可选): 控制是否返回差值的绝对值。默认为 True，返回绝对值。

        返回:
        np.array: 包含买单和卖单成交笔数之间的差值或绝对差值的数组，根据 'is_abs' 参数而定。

        逻辑:
        1. 首先，确保 'data' 包含 'ntrade_buy' 和 'ntrade_sell' 两列。
        2. 计算买单成交笔数和卖单成交笔数之间的差值。
        3. 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值。
        4. 返回包含差值或绝对差值的数组。

        示例用法:
        data = pd.DataFrame(...)
        ntrade_diff = YourClass.ntrade_imbalance(data)
        或
        ntrade_abs_diff = YourClass.ntrade_imbalance(data, is_abs=True)
        """
    
        # 确保 'data' 包含 'ntrade_buy' 和 'ntrade_sell' 两列
        assert np.isin(['ntrade_buy', 'ntrade_sell'], data.columns).all()
        
        # 计算买单成交笔数和卖单成交笔数之间的差值
        _res = data['ntrade_buy'] - data['ntrade_sell']
        
        # 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值
        result = abs(_res) if is_abs else _res
        
        return result


    @staticmethod
    def ntrade_up_imbalance(data: pd.DataFrame, is_abs=True):
        """
        计算DataFrame中的买单和卖单成交上涨笔数之间的差值。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'ntrade_up_buy' 和 'ntrade_up_sell' 列。
        is_abs (bool, 可选): 控制是否返回差值的绝对值。默认为 True，返回绝对值。

        返回:
        np.array: 包含买单和卖单成交上涨笔数之间的差值或绝对差值的数组，根据 'is_abs' 参数而定。

        逻辑:
        1. 首先，确保 'data' 包含 'ntrade_up_buy' 和 'ntrade_up_sell' 两列。
        2. 计算买单成交上涨笔数和卖单成交上涨笔数之间的差值。
        3. 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值。
        4. 返回包含差值或绝对差值的数组。

        示例用法:
        data = pd.DataFrame(...)
        ntrade_up_diff = YourClass.ntrade_up_imbalance(data)
        或
        ntrade_up_abs_diff = YourClass.ntrade_up_imbalance(data, is_abs=True)
        """
        
        # 确保 'data' 包含 'ntrade_up_buy' 和 'ntrade_up_sell' 两列
        assert np.isin(['ntrade_up_buy', 'ntrade_up_sell'], data.columns).all()
        
        # 计算买单成交上涨笔数和卖单成交上涨笔数之间的差值
        _res = data['ntrade_up_buy'] - data['ntrade_up_sell']
        
        # 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值
        result = abs(_res) if is_abs else _res
        
        return result


    @staticmethod
    def ntrade_down_imbalance(data: pd.DataFrame, is_abs=True):
        """
        计算DataFrame中的买单和卖单成交下跌笔数之间的差值。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'ntrade_down_buy' 和 'ntrade_down_sell' 列。
        is_abs (bool, 可选): 控制是否返回差值的绝对值。默认为 True，返回绝对值。

        返回:
        np.array: 包含买单和卖单成交下跌笔数之间的差值或绝对差值的数组，根据 'is_abs' 参数而定。

        逻辑:
        1. 首先，确保 'data' 包含 'ntrade_down_buy' 和 'ntrade_down_sell' 两列。
        2. 计算买单成交下跌笔数和卖单成交下跌笔数之间的差值。
        3. 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值。
        4. 返回包含差值或绝对差值的数组。

        示例用法:
        data = pd.DataFrame(...)
        ntrade_down_diff = YourClass.ntrade_down_imbalance(data)
        或
        ntrade_down_abs_diff = YourClass.ntrade_down_imbalance(data, is_abs=True)
        """
    
        # 确保 'data' 包含 'ntrade_down_buy' 和 'ntrade_down_sell' 两列
        assert np.isin(['ntrade_down_buy', 'ntrade_down_sell'], data.columns).all()
        
        # 计算买单成交下跌笔数和卖单成交下跌笔数之间的差值
        _res = data['ntrade_down_buy'] - data['ntrade_down_sell']
        
        # 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值
        result = abs(_res) if is_abs else _res
        
        return result


    @staticmethod
    def ntrade_flat_imbalance(data: pd.DataFrame, is_abs=True):
        """
        计算DataFrame中的买单和卖单成交平盘笔数之间的差值。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'ntrade_flat_buy' 和 'ntrade_flat_sell' 列。
        is_abs (bool, 可选): 控制是否返回差值的绝对值。默认为 True，返回绝对值。

        返回:
        np.array: 包含买单和卖单成交平盘笔数之间的差值或绝对差值的数组，根据 'is_abs' 参数而定。

        逻辑:
        1. 首先，确保 'data' 包含 'ntrade_flat_buy' 和 'ntrade_flat_sell' 两列。
        2. 计算买单成交平盘笔数和卖单成交平盘笔数之间的差值。
        3. 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值。
        4. 返回包含差值或绝对差值的数组。

        示例用法:
        data = pd.DataFrame(...)
        ntrade_flat_diff = YourClass.ntrade_flat_imbalance(data)
        或
        ntrade_flat_abs_diff = YourClass.ntrade_flat_imbalance(data, is_abs=True)
        """
    
        # 确保 'data' 包含 'ntrade_flat_buy' 和 'ntrade_flat_sell' 两列
        assert np.isin(['ntrade_flat_buy', 'ntrade_flat_sell'], data.columns).all()
        
        # 计算买单成交平盘笔数和卖单成交平盘笔数之间的差值
        _res = data['ntrade_flat_buy'] - data['ntrade_flat_sell']
        
        # 如果 'is_abs' 参数为 True，则返回差值的绝对值，否则返回原始差值
        result = abs(_res) if is_abs else _res
        
        return result

    
 
    @staticmethod
    def relative_trade_volume(data: pd.DataFrame, sw: int = 24, lw: int = 720):
        """
        计算DataFrame中的相对交易量。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'volume' 列。
        sw (int, 可选): 短期窗口大小，默认为 24。
        lw (int, 可选): 长期窗口大小，默认为 720。

        返回:
        np.array: 包含相对交易量的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'volume' 列。
        2. 使用滚动窗口计算短期窗口内的平均交易量和长期窗口内的平均交易量。
        3. 计算短期平均交易量与长期平均交易量的比值，得到相对交易量。
        4. 返回包含相对交易量的数组。

        示例用法:
        data = pd.DataFrame(...)
        relative_volume = YourClass.relative_trade_volume(data, sw=24, lw=720)
        """
    
        # 确保 'data' 包含 'volume' 列
        assert np.isin(['volume'], data.columns)
        
        # 使用滚动窗口计算短期窗口内的平均交易量和长期窗口内的平均交易量
        short_term_avg = data['volume'].rolling(window=sw, min_periods=int(0.8 * sw)).mean()
        long_term_avg = data['volume'].rolling(window=lw, min_periods=int(0.8 * lw)).mean()
        
        # 计算短期平均交易量与长期平均交易量的比值，得到相对交易量
        _res = short_term_avg / long_term_avg
        
        return _res

    
    @staticmethod
    def relative_tradeval(data: pd.DataFrame, sw: int = 24, lw: int = 720):
        """
        计算DataFrame中的相对交易价值。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'tradeval' 列。
        sw (int, 可选): 短期窗口大小，默认为 24。
        lw (int, 可选): 长期窗口大小，默认为 720。

        返回:
        np.array: 包含相对交易价值的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'tradeval' 列。
        2. 使用滚动窗口计算短期窗口内的平均交易价值和长期窗口内的平均交易价值。
        3. 计算短期平均交易价值与长期平均交易价值的比值，得到相对交易价值。
        4. 返回包含相对交易价值的数组。

        示例用法:
        data = pd.DataFrame(...)
        relative_tradeval = YourClass.relative_tradeval(data, sw=24, lw=720)
        """
    
        # 确保 'data' 包含 'tradeval' 列
        assert np.isin(['tradeval'], data.columns)
        
        # 使用滚动窗口计算短期窗口内的平均交易价值和长期窗口内的平均交易价值
        short_term_avg = data['tradeval'].rolling(window=sw, min_periods=int(0.8 * sw)).mean()
        long_term_avg = data['tradeval'].rolling(window=lw, min_periods=int(0.8 * lw)).mean()
        
        # 计算短期平均交易价值与长期平均交易价值的比值，得到相对交易价值
        _res = short_term_avg / long_term_avg
        
        return _res


    @staticmethod
    def relative_ntrade(data: pd.DataFrame, sw: int = 24, lw: int = 720):
        """
        计算DataFrame中的相对成交笔数。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'ntrade' 列。
        sw (int, 可选): 短期窗口大小，默认为 24。
        lw (int, 可选): 长期窗口大小，默认为 720。

        返回:
        np.array: 包含相对成交笔数的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'ntrade' 列。
        2. 使用滚动窗口计算短期窗口内的平均成交笔数和长期窗口内的平均成交笔数。
        3. 计算短期平均成交笔数与长期平均成交笔数的比值，得到相对成交笔数。
        4. 返回包含相对成交笔数的数组。

        示例用法:
        data = pd.DataFrame(...)
        relative_ntrade = YourClass.relative_ntrade(data, sw=24, lw=720)
        """
    
        # 确保 'data' 包含 'ntrade' 列
        assert np.isin(['ntrade'], data.columns)
        
        # 使用滚动窗口计算短期窗口内的平均成交笔数和长期窗口内的平均成交笔数
        short_term_avg = data['ntrade'].rolling(window=sw, min_periods=int(0.8 * sw)).mean()
        long_term_avg = data['ntrade'].rolling(window=lw, min_periods=int(0.8 * lw)).mean()
        
        # 计算短期平均成交笔数与长期平均成交笔数的比值，得到相对成交笔数
        _res = short_term_avg / long_term_avg
        
        return _res

    
    @staticmethod
    def relative_high_ratio(data: pd.DataFrame, sw: int = 3, lw: int = 72):
        """
        计算DataFrame中的相对最高价比率。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'high' 列。
        sw (int, 可选): 短期窗口大小，默认为 3。
        lw (int, 可选): 长期窗口大小，默认为 72。

        返回:
        np.array: 包含相对最高价比率的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'high' 列。
        2. 使用滚动窗口计算短期窗口内的最高价和长期窗口内的最高价。
        3. 计算短期最高价与长期最高价的比值，得到相对最高价比率。
        4. 返回包含相对最高价比率的数组。

        示例用法:
        data = pd.DataFrame(...)
        relative_high = YourClass.relative_high_ratio(data, sw=3, lw=72)
        """
    
        # 确保 'data' 包含 'high' 列
        assert np.isin(['high'], data.columns)
        
        # 使用滚动窗口计算短期窗口内的最高价和长期窗口内的最高价
        short_term_max = data['high'].rolling(window=sw, min_periods=int(0.8 * sw)).max()
        long_term_max = data['high'].rolling(window=lw, min_periods=int(0.8 * lw)).max()
        
        # 计算短期最高价与长期最高价的比值，得到相对最高价比率
        _res = short_term_max / long_term_max
        
        return _res


    @staticmethod
    def relative_low_ratio(data: pd.DataFrame, sw: int = 3, lw: int = 72):
        """
        计算DataFrame中的相对最低价比率。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'low' 列。
        sw (int, 可选): 短期窗口大小，默认为 3。
        lw (int, 可选): 长期窗口大小，默认为 72。

        返回:
        np.array: 包含相对最低价比率的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'low' 列。
        2. 使用滚动窗口计算短期窗口内的最低价和长期窗口内的最低价。
        3. 计算短期最低价与长期最低价的比值，得到相对最低价比率。
        4. 返回包含相对最低价比率的数组。

        示例用法:
        data = pd.DataFrame(...)
        relative_low = YourClass.relative_low_ratio(data, sw=3, lw=72)
        """
    
        # 确保 'data' 包含 'low' 列
        assert np.isin(['low'], data.columns)
        
        # 使用滚动窗口计算短期窗口内的最低价和长期窗口内的最低价
        short_term_min = data['low'].rolling(window=sw, min_periods=int(0.8 * sw)).min()
        long_term_min = data['low'].rolling(window=lw, min_periods=int(0.8 * lw)).min()
        
        # 计算短期最低价与长期最低价的比值，得到相对最低价比率
        _res = short_term_min / long_term_min
        
        return _res


    @staticmethod
    def relative_spread(data: pd.DataFrame, sw: int = 72, lw: int = 720):
        """
        计算DataFrame中的相对价差比率。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'close_ask' 和 'close_bid' 列。
        sw (int, 可选): 短期窗口大小，默认为 72。
        lw (int, 可选): 长期窗口大小，默认为 720。

        返回:
        np.array: 包含相对价差比率的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'close_ask' 和 'close_bid' 列。
        2. 计算价差 (spread) = 'close_ask' - 'close_bid'。
        3. 使用滚动窗口计算短期窗口内的价差平均值和长期窗口内的价差平均值。
        4. 计算短期价差平均值与长期价差平均值的比值，得到相对价差比率。
        5. 返回包含相对价差比率的数组。

        示例用法:
        data = pd.DataFrame(...)
        relative_spread = YourClass.relative_spread(data, sw=72, lw=720)
        """
    
        # 确保 'data' 包含 'close_ask' 和 'close_bid' 列
        assert np.isin(['close_ask', 'close_bid'], data.columns).all()
        
        # 计算价差 (spread) = 'close_ask' - 'close_bid'
        data['spread'] = data['close_ask'] - data['close_bid']
        
        # 使用滚动窗口计算短期窗口内的价差平均值和长期窗口内的价差平均值
        short_term_avg = data['spread'].rolling(window=sw, min_periods=int(0.8 * sw)).mean()
        long_term_avg = data['spread'].rolling(window=lw, min_periods=int(0.8 * lw)).mean()
        
        # 计算短期价差平均值与长期价差平均值的比值，得到相对价差比率
        _res = short_term_avg / long_term_avg
        
        return _res

  
    @staticmethod
    def corr_high_low(data: pd.DataFrame, w: int = 120):
        """
        计算DataFrame中最高价和最低价的滚动相关系数。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'high' 和 'low' 列。
        w (int, 可选): 滚动窗口大小，默认为 120。

        返回:
        np.array: 包含最高价和最低价的滚动相关系数的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'high' 和 'low' 列。
        2. 使用滚动窗口计算最高价和最低价之间的滚动相关系数。
        3. 返回包含滚动相关系数的数组。

        示例用法:
        data = pd.DataFrame(...)
        corr_high_low = YourClass.corr_high_low(data, w=120)
        """
        
        # 确保 'data' 包含 'high' 和 'low' 列
        assert np.isin(['high', 'low'], data.columns).all()
        
        # 使用滚动窗口计算最高价和最低价之间的滚动相关系数
        _res = data['high'].rolling(window=w, min_periods=int(0.8 * w)).corr(data['low'])
        
        return _res


    @staticmethod
    def super_high_low(data: pd.DataFrame, w: int = 168):
        """
        计算DataFrame中的超级最高价与最低价比率。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'high' 和 'low' 列。
        w (int, 可选): 滚动窗口大小，默认为 168。

        返回:
        np.array: 包含最高价与最低价比率的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'high' 和 'low' 列。
        2. 使用滚动窗口计算最高价和最低价的滚动最小值和滚动最大值。
        3. 计算滚动最小值与滚动最大值的比率，得到超级最高价与最低价比率。
        4. 返回包含超级最高价与最低价比率的数组。

        示例用法:
        data = pd.DataFrame(...)
        super_high_low = YourClass.super_high_low(data, w=168)
        """
        
        # 确保 'data' 包含 'high' 和 'low' 列
        assert np.isin(['high', 'low'], data.columns).all()
        
        # 使用滚动窗口计算最高价和最低价的滚动最小值和滚动最大值
        rolling_min_high = data['high'].rolling(window=w, min_periods=int(0.8 * w)).min()
        rolling_max_low = data['low'].rolling(window=w, min_periods=int(0.8 * w)).max()
    
        # 计算滚动最小值与滚动最大值的比率，得到最高价与最低价比率
        _res = rolling_min_high / rolling_max_low
        
        return _res


    @staticmethod
    def corr_high_rank_volume(data: pd.DataFrame, w: int = 168):
        """
        计算DataFrame中最高价和成交量的滚动相关系数。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'high' 和 'volume' 列。
        w (int, 可选): 滚动窗口大小，默认为 168。

        返回:
        np.array: 包含最高价和成交量的滚动相关系数的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'high' 和 'volume' 列。
        2. 对成交量列进行排名操作，得到排名后的成交量。
        3. 使用滚动窗口计算最高价和排名后的成交量之间的滚动相关系数。
        4. 返回包含滚动相关系数的数组。

        示例用法:
        data = pd DataFrame(...)
        corr_high_rank_volume = YourClass.corr_high_rank_volume(data, w=168)
        """
    
        # 确保 'data' 包含 'high' 和 'volume' 列
        assert np.isin(['high', 'volume'], data.columns).all()
        
        # 对成交量列进行排名操作，得到排名后的成交量
        rank_vol = data['volume'].rank()
        
        # 使用滚动窗口计算最高价和排名后的成交量之间的滚动相关系数
        _res = data['high'].rolling(window=w, min_periods=int(0.8 * w)).corr(rank_vol)
        
        return _res



    @staticmethod
    def price_diff_ratio(data: pd.DataFrame, sw: int = 24, lw: int = 720):
        """
        计算DataFrame中的价格差异比率。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'vwap' 列。
        sw (int, 可选): 短期窗口大小，默认为 24。
        lw (int, 可选): 长期窗口大小，默认为 720。

        返回:
        np.array: 包含价格差异比率的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'vwap' 列。
        2. 计算 VWAP（成交量加权平均价格）差异的绝对值，即每个时刻的 VWAP 与前一个时刻的 VWAP 差值的绝对值。
        3. 使用滚动窗口计算短期窗口内的 VWAP 差异平均值和长期窗口内的 VWAP 差异平均值。
        4. 计算短期 VWAP 差异平均值与长期 VWAP 差异平均值的比值，得到价格差异比率。
        5. 返回包含价格差异比率的数组。

        示例用法:
        data = pd.DataFrame(...)
        price_diff_ratio = YourClass.price_diff_ratio(data, sw=24, lw=720)
        """
    
        # 确保 'data' 包含 'vwap' 列
        assert np.isin(['vwap'], data.columns).all()
        
        # 计算 VWAP 差异的绝对值
        vwap_diff = data['vwap'].diff().abs()
        
        # 使用滚动窗口计算短期窗口内的 VWAP 差异平均值和长期窗口内的 VWAP 差异平均值
        short_term_avg = vwap_diff.rolling(window=sw, min_periods=int(0.8 * sw)).mean()
        long_term_avg = vwap_diff.rolling(window=lw, min_periods=int(0.8 * lw)).mean()
        
        # 计算短期 VWAP 差异平均值与长期 VWAP 差异平均值的比值，得到价格差异比率
        _res = short_term_avg / long_term_avg
        
        return _res



    @staticmethod
    def corr_std_high_vol(data: pd.DataFrame, w: int = 168):
        """
        计算DataFrame中最高价和成交量的滚动相关系数与滚动标准差乘积。

        参数:
        data (pd.DataFrame): 包含金融数据的DataFrame，必须包含 'high' 和 'volume' 列。
        w (int, 可选): 滚动窗口大小，默认为 168。

        返回:
        np.array: 包含最高价和成交量的滚动相关系数与滚动标准差乘积的数组。

        逻辑:
        1. 首先，确保 'data' 包含 'high' 和 'volume' 列。
        2. 使用滚动窗口计算最高价和成交量之间的滚动相关系数。
        3. 使用滚动窗口计算最高价的滚动标准差。
        4. 将滚动相关系数与滚动标准差排名的乘积作为结果。
        5. 返回包含结果的数组。

        示例用法:
        data = pd.DataFrame(...)
        corr_std_high_vol = YourClass.corr_std_high_vol(data, w=168)
        """
    
        # 确保 'data' 包含 'high' 和 'volume' 列
        assert np.isin(['high', 'volume'], data.columns).all()
        
        # 使用滚动窗口计算最高价和成交量之间的滚动相关系数
        corr_high_vol = data['high'].rolling(window=w, min_periods=int(0.8 * w)).corr(data['volume'])
        
        # 使用滚动窗口计算最高价的滚动标准差
        std_high = data['high'].rolling(window=w, min_periods=int(0.8 * w)).std()
        
        # 将滚动相关系数与滚动标准差排名的乘积作为结果
        _res = corr_high_vol * std_high.rank()
        
        return _res