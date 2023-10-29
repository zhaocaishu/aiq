import os
import argparse
from datetime import datetime, timedelta

import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd

from aiq.dataset import Dataset, DataLoader, Alpha158, Alpha101, ts_split
from aiq.models import XGBModel, LGBModel, DEnsembleModel
from aiq.strategies import TopkDropoutStrategy
from aiq.utils.config import config as cfg
from aiq.utils.date import date_add


class ZCSPandasData(bt.feeds.PandasData):
    lines = ('score',)
    params = {
        'datetime': None,
        'score': 'PREDICTION'
    }


class StampDutyCommissionScheme(bt.CommInfoBase):
    params = {
        'stamp_duty': 0.001,
        'commission': 0.00012,
        'stocklike': True,
        'commtype': bt.CommInfoBase.COMM_PERC
    }

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:
            return size * price * self.p.commission
        elif size < 0:
            return -size * price * (self.p.stamp_duty + self.p.commission)
        else:
            return 0


def parse_args():
    parser = argparse.ArgumentParser(description='Run backtrader')
    # model args
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for backtest')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--instruments', type=str, default='all', help='instruments name')
    parser.add_argument('--save_dir', type=str, help='the saved directory')

    # strategy args
    parser.add_argument('--cash', type=float, default=100000, help='cash value')
    parser.add_argument('--stamp_duty', type=float, default=0.001, help='stamp duty')
    parser.add_argument('--commission', type=float, default=0.00012, help='commission value')
    parser.add_argument('--dump_result', action='store_true', default=False, help='whether to dump decision result')
    parser.add_argument('--visualize', action='store_true', default=False, help='whether to plot chart')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # parse args
    args = parse_args()

    # config
    cfg.from_file(args.cfg_file)
    print(cfg)

    # dataset
    print(cfg.dataset.segments)
    if cfg.model.name in ['PatchTST', 'NLinear']:
        test_dataset = TSDataset(data_dir=args.data_dir, save_dir=args.save_dir, instruments=args.instruments,
                                 start_time=cfg.dataset.start_time, end_time=cfg.dataset.end_time,
                                 segment=cfg.dataset.segments['test'], feature_names=cfg.dataset.feature_names,
                                 adjust_price=True, seq_len=cfg.model.params.seq_len,
                                 pred_len=cfg.model.params.pred_len, training=False)
    else:
        handlers = (Alpha158(), Alpha101())
        dataset = Dataset(args.data_dir,
                          instruments=args.instruments,
                          start_time=cfg.dataset.start_time,
                          end_time=cfg.dataset.end_time,
                          handlers=handlers)
        test_dataset = ts_split(dataset, [cfg.dataset.segments['test']])[0]
    print('Loaded %d items to test dataset' % len(test_dataset))

    # model
    if cfg.model.name == 'XGB':
        model = XGBModel()
    elif cfg.model.name == 'LGB':
        model = LGBModel()
    elif cfg.model.name == 'DoubleEnsemble':
        model = DEnsembleModel()
    elif cfg.model.name == 'PatchTST':
        model = PatchTSTModel(model_params=cfg.model.params)
    elif cfg.model.name == 'NLinear':
        model = NLinearModel(model_params=cfg.model.params)
    model.load(args.save_dir)

    # prediction
    df_prediction = model.predict(test_dataset).to_dataframe()

    # 初始化策略
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    comminfo = StampDutyCommissionScheme(stamp_duty=args.stamp_duty, commission=args.commission)
    cerebro.broker.addcommissioninfo(comminfo)

    if args.dump_result:
        log_writer = open(os.path.join(args.save_dir, 'decision_result.txt'), 'w')
    else:
        log_writer = None
    cerebro.addstrategy(TopkDropoutStrategy, log_writer=log_writer)

    # 回测分析器
    cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # 添加多个股票回测数据
    symbols = DataLoader.load_symbols(args.data_dir, args.instruments,
                                      start_time=cfg.dataset.segments['test'][0],
                                      end_time=cfg.dataset.segments['test'][1])

    # 获取评测时间范围内的全部交易日期，区分交易所
    days = dict()
    df = pd.read_csv(os.path.join(args.data_dir, 'calendars/days.csv'))
    for index, row in df.iterrows():
        exchange, date = row['Exchange'], row['Trade_date']
        if cfg.dataset.segments['test'][0] <= date <= cfg.dataset.segments['test'][1]:
            if exchange in days:
                days[exchange].append(date)
            else:
                days[exchange] = [date]
    for exchange in days:
        days[exchange] = sorted(days[exchange])

    # 加入回测数据
    count = 0
    for symbol, list_date in symbols:
        exchange = symbol.split('.')[-1]
        assert exchange in ['SH', 'SZ']
        trade_days = days[exchange]
        data = df_prediction[df_prediction['Symbol'] == symbol]
        if list(data['Date'].values) != trade_days:
            continue
        count += 1
        data.index = pd.to_datetime(data['Date'])
        data_feed = ZCSPandasData(dataname=data)
        cerebro.adddata(data_feed, name=symbol)
    print('共计%d个股票，添加%d个股票进入回测数据' % (len(symbols), count))

    # 开始资金
    start_portfolio = cerebro.broker.getvalue()

    thestrats = cerebro.run()
    thestrat = thestrats[0]

    # 结束资金
    end_portfolio = cerebro.broker.getvalue()

    # 打印回测结果
    print(f'Start portfolio: %f, end portfolio: %f' % (round(start_portfolio, 2), round(end_portfolio, 2)))
    print(f'Annual Return:', thestrat.analyzers.annual_return.get_analysis())
    print(f'Max DrawDown:', thestrat.analyzers.drawdown.get_analysis()['max']['drawdown'] / 100)

    # 关闭文件句柄
    if log_writer is not None:
        log_writer.close()

    # 可视化结果
    if args.visualize:
        cerebro.plot()
