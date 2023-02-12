import os
import argparse

import backtrader as bt
import pandas as pd

from aiq.dataset import Dataset, Alpha158
from aiq.models import XGBModel
from aiq.strategies import TopkDropoutStrategy
from aiq.utils.config import config as cfg


class ZCSPandasData(bt.feeds.PandasData):
    lines = ('score',)
    params = (
        ('datetime', None),
        ('close', 'Close'),
        ('score', 'PREDICTION')
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Run backtrader')
    # model args
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for backtest')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--instruments', type=str, default='all', help='instruments name')
    parser.add_argument('--save_dir', type=str, help='the saved directory')

    # strategy args
    parser.add_argument('--cash', type=float, default=100000, help='cash value')
    parser.add_argument('--commission', type=float, default=0.00012, help='commission value')
    parser.add_argument('--topk', type=int, default=10, help='number of stocks in the portfolio')
    parser.add_argument('--n_drop', type=int, default=3, help='number of stocks to be replaced in each trading date')
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
    handler = Alpha158(test_mode=True)
    valid_dataset = Dataset(args.data_dir,
                            instruments=args.instruments,
                            start_time=cfg.dataset.segments['valid'][0],
                            end_time=cfg.dataset.segments['valid'][1],
                            handler=handler)

    # evaluation
    model = XGBModel(feature_cols=handler.feature_names,
                     label_col=handler.label_name,
                     model_params=cfg.model.params)
    model.load(os.path.join(args.save_dir, 'model.json'))
    df_prediction = model.predict(valid_dataset).to_dataframe()

    # 初始化策略
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(args.commission)
    cerebro.addstrategy(TopkDropoutStrategy, topk=args.topk, n_drop=args.n_drop)

    # 添加多个股票回测数据
    with open(os.path.join(args.data_dir, 'instruments/%s.txt' % args.instruments), 'r') as f:
        codes = {line.strip().split()[0] for line in f.readlines()}

    # 获取评测时间范围内的交易日期
    with open(os.path.join(args.data_dir, 'calendars/days.txt'), 'r') as f:
        days = []
        for line in f.readlines():
            date = line.strip()
            if cfg.dataset.segments['valid'][0] <= date <= cfg.dataset.segments['valid'][1]:
                days.append(date)

    code_cnt = 0
    for code in codes:
        data = df_prediction[df_prediction['Symbol'] == code]
        if list(data['Date'].values) != days:
            continue
        code_cnt += 1
        data.index = pd.to_datetime(data['Date'])
        data_feed = ZCSPandasData(dataname=data)
        cerebro.adddata(data_feed, name=code)
    print('合计添加%d个股票数据' % code_cnt)

    # 开始资金
    portvalue = cerebro.broker.getvalue()
    print(f'开始资金: {round(portvalue, 2)}')

    cerebro.run()

    # 结果资金
    portvalue = cerebro.broker.getvalue()
    print(f'结束资金: {round(portvalue, 2)}')

    # 可视化结果
    if args.visualize:
        cerebro.plot()
