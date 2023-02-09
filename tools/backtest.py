import os
import argparse

import backtrader as bt
import pandas as pd

from aiq.strategies import TopkDropoutStrategy


class ZCSPandasData(bt.feeds.PandasData):
    lines = ('score',)
    params = (
        ('datetime', None),
        ('close', 'Close'),
        ('score', 'prediction')
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Run backtrader')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--save_dir', type=str, help='the saved directory')
    parser.add_argument('--cash', type=float, default=100000, help='cash value')
    parser.add_argument('--commission', type=float, default=0.001, help='commission value')
    parser.add_argument('--topk', type=int, default=50, help='number of stocks in the portfolio')
    parser.add_argument('--n_drop', type=int, default=5, help='number of stocks to be replaced in each trading date')
    parser.add_argument('--visualize', action='store_true', default=False, help='whether to plot chart')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # parse args
    args = parse_args()

    # 初始化Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(args.commission)
    cerebro.addstrategy(TopkDropoutStrategy, topk=args.topk, n_drop=args.n_drop)

    # 添加多个股票回测数据
    with open(os.path.join(args.data_dir, 'instruments/csi300.txt'), 'r') as f:
        codes = {line.strip() for line in f.readlines()}

    with open(os.path.join(args.data_dir, 'calendars/days.txt'), 'r') as f:
        days = {line.strip() for line in f.readlines()}

    code_cnt = 0
    for code in codes:
        file_path = os.path.join(args.save_dir, code + '.csv')
        if not os.path.exists(file_path):
            continue
        data = pd.read_csv(file_path)
        if not (data['Date'].values == days).all():
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
