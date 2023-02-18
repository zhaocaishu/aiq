import os

import backtrader as bt
import pandas as pd

from aiq.strategies import TopkDropoutStrategy


class ZCSPandasData(bt.feeds.PandasData):
    lines = ('score',)
    params = (
        ('datetime', None),
        ('close', 'Close'),
        ('score', 'Volume')
    )


if __name__ == '__main__':
    # 初始化Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.00012)
    cerebro.addstrategy(TopkDropoutStrategy, topk=2, n_drop=1)

    # 添加多个股票回测数据
    codes = ['AAPL', 'BABA', 'GOOG']
    for code in codes:
        file_path = os.path.join('./data/features', code + '.csv')
        data = pd.read_csv(file_path)
        data.index = pd.to_datetime(data['Date'])
        data_feed = ZCSPandasData(dataname=data)
        cerebro.adddata(data_feed, name=code)
    print('合计添加%d个股票数据' % len(codes))

    cerebro.run()

    # 打印回测结束后的总资金
    portvalue = cerebro.broker.getvalue()
    print(f'结束资金: {round(portvalue, 2)}')

    # 可视化
    cerebro.plot()
