import os

import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd

from aiq.strategies import TopkDropoutStrategy


class ZCSPandasData(bt.feeds.PandasData):
    lines = ('score',)
    params = {
        'datetime': None,
        'score': 'Volume'
    }


class StampDutyCommissionScheme(bt.CommInfoBase):
    params = {
        'stamp_duty': 0.001,
        'commission': 0.00012,
        'percabs': True
    }

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:
            return size * price * self.p.commission
        elif size < 0:
            return -size * price * (self.p.stamp_duty + self.p.commission)
        else:
            return 0


if __name__ == '__main__':
    # 初始化Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    comminfo = StampDutyCommissionScheme(stamp_duty=0.001, commission=0.00012)
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.addstrategy(TopkDropoutStrategy, topk=2, n_drop=1)

    # 回测分析器
    cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # 添加多个股票回测数据
    codes = ['000951.SZ', '601099.SH', '688326.SH']
    for code in codes:
        file_path = os.path.join('./data/features', code + '.csv')
        data = pd.read_csv(file_path)
        data.index = pd.to_datetime(data['Date'])
        data_feed = ZCSPandasData(dataname=data)
        cerebro.adddata(data_feed, name=code)
    print('合计添加%d个股票数据' % len(codes))

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

    # 可视化
    cerebro.plot()
