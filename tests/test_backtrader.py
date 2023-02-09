import os

import backtrader as bt
import pandas as pd

from aiq.strategies import TopkDropoutStrategy


class MultiTestStrategy(bt.Strategy):
    params = (
        ('maperiod', 20),
    )

    def downcast(self, amount, lot):
        return abs(amount // lot * lot)

    def __init__(self):
        # 初始化交易指令
        self.order = None
        self.buy_list = []
        # 添加移动平均线指标，循环计算每个股票的指标
        self.sma = {x: bt.ind.SMA(self.getdatabyname(x), period=self.p.maperiod) for x in self.getdatanames()}

    def next(self):
        if self.order:  # 检查是否有指令等待执行
            return

        # 是否持仓
        if len(self.buy_list) < 2:  # 没有持仓
            # 没有购买的票
            for secu in set(self.getdatanames()) - set(self.buy_list):
                data = self.getdatabyname(secu)
                # 如果突破20日均线买买买
                if data.close[0] > self.sma[secu]:
                    # 买买买
                    order_value = self.broker.getvalue() * 0.48
                    order_amount = self.downcast(order_value / data.close[0], 100)
                    self.order = self.buy(data, size=order_amount, name=secu)
                    self.log(f"买{secu}, price:{data.close[0]:.2f}, amout: {order_amount}")
                    self.buy_list.append(secu)
        elif self.position:
            now_list = []
            for secu in self.buy_list:
                data = self.getdatabyname(secu)
                # 执行卖出条件判断：收盘价格跌破20日均线
                if data.close[0] < self.sma[secu]:
                    # 卖卖卖
                    self.order = self.order_target_percent(data, 0, name=secu)
                    self.log(f"卖{secu}, price:{data.close[0]:.2f}, pct: 0")
                    continue
                now_list.append(secu)
            self.buy_list = now_list

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"""买入: {order.info['name']}, 成交量: {order.executed.size}，成交价: {order.executed.price:.2f}""")
                self.log(
                    f'资产: {self.broker.getvalue():.2f} 持仓: {[(x, self.getpositionbyname(x).size) for x in self.buy_list]}')
            elif order.issell():
                self.log(f"""卖出: {order.info['name']}, 成交量: {order.executed.size}，成交价: {order.executed.price:.2f}""")
                self.log(
                    f'资产: {self.broker.getvalue():.2f} 持仓: {[(x, self.getpositionbyname(x).size) for x in self.buy_list]}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/金额不足/拒绝')

        # Write down: no pending order
        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datetime.date(0)  # 现在的日期
        print('%s , %s' % (dt.isoformat(), txt))

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
    # cerebro.addstrategy(MultiTestStrategy, maperiod=20)
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
