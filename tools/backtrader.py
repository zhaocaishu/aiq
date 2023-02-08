import os
import argparse

import backtrader as bt
import pandas as pd


class MultiStrategy(bt.Strategy):
    params = (
        ('num_positions', 20),
    )

    def downcast(self, amount, lot):
        return abs(amount // lot * lot)

    def __init__(self):
        # 初始化交易指令
        self.order = None
        self.buy_list = []

    def next(self):
        # 检查是否有指令等待执行
        if self.order:
            return

        # 是否持仓
        if len(self.buy_list) < self.p.num_positions:  # 没有持仓
            # 没有购买的票
            for secu in set(self.getdatanames()) - set(self.buy_list):
                data = self.getdatabyname(secu)
                # 如果涨幅大于2%买买买
                if (data.score[0] + 1.0) > 0.02:
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
                # 执行卖出条件判断：涨幅小于0.5%
                if (data.score[0] + 1.0) < 0.005:
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
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(f"""买入{order.info['name']}, 成交量{order.executed.size}，成交价{order.executed.price:.2f}""")
                self.log(
                    f'资产：{self.broker.getvalue():.2f} 持仓：{[(x, self.getpositionbyname(x).size) for x in self.buy_list]}')
            elif order.issell():
                self.log(f"""卖出{order.info['name']}, 成交量{order.executed.size}，成交价{order.executed.price:.2f}""")
                self.log(
                    f'资产：{self.broker.getvalue():.2f} 持仓：{[(x, self.getpositionbyname(x).size) for x in self.buy_list]}')

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
        ('score', 'prediction')
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Run backtrader')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--save_dir', type=str, help='the saved directory')
    parser.add_argument('--cash', type=float, default=100000, help='cash value')
    parser.add_argument('--commission', type=float, default=0.001, help='commission value')
    parser.add_argument('--num_positions', type=int, default=20, help='number of positions')
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
    cerebro.addstrategy(MultiStrategy, args.num_positions)

    # 添加多个股票回测数据
    with open(os.path.join(args.data_dir, 'instruments/csi.txt'), 'r') as f:
        codes = {line.split()[0] for line in f.readlines()}

    for code in codes:
        file_path = os.path.join(args.save_dir, code + '.csv')
        if not os.path.exists(file_path):
            continue
        data = pd.read_csv(file_path)
        data.index = pd.to_datetime(data['Date'])
        data_feed = ZCSPandasData(dataname=data)
        cerebro.adddata(data_feed, name=code)
        print('添加股票数据：code: %s' % code)

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
