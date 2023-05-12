import json

import pandas as pd
import backtrader as bt

from aiq.utils.logging import get_logger

logger = get_logger('Signal Strategy')


class TopkDropoutStrategy(bt.Strategy):
    params = {
        'topk': 30,  # number of stocks in the portfolio
        'n_drop': 10,  # number of stocks to be replaced in each trading date
        'hold_thresh': 1,  # minimum holding days
        'log_writer': None  # write handler
    }

    def __init__(self):
        """
        Top-k dropout strategy
        """
        self.method_sell = 'bottom'
        self.method_buy = 'top'
        self.risk_degree = 0.95  # 5% reserve capital

        # 当前持仓股票列表
        self.current_stock_list = []

        # To keep track of pending orders
        self.order = {}
        for i, data in enumerate(self.datas):
            self.order[data._name] = None

        # trade day index
        self.trade_day_index = 0
        self.trade_day_interval = 5

    def generate_trade_decision(self):
        def get_first_n(li, n):
            return list(li)[:n]

        def get_last_n(li, n):
            return list(li)[-n:]

        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []

        # load score
        index = []
        scores = []
        for i, data in enumerate(self.datas):
            index.append(data._name)
            scores.append(data.score[0])
        pred_score = pd.DataFrame({'score': scores}, index=index)

        # last position (sorted by score)
        last = pred_score.reindex(self.current_stock_list).sort_values(by='score', ascending=False).index

        # The new stocks today want to buy **at most**
        if self.method_buy == "top":
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(by='score', ascending=False).index,
                self.p.n_drop + self.p.topk - len(last)
            )
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # combine(new stocks + last stocks),  we will drop stocks from this list
        # In case of dropping higher score stock and buying lower score stock.
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(by='score', ascending=False).index

        # Get the stock list we really want to sell (After filtering the case that we sell high and buy low)
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.p.n_drop))]
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # Get the stock list we really want to buy and sell
        buy = today[:len(sell) + self.p.topk - len(last)]
        for code in buy:
            buy_order_list.append(code)

        for code in self.current_stock_list:
            if code in sell:
                sell_order_list.append(code)

        # Get current stock list
        self.current_stock_list = list(set(self.current_stock_list) - set(sell_order_list)) + buy_order_list

        return buy_order_list, sell_order_list

    @staticmethod
    def downcast(amount, lot):
        return abs(amount // lot * lot)

    def next(self):
        # 检查是否有指令等待执行，如果有就不执行这根bar
        for i, data in enumerate(self.datas):
            if self.order[data._name]:
                return

        # 如果是指数的最后一根bar，则退出，防止取下一日开盘价越界错
        if len(self.datas[0]) == self.data0.buflen():
            return

        if self.trade_day_index % self.trade_day_interval == 0:
            buy_order_list, sell_order_list = self.generate_trade_decision()

            if self.p.log_writer is not None:
                order_str = json.dumps({'date': str(self.datetime.date(0)), 'buy': buy_order_list, 'sell': sell_order_list})
                self.p.log_writer.write(order_str + '\n')

            # cash for buy
            cash = self.broker.getcash()

            # remove those no longer top ranked
            # do this first to issue sell orders and free cash
            for secu in sell_order_list:
                data = self.getdatabyname(secu)
                order_price = data.open[1]
                self.order[secu] = self.order_target_percent(data=data, target=0.0, price=order_price, name=secu)

                # 因为设置的是先卖出后买入, 需要根据卖出的股票更新可用现金。如果设置先买入再卖出，则不需更新可用现金
                order_size = self.getposition(data).size
                cash += order_price * order_size

            # issue a target order for the newly top ranked stocks
            # do this last, as this will generate buy orders consuming cash
            target_value = cash * self.risk_degree / len(buy_order_list) if len(buy_order_list) > 0 else 0
            for secu in buy_order_list:
                data = self.getdatabyname(secu)
                order_price = data.close[1]
                order_size = self.downcast(target_value / order_price, 100)
                self.order[secu] = self.buy(data=data, size=order_size, price=order_price, name=secu)

        self.trade_day_index += 1

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"""买入: {order.data._name}, 成交量: {order.executed.size}，成交价: {order.executed.price:.2f}""")
                positions = [(data._name, self.getposition(data).size) for data in self.datas if self.getposition(data).size]
                self.log(
                    f'现金：{self.broker.getcash():.2f} 资产：{self.broker.getvalue():.2f} 持仓: {positions}')
            elif order.issell():
                self.log(
                    f"""卖出: {order.data._name}, 成交量: {order.executed.size}，成交价: {order.executed.price:.2f}""")
                positions = [(data._name, self.getposition(data).size) for data in self.datas if self.getposition(data).size]
                self.log(
                    f'现金：{self.broker.getcash():.2f} 资产：{self.broker.getvalue():.2f} 持仓：{positions}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/金额不足/拒绝')

        # Write down: no pending order
        self.order[order.data._name] = None

    def log(self, txt, dt=None):
        dt = dt or self.datetime.date(0)
        logger.info('%s, %s' % (dt.isoformat(), txt))
