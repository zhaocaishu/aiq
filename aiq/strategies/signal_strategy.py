import logging

import pandas as pd
import numpy as np
import backtrader as bt

from aiq.utils.logging import get_logger

logger = get_logger('Signal Strategy')


class TopkDropoutStrategy(bt.Strategy):
    # topk(int): number of stocks in the portfolio
    # n_drop(int): number of stocks to be replaced in each trading date
    # hold_thresh(int): minimum holding days
    params = (
        ('topk', None),
        ('n_drop', None),
        ('hold_thresh', 1)
    )

    def __init__(self):
        """
        Top-k dropout strategy
        """
        self.method_sell = 'bottom'
        self.method_buy = 'top'
        self.reserve = 0.05  # 5% reserve capital
        self.min_score = 0.01  # minimum confidence score to keep/buy a stock

        # 初始化交易指令
        self.order = None
        self.current_stock_list = []

    def generate_trade_decision(self):
        def get_first_n(li, n):
            return list(li)[:n]

        def get_last_n(li, n):
            return list(li)[-n:]

        def filter_stock(li):
            return li

        # generate order list for this adjust date
        sell_order_list = []
        keep_order_list = []
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
                self.p.n_drop + self.p.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(by='score', ascending=False).index, self.p.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.p.n_drop + self.p.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # combine(new stocks + last stocks),  we will drop stocks from this list
        # In case of dropping higher score stock and buying lower score stock.
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(by='score', ascending=False).index

        # Get the stock list we really want to sell (After filtering the case that we sell high and buy low)
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.p.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.p.n_drop, replace=False) if len(last) else [])
            except ValueError:  # No enough candidates
                sell = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # Get the stock list we really want to buy and sell
        buy = today[:len(sell) + self.p.topk - len(last)]
        for code in buy:
            score = pred_score['score'][code]
            if score > self.min_score:
                buy_order_list.append(code)
            else:
                sell_order_list.append(code)

        for code in self.current_stock_list:
            score = pred_score['score'][code]
            if code in sell or score < -self.min_score:
                sell_order_list.append(code)
            else:
                keep_order_list.append(code)

        # Get current stock list
        self.current_stock_list = buy_order_list + keep_order_list

        return buy_order_list, keep_order_list, sell_order_list

    def downcast(self, amount, lot):
        return abs(amount // lot * lot)

    def next(self):
        # 检查是否有指令等待执行
        if self.order:
            return

        buy_order_list, keep_order_list, sell_order_list = self.generate_trade_decision()

        # remove those no longer top ranked
        # do this first to issue sell orders and free cash
        for secu in sell_order_list:
            data = self.getdatabyname(secu)
            self.order = self.order_target_percent(data, 0, name=secu)
            self.log(f"Sell {secu}, price:{data.close[0]:.2f}, pct: 0")

        # re-balance those already top ranked and still there
        for secu in keep_order_list:
            data = self.getdatabyname(secu)
            order_value = self.broker.getvalue() * (1 - self.reserve) / self.p.topk
            order_amount = self.downcast(order_value / data.close[0], 100)
            self.order = self.order_target_size(data, target=order_amount)
            self.log(f"Keep {secu}, price:{data.close[0]:.2f}, amount: {order_amount:.2f}")

        # issue a target order for the newly top ranked stocks
        # do this last, as this will generate buy orders consuming cash
        for secu in buy_order_list:
            data = self.getdatabyname(secu)
            order_value = self.broker.getvalue() * (1 - self.reserve) / self.p.topk
            order_amount = self.downcast(order_value / data.close[0], 100)
            self.order = self.buy(data, size=order_amount, name=secu)
            self.log(f"Buy {secu}, price:{data.close[0]:.2f}, amount: {order_amount}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"""买入: {order.data._name}, 成交量: {order.executed.size}，成交价: {order.executed.price:.2f}""")
                self.log(
                    f'资产：{self.broker.getvalue():.2f} 持仓: {[(x, self.getpositionbyname(x).size) for x in self.current_stock_list]}')
            elif order.issell():
                self.log(f"""卖出: {order.data._name}, 成交量: {order.executed.size}，成交价: {order.executed.price:.2f}""")
                self.log(
                    f'资产：{self.broker.getvalue():.2f} 持仓：{[(x, self.getpositionbyname(x).size) for x in self.current_stock_list]}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/金额不足/拒绝')

        # Write down: no pending order
        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datetime.date(0)
        logger.info('%s, %s' % (dt.isoformat(), txt))
