import os
import argparse

import backtrader as bt
import pandas as pd

from aiq.dataset import Dataset, Alpha158, random_split
from aiq.models import XGBModel, LGBModel, DEnsembleModel
from aiq.strategies import TopkDropoutStrategy, TopkStrategy
from aiq.utils.config import config as cfg


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
        'percabs': True
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
    handler = Alpha158(test_mode=True)
    dataset = Dataset(args.data_dir,
                      instruments=args.instruments,
                      start_time=cfg.dataset.segments['train'][0],
                      end_time=cfg.dataset.segments['test'][1],
                      handler=handler)
    test_dataset = random_split(dataset, [cfg.dataset.segments['test']])[0]
    print('Loaded %d items to test dataset' % len(test_dataset))

    # evaluation
    if cfg.model.name == 'XGB':
        model = XGBModel()
    elif cfg.model.name == 'LGB':
        model = LGBModel()
    elif cfg.model.name == 'DoubleEnsemble':
        model = DEnsembleModel()

    model.load(args.save_dir)
    df_prediction = model.predict(test_dataset).to_dataframe()

    # ???????????????
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    comminfo = StampDutyCommissionScheme(stamp_duty=args.stamp_duty, commission=args.commission)
    cerebro.broker.addcommissioninfo(comminfo)

    if args.dump_result:
        log_writer = open(os.path.join(args.save_dir, 'decision_result.txt'), 'w')
    else:
        log_writer = None
    cerebro.addstrategy(TopkDropoutStrategy, log_writer=log_writer)

    # ??????????????????????????????
    with open(os.path.join(args.data_dir, 'instruments/%s.txt' % args.instruments), 'r') as f:
        codes = {line.strip().split()[0] for line in f.readlines()}

    # ??????????????????????????????????????????????????????????????????
    with open(os.path.join(args.data_dir, 'calendars/days.txt'), 'r') as f:
        days = dict()
        for line in f.readlines():
            exchange, date = line.strip().split()
            if cfg.dataset.segments['test'][0] <= date <= cfg.dataset.segments['test'][1]:
                if exchange in days:
                    days[exchange].append(date)
                else:
                    days[exchange] = [date]
        for exchange in days:
            days[exchange] = sorted(days[exchange])

    # ??????????????????
    code_cnt = 0
    for code in codes:
        exchange = code.split('.')[-1]
        assert exchange in ['SH', 'SZ']
        trade_days = days[exchange]
        data = df_prediction[df_prediction['Symbol'] == code]
        if list(data['Date'].values) != trade_days:
            continue
        code_cnt += 1
        data.index = pd.to_datetime(data['Date'])
        data_feed = ZCSPandasData(dataname=data)
        cerebro.adddata(data_feed, name=code)
    print('??????%d??????????????????%d???????????????????????????' % (len(codes), code_cnt))

    # ????????????
    portvalue = cerebro.broker.getvalue()
    print(f'????????????: {round(portvalue, 2)}')

    cerebro.run()

    # ????????????
    portvalue = cerebro.broker.getvalue()
    print(f'????????????: {round(portvalue, 2)}')

    # ??????????????????
    if log_writer is not None:
        log_writer.close()

    # ???????????????
    if args.visualize:
        cerebro.plot()
