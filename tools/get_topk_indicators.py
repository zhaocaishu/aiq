import argparse
import os

from aiq.models import XGBModel, LGBModel
from aiq.utils.config import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Feature importance')
    parser.add_argument('--save_dir', type=str, help='the saved directory')
    parser.add_argument('--model_type', type=str, default='XGB', help='model type')
    parser.add_argument('--topk', type=int, default=50, help='top k')

    args = parser.parse_args()

    return args


def main():
    # parse args
    args = parse_args()

    # model
    if args.model_type == 'LGB':
        model = LGBModel()
    elif args.model_type == 'XGB':
        model = XGBModel()
    model.load(args.save_dir)

    # feature importance
    topk_indicators = []
    feature_importance = model.get_feature_importance(importance_type='gain')
    count = 0
    for row in feature_importance.items():
        if count >= args.topk:
            break
        if args.model_type == 'XGB':
            feature_index = int(row[0].replace('f', ''))
        else:
            feature_index = int(row[0])
        topk_indicators.append(model.feature_cols[feature_index])
        count += 1

    print(','.join(topk_indicators))


if __name__ == '__main__':
    main()
