import os
import argparse

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Run collect days')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--csv_file', type=str, help='csv file contains dates')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # parse args
    args = parse_args()

    fp = open(os.path.join(args.data_dir, 'calendars/days.txt'), 'w')
    data = pd.read_csv(args.csv_file)
    dates = []
    for date in data['Date']:
        dates.append(date)
        fp.write(date + '\n')
    fp.close()
