from datetime import datetime, timedelta


def date_add(date, n_days=0):
    return (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=n_days)).strftime('%Y-%m-%d')


def date_diff(date_start, date_end):
    date_start = datetime.strptime(date_start, '%Y-%m-%d')
    date_end = datetime.strptime(date_end, '%Y-%m-%d')
    return (date_end - date_start).days


def now_date():
    return datetime.now().strftime('%Y-%m-%d')
