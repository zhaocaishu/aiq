from datetime import datetime, timedelta


def date_add(date, n_days=0):
    return (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=n_days)).strftime('%Y-%m-%d')


def date_diff(date1, date2):
    date1 = datetime.strptime(date1, '%Y-%m-%d')
    date2 = datetime.strptime(date2, '%Y-%m-%d')
    return (date1 - date2).days


def now_date():
    return datetime.now().strftime('%Y-%m-%d')
