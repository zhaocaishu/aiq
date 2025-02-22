import mysql.connector


def init_db_connection():
    connection = mysql.connector.connect(
        host="127.0.0.1",
        user="zcs",
        passwd="2025zcsdaydayup",
        database="stock_info",
    )

    return connection
