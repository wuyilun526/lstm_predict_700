# -*- coding:utf-8 -*- 

import os
import json
import sqlite3
import platform

import pandas as pd
from sqlalchemy import create_engine

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)

SYSTEM = platform.system()
DB_NAME = '%s/db/00700.db' % parent_dir
if SYSTEM == 'Windows':
    DB_NAME = '%s\\db\\00700.db' % parent_dir


# "2021-07-02",    # 交易日
# "598.500",    # 开盘价
# "574.500",    # 收盘价   
# "598.500",    # 最高价
# "572.500",    # 最低价
# "24938624.000"    # 总手

def create_record_table():
    print(DB_NAME)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        'CREATE TABLE record_00700(\
            id INTEGER PRIMARY KEY autoincrement,\
            day VARCHAR(10) NOT NULL unique,\
            open_price float NOT NULL,\
            close_price float NOT NULL,\
            highest_price float NOT NULL,\
            lowest_price float NOT NULL,\
            deal_times double NOT NULL\
        )'
        )
    cursor.close()
    conn.commit()
    conn.close()


def insert_record(day, open_price, close_price, highest_price, lowest_price, deal_times):
    print(day, open_price, close_price, highest_price, lowest_price, deal_times)
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "insert into record_00700(day, open_price, \
                close_price, highest_price, lowest_price, deal_times) \
            values(?,?,?,?,?,?)", (
                day, open_price, close_price,
                highest_price, lowest_price, deal_times))
        cursor.close()
        conn.commit()
        conn.close()
    except Exception as e:
        print('Error when insert into db:', e)


def read_data(begin_day=None, end_day=None):
    # engine = create_engine('sqlite3:%s' % DB_NAME)
    conn = sqlite3.connect(DB_NAME)
    sql = 'select * from record_00700'
    if begin_day and end_day:
        sql = "%s where day >= '%s' and day <= '%s'" % (
            sql, begin_day, end_day)
    elif begin_day:
        sql = "%s where day >= '%s'" % (
            sql, begin_day)
    elif end_day:
        sql = "%s where day <= '%s'" % (
            sql, end_day)
    frame = pd.read_sql(sql, conn)
    return frame



if __name__ == '__main__':

    # create_record_table()
    begin_day = '2004-06-16'
    end_day = '2004-06-29'
    df = read_data(begin_day, end_day)
    print(df[:3])
