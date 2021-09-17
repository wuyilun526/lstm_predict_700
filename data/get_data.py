"""
通过接口获取腾讯股票数据，参考：https://blog.csdn.net/geofferysun/article/details/114640013。
调用例子：
param=代码，日k，开始日期，结束日期，获取多少个交易日，前复权
https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=hk00700,day,2021-07-01,2021-07-10,10,qfq
返回例子：
{
    "code":0,
    "msg":"",
    "data":{
        "hk00700":{
            "day":[
                [
                    "2021-07-02",    # 交易日
                    "598.500",    # 开盘价
                    "574.500",    # 收盘价
                    "598.500",    # 最高价
                    "572.500",    # 最低价
                    "24938624.000"    # 总手
                ],
            ],
            "prec":"3.700",
            "vcm":"",
            "version":"15"
        }
    }
}
"""
import datetime
import requests

import db

def get_data_from_api(begin_date, end_date, days_num):
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=hk00700,day,%s,%s,%s,qfq"
    url = url % (begin_date, end_date, days_num)
    try_times = 3
    while try_times:
        try:
            r = requests.get(url, timeout=10)
        except Exception as e:
            print('Error when get data:', e)
        else:
            return r.json()
        try_times -= 1

def get_datas(begin_date, end_date, days_num):
    print(begin_date, end_date, days_num)
    results = get_data_from_api(begin_date, end_date, days_num)
    print(results)
    datas = results['data']['hk00700']['day']
    print(datas)
    for data in datas:
        db.insert_record(*data[:6])

def get_dates(begin_date, end_date):
    dates = []
    dt = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    date = begin_date[:]
    while date <= end_date:
        dates.append(date)
        dt = dt + datetime.timedelta(1)
        date = dt.strftime("%Y-%m-%d")
    return dates

def get_history_data():
    begin_date = '2004-01-01'
    end_date = '2021-09-10'
    dates = get_dates(begin_date, end_date)
    for date in dates:
        print(date)
        get_datas(date, date, 2)    # 一天一天的取


if __name__ == "__main__":
    # begin_date = '2021-01-05'
    # end_date = '2021-01-05'
    # days_num = 1
    # results = get_data_from_api(begin_date, end_date, days_num)
    # datas = results['data']['hk00700']['day']
    # for data in datas:
    #     print(data)
    # print(len(datas))
    # get_datas(begin_data, end_data, days_num)
    get_history_data()
