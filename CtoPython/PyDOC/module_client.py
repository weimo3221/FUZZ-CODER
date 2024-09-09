import requests
import torch
import os
pad = 1000


def writefragment(mes):
    with open("../docset/fragment.txt", "a") as f:
        f.write(repr(mes))
        f.write("\n")


def request_data(data):
    # 用于请求数据
    url = 'http://127.0.0.1:5000/'
    # 这里需要根据module_app或者module_app_model里面的设置动态调整，现在调整的是一样的
    # (Here it needs to be dynamically adjusted according to the settings inside module_app_model.py
    # , which are now adjusted the same way)
    params = {'input': data}
    response = requests.get(url, params=params)
    return response.text


def run(data):
    # data是16进制的字符串，如"0011223344..."，但保证长度为2的倍数
    # (data is a hexadecimal string, such as "0011223344...". but guaranteed to be a multiple of 2)
    data = data.strip().replace('\n', '').replace('\r', '')
    result = request_data(data)
    return result


def readdoc():
    with open("../docset/in.txt", 'r') as f:
        content = f.read()
    return content


def writedoc(data):
    with open("../docset/out.txt", "w") as f:
        f.write(data)


def writeerr(mes):
    with open("../docset/err.txt", "w") as f:
        f.write(repr(mes))


if __name__ == '__main__':
    data = readdoc()
    try:
        result = run(data)
    except Exception as e:
        writeerr(e)
        result = ""
    writedoc(result)
