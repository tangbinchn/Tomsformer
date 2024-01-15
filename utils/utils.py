import datetime
import os

import pytz


def get_save_path(args, config):
    current_time = get_current_time(level='second')
    save_model_path = os.path.join(config['save_model_path'], current_time)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    save_model_path = os.path.join(save_model_path, f'batch_size{args.batch_size}model.pth')

    return save_model_path


def get_current_time(level='second'):
    # 检查level参数
    try:
        assert level in ['second', 'day']
    except AssertionError:
        print("ERROR: get_current_time level must be 'second' or 'day'.")
        exit()

    global formatted_time
    # 设置东八区时区
    eastern_eight_zone = pytz.timezone('Asia/Shanghai')
    # 获取当前时间
    current_time = datetime.datetime.now(pytz.utc)
    # 转换为东八区时间
    eastern_eight_time = current_time.astimezone(eastern_eight_zone)

    # 格式化输出
    if level == 'second':
        formatted_time = eastern_eight_time.strftime("%Y%m%d-%H:%M:%S")
    elif level == 'day':
        formatted_time = eastern_eight_time.strftime("%Y%m%d")
    return formatted_time
