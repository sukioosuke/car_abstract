import pandas as pd
import numpy as np
import time
from multiprocessing import cpu_count, Pool

# 分块个数
partitions = cpu_count()


def parallelize(df, func):
    # 数据切分
    data_split = np.array_split(df, partitions)
    # 进程池
    pool = Pool(partitions)
    # 数据分发
    data = pd.concat(pool.map(func(data_split)))
    # 关闭进程池，并等待所有进程处理完毕
    pool.close()
    pool.join()
    return data


def get_running_time(func, *args, **kwargs):
    start = time.clock()
    func(*args, **kwargs)
    end = time.clock()
    print("using time: " + str(start - end))
