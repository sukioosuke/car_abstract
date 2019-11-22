import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

# cpu 数量
cores = cpu_count()
# 分块个数
partitions = cores


def parallelize(df, func):
    # 数据切分
    data_split = np.array_list(df, partitions)
    # 进程池
    pool = Pool(partitions)
    # 数据分发
    data = pd.concat(pool.map(func, data_split))
    # 关闭进程池，并等待所有进程处理完毕
    pool.close()
    pool.join()
    return data
