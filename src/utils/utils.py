import pandas as pd
import numpy as np
import time

"""
Get function running time.
"""
def get_running_time(func, *args, **kwargs):
    start = time.clock()
    func(*args, **kwargs)
    end = time.clock()
    print("using time: " + str(start - end))
