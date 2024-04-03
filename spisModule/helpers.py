# This file contains some helper functions that can be useful in some situations

import logging as log
import sys
from typing import Callable, ParamSpec, TypeVar


T = TypeVar('T')
P = ParamSpec('P')

def default_log_config():
    log.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s -  %(message)s', 
                    level=log.DEBUG, 
                    filename="latest_run.log",
                    filemode='w')
    handler = log.StreamHandler(sys.stdout)
    log.getLogger().addHandler(handler)
    handler.setLevel(log.INFO)
    handler.setFormatter(log.Formatter(fmt = '%(asctime)s - %(levelname)s - %(filename)s -  %(message)s'))


def log_function_entry_and_exit(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        log.info("Entered function: " + func.__name__)
        x =  func(*args, **kwargs)
        log.info("Exited function: " + func.__name__)
        return x 
    return wrapper
