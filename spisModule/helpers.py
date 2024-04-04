# This file contains some helper functions that can be useful in some situations

import logging
import logging.handlers
import sys
from typing import Callable, ParamSpec, TypeVar
from functools import wraps
from pathlib import Path


log = logging.getLogger("spisModule")

T = TypeVar('T')
P = ParamSpec('P')

def default_log_config():

    formater = logging.Formatter(fmt = '%(asctime)s - %(levelname)s:%(name)s - %(module)s - %(funcName)s -  %(message)s')
    filename="latest_run.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


    handler_file = logging.handlers.RotatingFileHandler(filename=filename, mode='a', maxBytes=5*1024*1024, 
                                                        backupCount=1, encoding=None)
    handler_file.setLevel(logging.DEBUG)
    handler_file.setFormatter(formater)
    logging.getLogger().addHandler(handler_file)
    

    handler_stdout = logging.StreamHandler(sys.stdout)  
    handler_stdout.setLevel(logging.INFO)
    handler_stdout.setFormatter(formater)
    logging.getLogger().addHandler(handler_stdout)



def make_log_message(function:Callable[P,T], message:str, level:int) -> logging.LogRecord:
    return log.makeRecord(name = log.name,
                          level=level,
                          fn=function.__code__.co_filename,
                          lno=function.__code__.co_firstlineno,
                          msg=message,
                          func=function.__name__,
                          args=(),
                          exc_info=None)


def log_function_entry_and_exit(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        log.handle(make_log_message(func, "Entered function: " + func.__name__, logging.INFO))
        x =  func(*args, **kwargs)
        log.handle(make_log_message(func, "Exited function: " + func.__name__, logging.INFO))
        return x 
    return wrapper


def LogFileOpening[T](function: Callable[[Path], T]) -> Callable[[Path], T]:
    '''A decorator to add logging to a function that reads a file given a path. Furthermore it checks if a file exist.'''
    @wraps(function)
    def inner(path: Path) -> T:
        log.handle(make_log_message(function, f"Reading file:\t {str(path)} \t during the call of function {function.__name__} ", logging.DEBUG))
        if not path.exists():
            log.handle(make_log_message(function, f"Cant read file:\t {str(path)} \t during the call of function {function.__name__}. File doesn't exist.", logging.ERROR))
        return function(path)
    return inner

