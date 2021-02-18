import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union


def setup_logger(log_filename: Union[Path, str, None] = None, log_level: str = 'info') -> None:
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d-%H-%M-%S')
    if log_filename is not None:
        log_filename = '{}-{}'.format(log_filename, date_time)
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    formatter = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
    level = logging.ERROR
    if log_level == 'debug':
        level = logging.DEBUG
    elif log_level == 'info':
        level = logging.INFO
    elif log_level == 'warning':
        level = logging.WARNING
    logging.basicConfig(filename=log_filename,
                        format=formatter,
                        level=level,
                        filemode='w')
    if log_filename is not None:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger('').addHandler(console)
