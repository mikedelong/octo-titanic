# https://blog.socialcops.com/engineering/machine-learning-python/
import logging
import time

import pandas as pd

if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    input_file = '../input/titanic3.xls'
    logger.debug('reading data from %s' % input_file)

    titanic_df = pd.read_excel(input_file, encoding='ISO-8859-1')
    logger.debug('data has shape %d x %d' % titanic_df.shape)
    logger.debug('data has columns %s' % titanic_df.columns.values)
    logger.debug('mean by passenger class: \n%s' % titanic_df.groupby('pclass').mean())
    logger.debug('survived mean: %.4f' % titanic_df['survived'].mean())

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
