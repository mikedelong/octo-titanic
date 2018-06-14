# https://blog.socialcops.com/engineering/machine-learning-python/
import logging
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def preprocess_titanic_df(df):
    result = df.copy()
    label_encoder = LabelEncoder()
    result.sex = label_encoder.fit_transform(result.sex)
    result.embarked = label_encoder.fit_transform(result.embarked)
    result = result.drop(['name', 'ticket', 'home.dest'], axis=1)
    return result


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

    partial_columns = ['body', 'cabin', 'boat']
    t0 = titanic_df.drop(partial_columns, axis=1)
    t0['home.dest'] = t0['home.dest'].fillna('NA')
    t0 = t0.dropna()
    processed_df = preprocess_titanic_df(t0)

    for target_column in ['embarked', 'pclass', 'survived', 'sex']:
        X = processed_df.drop([target_column], axis=1).values
        y = processed_df[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf_dt = DecisionTreeClassifier(max_depth=10)

        clf_dt.fit(X_train, y_train)
        logger.debug('target: %s score: %.4f' % (target_column, clf_dt.score(X_test, y_test)))

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
