import numpy as np
import scipy
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def pd_read_csv(file_name):
    df = pd.read_csv('csv_bento/' + file_name,
              )
    return df


def main():
    KFold = KFold(n_splits=5, shuffle=True, random_state=0)

    train = pd_read_csv('train.csv')
    test = pd_read_csv('test.csv')
    train_y = train.y
    train.drop("y", axis=1, inplace=True)

    # for sepalate after
    train['is_train'] = 1
    test['is_train'] = 0
    len(train), len(test)


if __name__ == '__main__':
