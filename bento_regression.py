import numpy as np
import scipy
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score


def get_dummy_encoded_datetime_df(df):
    get_year = lambda x: str(x.year)
    year = df.datetime.map(get_year)
    get_month = lambda x: str(x.month)
    month = df.datetime.map(get_month)
    get_day = lambda x: str(x.day)
    day = df.datetime.map(get_day)
    splitted_datetime_df = pd.DataFrame({
        'year': year,
        'month': month,
        'day': day, },
        columns=['year', 'month', 'day']
    )
    dummy_df = pd.get_dummies(splitted_datetime_df[['year', 'month', 'day']])
    return dummy_df


def prepare(df):
    # fillna to 0
    df.fillna(0, inplace=True)

    dummy_datetime_df = get_dummy_encoded_datetime_df(df)
    df = df.join(dummy_datetime_df)

    dummy_week_df = pd.get_dummies(df[['week']])
    df = df.join(dummy_week_df)
    dummy_menu_df = pd.get_dummies(df[['name']])
    df = df.join(dummy_menu_df)

    # 特記事項 文字列が入っている。フラグにする
    df.remarks.where(df.remarks == 0, 1, inplace=True)

    df.event.fillna(-1, inplace=True)
    df.event.where(df.event == -1, 1, inplace=True)

    dummy_weather_df = pd.get_dummies(df[['weather']])
    df = df.join(dummy_weather_df)

    df.precipitation = df.precipitation.str.replace('--', '0')
    # ダミー変数に変換した元の次元を削除
    df.drop(['datetime', 'week', 'name', 'weather'], axis=1, inplace=True)

    return df


def pd_read_csv(file_name):
    df = pd.read_csv('csv_bento/' + file_name,
                     parse_dates=['datetime',],
              )
    return df


def main():
    train = pd_read_csv('train.csv')

    # delete y
    train_y = train.y
    train.drop("y", axis=1, inplace=True)

    train_df = prepare(train)

    X = train_df
    y = train_y

    lr = LinearRegression().fit(X, y)

    loo = LeaveOneOut()
    scores = cross_val_score(lr, X, y, scoring="neg_mean_squared_error", cv=loo)
    rmse_scores = np.sqrt(-scores)
    print('Cross Validation rmse score{0}'.format(rmse_scores))
    print('Average:{0}'.format(rmse_scores.mean()))


if __name__ == '__main__':
    main()