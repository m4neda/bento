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


def prepare(df):
    # datetime
    df = pd.concat([df, df['datetime'].str.split('-', expand=True)], axis=1)
    df.rename(columns={0: 'year', 1: 'month', 2: 'day'}, inplace=True)

    # week nanはない
    dummy_week_df = pd.get_dummies(df[['week']])
    df = pd.concat([df, dummy_week_df], axis=1)

    # name(メニュー)　nanはない
    dummy_menu_df = pd.get_dummies(df[['name']])
    df = pd.concat([df, dummy_menu_df], axis=1)

    # 特記事項 文字列が入っている。
    df.remarks.fillna(-1, inplace=True)
    df.remarks.where(df.remarks == -1, 1, inplace=True)

    # 1 = 給料日
    df.payday.fillna(0, inplace=True)

    # event
    df.event.fillna(-1, inplace=True)
    df.event.where(df.event == -1, 1, inplace=True)

    # weather nanはない
    dummy_weather_df = pd.get_dummies(df[['weather']])
    df = pd.concat([df, dummy_weather_df], axis=1)

    # precipitation
    df.precipitation = df.precipitation.str.replace('--', '0')
    # ダミー変数に変換した元の次元を削除
    df.drop(['datetime', 'week', 'name', 'weather'], axis=1, inplace=True)
    return df


def main():
    KF = KFold(n_splits=5, shuffle=True, random_state=0)

    train = pd_read_csv('train.csv')
    test = pd_read_csv('test.csv')
    train_y = train.y
    train.drop("y", axis=1, inplace=True)

    # for sepalate after
    train['is_train'] = 1
    test['is_train'] = 0
    len(train), len(test)
    # join
    joined_train_test = pd.concat([train, test])
    len(joined_train_test.index)

    # 前処理
    joined_df = prepare(joined_train_test)


    sepalated_train_df = joined_df[joined_df.is_train == 1]
    sepalated_test_df = joined_df[joined_df.is_train == 0]

    train_df = sepalated_train_df.drop('is_train', axis=1)
    test_df = sepalated_test_df.drop('is_train', axis=1)

    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    X = train_df
    y = train_y

    lr = LinearRegression()
    lr.fit(X, y)

    # Scoring method mean_squared_error was renamed to neg_mean_squared_error
    scores = cross_val_score(lr, X, y, scoring="neg_mean_squared_error", cv=KF)
    rmse_scores = np.sqrt(-scores)
    print('rmse scores mean:{0}'.format(rmse_scores.mean()))

    # 検証データの予測
    test_predict_results = lr.predict(test_df)

    test_predict_series = pd.Series(test_predict_results)
    test_predict_series_rounded = np.round(test_predict_results)

    test_2 = pd.read_csv('csv_bento/test.csv')
    index_df = pd.DataFrame(test_2.datetime)

    predict_df = pd.DataFrame(test_predict_series_rounded)
    predict_results_df = index_df.join(predict_df)

    predict_results_df[[0]] = predict_results_df[[0]].astype(int)

    predict_results_df.to_csv(
        'bento_predict_use_100data.csv',
        header=False,
        index=False,
    )


if __name__ == '__main__':
    main()