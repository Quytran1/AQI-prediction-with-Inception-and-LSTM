import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def train_test_split_by_column(df, column_split, target):
    X_train_df = pd.DataFrame()
    X_test_df = pd.DataFrame()

    for station in df[column_split].unique():
        df_temp = df.loc[df[column_split] == station]
        threshold = int(df_temp.shape[0] * 0.8)

        train_temp = df_temp[:threshold]
        test_temp = df_temp[threshold:]

        X_train_df = pd.concat([X_train_df, train_temp], axis=0)
        X_test_df = pd.concat([X_test_df, test_temp], axis=0)

    y_train_df = X_train_df[target].values
    y_test_df = X_test_df[target].values

    X_train_df.set_index((i for i in range(len(X_train_df))), inplace=True)
    X_test_df.set_index((i for i in range(len(X_test_df))), inplace=True)

    # X_train_df.drop([target],axis=1,inplace=True)
    # X_test_df.drop([target],axis=1,inplace=True)

    return X_train_df, y_train_df, X_test_df, y_test_df


def window_slide(train):

    window_size = 24
    X = []
    Y = []
    for station in train["station"].unique():
        df_station = train[train["station"] == station]
        label = df_station["AQI"]
        label = np.reshape(label, (len(label), 1))
        df_station.drop(["AQI"], axis=1, inplace=True)
        # label_index=df_station.index[0]+window_size-1
        for i in range(window_size, len(df_station)):
            X.append(df_station.iloc[i - window_size : i, :].values)
            Y.append(label[i, :])

    return np.array(X), np.array(Y)


def window_slide_one_station(train):

    window_size = 24
    X = []
    Y = []
    label = train["AQI"]
    train.drop(["AQI"], axis=1, inplace=True)
    label = np.reshape(label, (len(label), 1))
    for i in range(window_size, len(train)):
        X.append(train.iloc[i - window_size : i, :].values)
        Y.append(label[i, :])

    return np.array(X), np.array(Y)


# #Scale dữ liệu về đoạn 0-1
def scale_data(df_train, df_test, list_scale_features):
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_train = df_train[list_scale_features].values
    scaled_values_train = scaler.fit_transform(values_train)
    df_train[list_scale_features] = scaled_values_train

    values_test = df_test[list_scale_features].values
    scaled_values_test = scaler.transform(values_test)
    df_test[list_scale_features] = scaled_values_test

    return df_train, df_test, scaler


def read_data_csv(data_path):
    df = pd.read_csv(data_path)
    return df


def read_and_process_data():
    df_Taiwan = read_data_csv(
        "D:\AQI-Forecasting\data\data processed\Taiwan_data_processed.csv"
    )
    df_Taiwan = df_Taiwan.drop(columns=["time"])

    X_train_Taiwan, y_train_Taiwan, X_test_Taiwan, y_test_Taiwan = (
        train_test_split_by_column(df_Taiwan, "station", "AQI")
    )

    X_train_m1_Taiwan = X_train_Taiwan[X_train_Taiwan["station"] == "Banqiao"].drop(
        ["station"], axis=1
    )

    X_train_m2_Taiwan = X_train_Taiwan[X_train_Taiwan["station"] == "Tucheng"].drop(
        ["station"], axis=1
    )

    X_train_m3_Taiwan = X_train_Taiwan[X_train_Taiwan["station"] == "Xinzhuang"].drop(
        ["station"], axis=1
    )

    X_test_m1_Taiwan = X_test_Taiwan[X_test_Taiwan["station"] == "Banqiao"].drop(
        ["station"], axis=1
    )

    X_test_m2_Taiwan = X_test_Taiwan[X_test_Taiwan["station"] == "Tucheng"].drop(
        ["station"], axis=1
    )

    X_test_m3_Taiwan = X_test_Taiwan[X_test_Taiwan["station"] == "Xinzhuang"].drop(
        ["station"], axis=1
    )

    scaled_features = ["PM2.5", "PM10", "SO2", "CO", "O3", "NO2", "NOx", "NO"]

    df_train_m1_Taiwan, df_test_m1_Taiwan, scaler = scale_data(
        X_train_m1_Taiwan, X_test_m1_Taiwan, scaled_features
    )

    df_train_m2_Taiwan, df_test_m2_Taiwan, scaler = scale_data(
        X_train_m2_Taiwan, X_test_m2_Taiwan, scaled_features
    )

    df_train_m3_Taiwan, df_test_m3_Taiwan, scaler = scale_data(
        X_train_m3_Taiwan, X_test_m3_Taiwan, scaled_features
    )
    print(df_train_m1_Taiwan.head(10))

    X_train_m1_final_Taiwan, y_train_m1_final_Taiwan = window_slide_one_station(
        df_train_m1_Taiwan
    )
    X_test_m1_final_Taiwan, y_test_m1_final_Taiwan = window_slide_one_station(
        df_test_m1_Taiwan
    )

    ##### lưu 2 file train test của thành phố Banqiao
    # np.savez(
    #     f"{base_data_path}/Banqiao_train.npz",
    #     samples=X_train_m1_final_Taiwan,
    #     labels=y_train_m1_final_Taiwan,
    # )
    # np.savez(
    #     f"{base_data_path}/Banqiao_test.npz",
    #     samples=X_test_m1_final_Taiwan,
    #     labels=y_test_m1_final_Taiwan,
    # )

    X_train_m2_final_Taiwan, y_train_m2_final_Taiwan = window_slide_one_station(
        df_train_m2_Taiwan
    )

    X_test_m2_final_Taiwan, y_test_m2_final_Taiwan = window_slide_one_station(
        df_test_m2_Taiwan
    )

    # ##### lưu 2 file train test của thành phố Tucheng
    # np.savez(
    #     f"{base_data_path}/Tucheng_train.npz",
    #     samples=X_train_m2_final_Taiwan,
    #     labels=y_train_m2_final_Taiwan,
    # )

    # np.savez(
    #     f"{base_data_path}/Tucheng_test.npz",
    #     samples=X_test_m2_final_Taiwan,
    #     labels=y_test_m2_final_Taiwan,
    # )

    X_train_m3_final_Taiwan, y_train_m3_final_Taiwan = window_slide_one_station(
        df_train_m3_Taiwan
    )

    X_test_m3_final_Taiwan, y_test_m3_final_Taiwan = window_slide_one_station(
        df_test_m3_Taiwan
    )

    # ##### lưu 2 file train test của thành phố Xinzhuang
    # np.savez(
    #     f"{base_data_path}/Xinzhuang_train.npz",
    #     samples=X_train_m3_final_Taiwan,
    #     labels=y_train_m3_final_Taiwan,
    # )

    # np.savez(
    #     f"{base_data_path}/Xinzhuang_test.npz",
    #     samples=X_test_m3_final_Taiwan,
    #     labels=y_test_m3_final_Taiwan,
    # )

    X_train_final_Taiwan = np.concatenate(
        [X_train_m1_final_Taiwan, X_train_m2_final_Taiwan, X_train_m3_final_Taiwan]
    )

    X_test_final_Taiwan = np.concatenate(
        [X_test_m1_final_Taiwan, X_test_m2_final_Taiwan, X_test_m3_final_Taiwan]
    )

    y_train_final_Taiwan = np.concatenate(
        [y_train_m1_final_Taiwan, y_train_m2_final_Taiwan, y_train_m3_final_Taiwan]
    )
    y_test_final_Taiwan = np.concatenate(
        [y_test_m1_final_Taiwan, y_test_m2_final_Taiwan, y_test_m3_final_Taiwan]
    )

    np.savez(
        f"{base_data_path}/Taiwan_train.npz",
        samples=X_train_final_Taiwan,
        labels=y_train_final_Taiwan,
    )

    np.savez(
        f"{base_data_path}/Taiwan_test.npz",
        samples=X_test_final_Taiwan,
        labels=y_test_final_Taiwan,
    )


if __name__ == "__main__":
    base_data_path = "D:/AQI-Forecasting/data/dataset_Taiwan"
    read_and_process_data()
    pass

# batchsize lớn/nhỏ thì sẽ nhưu thế nào
# shuffle theo batchsize hay là shuffle dữ liệu trước rồi chia batchsize
#
