import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def train_test_split_by_column(df, column_split, target):
    X_train_df = pd.DataFrame()
    X_test_df = pd.DataFrame()

    for city in df[column_split].unique():
        df_temp = df.loc[df[column_split] == city]
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
    for city in train["City"].unique():
        df_city = train[train["City"] == city]
        label = df_city["AQI"]
        label = np.reshape(label, (len(label), 1))
        df_city.drop(["AQI"], axis=1, inplace=True)
        # label_index=df_city.index[0]+window_size-1
        for i in range(window_size, len(df_city)):
            X.append(df_city.iloc[i - window_size : i, :].values)
            Y.append(label[i, :])

    return np.array(X), np.array(Y)


def window_slide_one_city(train):

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
    df_India = read_data_csv(
        "D:\AQI-Forecasting\data\data processed\India_data_processed.csv"
    )
    df_India = df_India.drop(columns=["NH3"])

    X_train_India, y_train_India, X_test_India, y_test_India = (
        train_test_split_by_column(df_India, "City", "AQI")
    )

    X_train_m1_India = X_train_India[X_train_India["City"] == "Bengaluru"].drop(
        ["City"], axis=1
    )

    X_train_m2_India = X_train_India[X_train_India["City"] == "Hyderabad"].drop(
        ["City"], axis=1
    )

    X_train_m3_India = X_train_India[X_train_India["City"] == "Delhi"].drop(
        ["City"], axis=1
    )

    X_test_m1_India = X_test_India[X_test_India["City"] == "Bengaluru"].drop(
        ["City"], axis=1
    )

    X_test_m2_India = X_test_India[X_test_India["City"] == "Hyderabad"].drop(
        ["City"], axis=1
    )

    X_test_m3_India = X_test_India[X_test_India["City"] == "Delhi"].drop(
        ["City"], axis=1
    )

    scaled_features = ["PM2.5", "PM10", "SO2", "CO", "O3", "NO2", "NOx", "NO"]

    df_train_m1_India, df_test_m1_India, scaler = scale_data(
        X_train_m1_India, X_test_m1_India, scaled_features
    )

    df_train_m2_India, df_test_m2_India, scaler = scale_data(
        X_train_m2_India, X_test_m2_India, scaled_features
    )

    df_train_m3_India, df_test_m3_India, scaler = scale_data(
        X_train_m3_India, X_test_m3_India, scaled_features
    )

    #####
    X_train_m1_final_India, y_train_m1_final_India = window_slide_one_city(
        df_train_m1_India
    )
    X_test_m1_final_India, y_test_m1_final_India = window_slide_one_city(
        df_test_m1_India
    )

    ##### lưu 2 file train test của thành phố Bengaluru
    np.savez(
        f"{base_data_path}/Bengaluru_train.npz",
        samples=X_train_m1_final_India,
        labels=y_train_m1_final_India,
    )
    np.savez(
        f"{base_data_path}/Bengaluru_test.npz",
        samples=X_test_m1_final_India,
        labels=y_test_m1_final_India,
    )

    X_train_m2_final_India, y_train_m2_final_India = window_slide_one_city(
        df_train_m2_India
    )

    X_test_m2_final_India, y_test_m2_final_India = window_slide_one_city(
        df_test_m2_India
    )

    ##### lưu 2 file train test của thành phố Hyderabad
    np.savez(
        f"{base_data_path}/Hyderabad_train.npz",
        samples=X_train_m2_final_India,
        labels=y_train_m2_final_India,
    )

    np.savez(
        f"{base_data_path}/Hyderabad_test.npz",
        samples=X_test_m2_final_India,
        labels=y_test_m2_final_India,
    )

    X_train_m3_final_India, y_train_m3_final_India = window_slide_one_city(
        df_train_m3_India
    )

    X_test_m3_final_India, y_test_m3_final_India = window_slide_one_city(
        df_test_m3_India
    )

    ##### lưu 2 file train test của thành phố Dehli
    np.savez(
        f"{base_data_path}/Dehli_train.npz",
        samples=X_train_m3_final_India,
        labels=y_train_m3_final_India,
    )

    np.savez(
        f"{base_data_path}/Dehli_test.npz",
        samples=X_test_m3_final_India,
        labels=y_test_m3_final_India,
    )


if __name__ == "__main__":
    base_data_path = "D:\AQI-Forecasting\data\dataset_india1"
    read_and_process_data()
    pass
