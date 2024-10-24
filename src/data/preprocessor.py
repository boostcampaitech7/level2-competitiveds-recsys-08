import pandas as pd

import src.data.features as ft


class DataPreprocessor:
    def __init__(self, path, dataset):
        self.path = path
        self.dataset = dataset
        self.train_data = pd.read_csv(path + dataset + ".csv")
        self.park = pd.read_csv(path + "parkInfo.csv")
        self.school = pd.read_csv(path + "schoolinfo.csv")
        self.subway = pd.read_csv(path + "subwayInfo.csv")

    def remove_duplicates(self):
        self.train_data = self.train_data.drop_duplicates(
            subset=self.train_data.columns.drop("index"), keep="first"
        )

    def add_nearest_subway_distance(self):
        self.train_data = ft.calculate_nearest_subway_distance(
            self.train_data, self.subway
        )

    def add_nearest_school_distance(self):
        self.train_data = ft.calculate_nearest_school_distance(
            self.train_data, self.school
        )

    def add_park_density(self, radius_km=3):
        self.train_data = ft.map_park_density(self.train_data, self.park, radius_km)

    def add_school_level_counts(self, distance_kms):
        self.train_data = ft.map_school_level_counts(
            self.train_data, self.school, distance_kms, n_jobs=8
        )

    def select_features(self, train_columns):
        return self.train_data[train_columns]

    def preprocess(self):
        if self.dataset == "train":
            self.remove_duplicates()

        self.add_nearest_subway_distance()
        self.add_nearest_school_distance()
        self.add_park_density()
        distance_kms = {
            "elementary": 1,
            "middle": 5,
            "high": 5,
        }
        self.add_school_level_counts(distance_kms)

        train_columns = [
            "area_m2",
            "contract_year_month",
            "floor",
            "built_year",
            "latitude",
            "longitude",
            "nearest_subway_distance",
            "nearest_elementary_distance",
            "nearest_middle_distance",
            "nearest_high_distance",
            "park_density",
            "elementary",
            "middle",
            "high",
        ]

        if self.dataset == "train":
            X_train, y_train = (
                self.train_data.drop(columns=["deposit"]),
                self.train_data["deposit"],
            )
            X_train = self.select_features(train_columns)
            return X_train, y_train

        X_train = self.select_features(train_columns)

        return X_train
