from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
from tqdm import tqdm
from typing import Dict
from joblib import Parallel, delayed



def apply_kmeans_clustering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_clusters: int = 25,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KMeans 클러스터링을 적용하고 클러스터 라벨을 반환합니다.

    이 함수는 주어진 훈련 데이터에 KMeans 클러스터링을 적용하고,
    테스트 데이터에 대해서는 가장 가까운 훈련 데이터 포인트의 클러스터 라벨을 할당합니다.

    Args:
        X_train (pd.DataFrame): 훈련 데이터. 'latitude'와 'longitude' 열을 포함해야 합니다.
        X_test (pd.DataFrame): 테스트 데이터. 'latitude'와 'longitude' 열을 포함해야 합니다.
        n_clusters (int, optional): KMeans의 클러스터 수. 기본값은 25입니다.
        random_state (int, optional): 난수 시드. 기본값은 42입니다.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 훈련 데이터와 테스트 데이터에 대한 클러스터 라벨.
            첫 번째 요소는 훈련 데이터의 클러스터 라벨이고,
            두 번째 요소는 테스트 데이터의 클러스터 라벨입니다.
    """
    # 예외처리
    if X_train.empty or X_test.empty:
        raise ValueError("Input DataFrames cannot be empty")
    if "latitude" not in X_train.columns or "longitude" not in X_train.columns:
        raise ValueError("X_train must contain 'latitude' and 'longitude' columns")
    if "latitude" not in X_test.columns or "longitude" not in X_test.columns:
        raise ValueError("X_test must contain 'latitude' and 'longitude' columns")

    # 스케일링 된 위치 데이터 생성
    X_train_location, X_test_location = _get_scaled_location(X_train, X_test)

    # Train 데이터에 KMeans 적용
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    train_clusters = kmeans.fit_predict(X_train_location)

    # Validation 데이터에 클러스터 라벨 할당
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_train_location)

    _, indices = nn.kneighbors(X_test_location)
    test_clusters = train_clusters[indices.flatten()]

    return train_clusters, test_clusters


def apply_dbscan_clustering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    eps: float = 0.0175,
    min_samples: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DBSCAN 클러스터링을 적용하고 클러스터 라벨을 반환합니다.

    이 함수는 주어진 훈련 데이터에 DBSCAN 클러스터링을 적용하고,
    테스트 데이터에 대해서는 가장 가까운 훈련 데이터 포인트의 클러스터 라벨을 할당합니다.

    Args:
        X_train (pd.DataFrame): 훈련 데이터. 'latitude'와 'longitude' 열을 포함해야 합니다.
        X_test (pd.DataFrame): 테스트 데이터. 'latitude'와 'longitude' 열을 포함해야 합니다.
        eps (float, optional): DBSCAN의 epsilon 파라미터. 기본값은 0.0175입니다.
        min_samples (int, optional): DBSCAN의 최소 샘플 수 파라미터. 기본값은 15입니다.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 훈련 데이터와 테스트 데이터에 대한 클러스터 라벨.
            첫 번째 요소는 훈련 데이터의 클러스터 라벨이고,
            두 번째 요소는 테스트 데이터의 클러스터 라벨입니다.
    """
    # 예외처리
    if X_train.empty or X_test.empty:
        raise ValueError("Input DataFrames cannot be empty")
    if "latitude" not in X_train.columns or "longitude" not in X_train.columns:
        raise ValueError("X_train must contain 'latitude' and 'longitude' columns")
    if "latitude" not in X_test.columns or "longitude" not in X_test.columns:
        raise ValueError("X_test must contain 'latitude' and 'longitude' columns")

    # 스케일링 된 위치 데이터 생성
    X_train_location, X_test_location = _get_scaled_location(X_train, X_test)

    # Train 데이터에 DBSCAN 적용
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    train_clusters = dbscan.fit_predict(X_train_location)

    # Validation 데이터에 클러스터 라벨 할당
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_train_location)

    _, indices = nn.kneighbors(X_test_location)
    test_clusters = train_clusters[indices.flatten()]

    return train_clusters, test_clusters


def _get_scaled_location(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    훈련 및 테스트 데이터에서 위치 정보를 추출하고 스케일링합니다.

    이 함수는 입력 데이터프레임에서 'latitude'와 'longitude' 열을 추출하고,
    StandardScaler를 사용하여 이 데이터를 스케일링합니다.

    Args:
        X_train (pd.DataFrame): 훈련 데이터. 'latitude'와 'longitude' 열을 포함해야 합니다.
        X_test (pd.DataFrame): 테스트 데이터. 'latitude'와 'longitude' 열을 포함해야 합니다.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 스케일링된 훈련 및 테스트 위치 데이터.
            첫 번째 요소는 스케일링된 훈련 위치 데이터이고,
            두 번째 요소는 스케일링된 테스트 위치 데이터입니다.
    """
    # 예외처리
    if "latitude" not in X_train.columns or "longitude" not in X_train.columns:
        raise ValueError("X_train must contain 'latitude' and 'longitude' columns")
    if "latitude" not in X_test.columns or "longitude" not in X_test.columns:
        raise ValueError("X_test must contain 'latitude' and 'longitude' columns")

    X_train_location = X_train[["latitude", "longitude"]].copy()
    X_test_location = X_test[["latitude", "longitude"]].copy()

    scaler = StandardScaler()
    X_train_location_scaled = scaler.fit_transform(X_train_location)
    X_test_location_scaled = scaler.transform(X_test_location)

    return X_train_location_scaled, X_test_location_scaled


def haversine(lonlat1: np.ndarray, lonlat2: np.ndarray) -> np.ndarray:
    """
    두 개의 위도/경도 배열을 받아서, 각 좌표 간의 거리를 계산합니다.

    Args:
        lonlat1 (np.ndarray): 첫 번째 좌표 배열 (경도, 위도)
        lonlat2 (np.ndarray): 두 번째 좌표 배열 (경도, 위도)

    Returns:
        np.ndarray: 각 좌표 쌍 사이의 거리 (km)
    """
    # 지구의 반지름 (km)
    R = 6371.0

    # 경도와 위도를 분리
    lon1, lat1 = np.radians(lonlat1[:, 0]), np.radians(lonlat1[:, 1])
    lon2, lat2 = np.radians(lonlat2[:, 0]), np.radians(lonlat2[:, 1])

    # 차이 계산
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine 공식 적용
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # 결과적으로 거리는 R * c
    distance = R * c

    return distance


def haversine_vectorized(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    2차원의 coords1와 1차원의 coords2에 대한 harversine 연산을 수행하기 위한 함수입니다.
    """
    R = 6371

    coords1_rad = np.radians(coords1)
    coords2_rad = np.radians(coords2)

    dlat = coords2_rad[:, 0] - coords1_rad[:, 0]
    dlon = coords2_rad[:, 1] - coords1_rad[:, 1]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(coords1_rad[:, 0]) * np.cos(coords2_rad[:, 0]) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_nearest(
    apart_coords: np.ndarray, reference_coords: np.ndarray, k: int = 1
) -> np.ndarray:
    """
    각 아파트 좌표에 대해 가장 가까운 k개의 참조 좌표를 찾습니다.

    Args:
        apart_coords (np.ndarray): 아파트 좌표 배열
        reference_coords (np.ndarray): 참조 좌표 배열
        k (int, optional): 찾을 가장 가까운 이웃의 수. 기본값은 1입니다.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 거리와 해당하는 참조 좌표
    """
    tree = cKDTree(reference_coords)

    distances, indices = tree.query(apart_coords, k=k)

    return distances, tree.data[indices]


def calculate_nearest_subway_distance(
    apart_coords: np.ndarray, subway_coords: np.ndarray
) -> np.ndarray:
    """
    각 행에 대해 가장 가까운 지하철역까지의 거리를 계산하는 함수입니다.
    calculate_nearest 함수를 통해 가장 가까운 지하철의 좌표를 받아 실제 거리를 계산한 후 반환합니다.
    """
    distances, nearest_subway_coords = calculate_nearest(apart_coords, subway_coords)

    # haversine을 이용해 실제 거리 계산
    return haversine(nearest_subway_coords, apart_coords)


def calculate_nearest_school_distance(
    apart_coords: np.ndarray, school_info: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    아파트의 위도/경도 좌표와 학교 정보를 입력받아 각 학교 레벨별로 가장 가까운 학교까지의 거리를 계산하는 함수입니다.
    """

    # 학교 레벨별로 거리 계산을 위한 빈 딕셔너리 초기화
    nearest_distances: Dict[str, np.ndarray] = {}

    # 각 학교 레벨에 대한 거리 계산
    for level in school_info["schoolLevel"].unique():
        level_coords = school_info[school_info["schoolLevel"] == level][
            ["latitude", "longitude"]
        ].to_numpy()

        if level_coords.shape[0] > 0:
            tree = cKDTree(level_coords)
            distances, indices = tree.query(apart_coords)

            nearest_distances[level] = haversine(tree.data[indices], apart_coords)

    return nearest_distances


# 최근접 공원 거리 계산 함수
def calculate_nearest_park_distance(
    apart_coords: np.ndarray, park_coords: np.ndarray
) -> np.ndarray:
    """
    아파트 좌표와 공원 좌표를 받아서, 각 아파트에서 가장 가까운 공원까지의 거리를 계산하는 함수입니다.
    KDTree를 사용하여 가장 가까운 공원을 찾고, Haversine 공식을 사용해 실제 거리를 계산합니다.
    """
    # 공원 좌표를 KDTree로 변환
    tree = cKDTree(np.radians(park_coords))

    # 최근접 공원 거리 검색
    distances, indices = tree.query(np.radians(apart_coords), k=1)

    return haversine(apart_coords, park_coords[indices])


def nearest_park_area(
    apart_coords: np.ndarray, park_coords: np.ndarray, park_areas: np.ndarray
) -> np.ndarray:
    """
    아파트 좌표와 공원 좌표, 공원 면적 정보를 받아, 각 아파트에서 가장 가까운 공원의 면적을 반환하는 함수입니다.
    KDTree를 사용하여 가장 가까운 공원을 찾고, 해당 공원의 면적을 반환합니다.
    """

    # 공원 좌표를 KDTree로 변환 (라디안으로 변환하여 처리)
    tree = cKDTree(np.radians(park_coords))

    # 아파트 좌표에 대해 가장 가까운 공원의 인덱스를 검색 (라디안으로 변환하여 처리)
    distances, indices = tree.query(np.radians(apart_coords), k=1)

    # 가장 가까운 공원의 면적을 반환
    return park_areas[indices]


def calculate_item_density_single_with_area(
    apartment_coord: np.ndarray,
    tree,
    item_coords: np.ndarray,
    item_areas: np.ndarray,
    radius_km: float,
    zone_area: float,
) -> np.ndarray:
    """
    아파트와 주어진 대상의 밀도를 계산하는 함수입니다.
    """

    distances, indices = tree.query(apartment_coord, k=len(item_coords))

    # 하버사인 연산 후
    distances_haversine = haversine_vectorized(
        np.tile(apartment_coord, (len(item_coords), 1)), item_coords
    )

    # 특정 반경 내에 있는 아이템들의 면적 합계
    nearby_areas = np.sum(item_areas[indices][distances_haversine <= radius_km])

    # 총면적/반경
    return nearby_areas / zone_area


def calculate_item_density_with_area(
    apartment_coords: np.ndarray, item_info: pd.DataFrame, radius_km: float, n_jobs=8
) -> np.ndarray:
    """
    각 아파트 좌표에 대해 주어진 반경 내의 아이템 밀도를 계산합니다.

    Args:
        apartment_coords (np.ndarray): 아파트 좌표 배열
        item_info (pd.DataFrame): 아이템 정보 (위도, 경도, 면적 포함)
        radius_km (float): 밀도 계산을 위한 반경 (km)
        n_jobs (int, optional): 병렬 처리를 위한 작업 수. 기본값은 8입니다.

    Returns:
        np.ndarray: 각 아파트에 대한 아이템 밀도
    """

    item_coordinates = item_info[["latitude", "longitude"]].to_numpy()
    item_areas = item_info["area"].to_numpy()
    tree = cKDTree(item_coordinates)
    zone_area = np.pi * (radius_km**2)

    item_densities = Parallel(n_jobs=n_jobs)(
        delayed(calculate_item_density_single_with_area)(
            apartment_coord, tree, item_coordinates, item_areas, radius_km, zone_area
        )
        for apartment_coord in tqdm(apartment_coords)
    )

    return np.array(item_densities)


def map_item_count_or_density_with_area(
    data: pd.DataFrame,
    item_info: pd.DataFrame,
    distance_km: float,
    item_name: str,
    n_jobs=8,
) -> pd.DataFrame:
    """
    아파트 데이터에 주어진 대상의 밀도를 매핑하는 함수입니다.
    """

    # 아파트 데이터에 유니크한 좌표를 가진 데이터를 남깁니다.
    unique_apartment_coords = (
        data[["latitude", "longitude"]].drop_duplicates().to_numpy()
    )

    # 유니크한 데이터에 대해 주어진 대상의 밀도를 계산합니다.
    item_densities = calculate_item_density_with_area(
        unique_apartment_coords, item_info, distance_km, n_jobs=n_jobs
    )

    # 결과를 원래 데이터에 반영합니다.
    result = data[["latitude", "longitude"]].drop_duplicates()
    result[f"{item_name}_density"] = item_densities
    data = data.merge(result, on=["latitude", "longitude"], how="left")

    return data


def count_schools_by_level_single(
    apartment_coord: np.ndarray,
    tree,
    school_coords: np.ndarray,
    school_levels: np.ndarray,
    distance_kms: list,
    levels: list,
) -> dict:
    """
    각 아파트 좌표에 대해 지정된 반경 내의 학교 수를 학교 레벨별로 계산합니다.

    Args:
        apartment_coords (np.ndarray): 아파트 좌표 배열
        school_info (pd.DataFrame): 학교 정보 (위도, 경도, 학교 레벨 포함)
        distance_kms (Dict[str, float]): 각 학교 레벨별 고려할 거리
        n_jobs (int, optional): 병렬 처리를 위한 작업 수. 기본값은 8입니다.

    Returns:
        pd.DataFrame: 각 아파트에 대한 학교 레벨별 개수
    """

    distances, indices = tree.query(apartment_coord, k=len(school_coords))

    # 학교 레벨별로 지정해준 거리를 필터링
    level_counts = {}
    for i, level in enumerate(levels):
        if level == "elementary":
            distance_km = distance_kms["elementary"]
        elif level == "middle":
            distance_km = distance_kms["middle"]
        elif level == "high":
            distance_km = distance_kms["high"]

        # 연산량을 줄이기 위해 1차적으로 유클리드 거리로 radius_kms 이내의 거리를 가진 인덱스만 필터링
        nearby_indices = indices[distances <= distance_km]

        # 해당 인덱스들에 대해 하버사인 거리로 다시 계산
        nearby_school_coords = school_coords[nearby_indices]
        distances_haversine = haversine_vectorized(
            np.tile(apartment_coord, (len(nearby_school_coords), 1)),
            nearby_school_coords,
        )

        # 하버사인 거리로 레벨별 범위 이내에 있는 학교들 필터링
        nearby_schools = distances_haversine <= distance_km

        # 학교의 레벨별로 정리
        nearby_school_levels = school_levels[nearby_indices][nearby_schools]

        # 학교 수 세기
        level_counts[level] = (
            np.sum(nearby_school_levels == level)
            if np.sum(nearby_school_levels == level) > 0
            else 0
        )

    return level_counts


def count_schools_by_level_within_radius(
    apartment_coords: np.ndarray,
    school_info: pd.DataFrame,
    distance_kms: list,
    n_jobs=8,
) -> pd.DataFrame:
    """
    학교 레벨 별로 특정 거리 내에 있는 학교의 개수를 세는 함수입니다.
    추가적으로, 연산량을 줄이기 위해 병렬 처리를 사용합니다.
    """
    school_coordinates = school_info[["latitude", "longitude"]].to_numpy()
    school_levels = school_info["schoolLevel"].to_numpy()
    levels = ["elementary", "middle", "high"]

    tree = cKDTree(school_coordinates)

    school_counts = Parallel(n_jobs=n_jobs)(
        delayed(count_schools_by_level_single)(
            apartment_coord,
            tree,
            school_coordinates,
            school_levels,
            distance_kms,
            levels,
        )
        for apartment_coord in tqdm(apartment_coords)
    )

    return pd.DataFrame(school_counts)


def map_school_level_counts(
    data: pd.DataFrame, school_info: np.ndarray, distance_kms: list, n_jobs=8
) -> pd.DataFrame:
    """
    아파트 데이터에 특정 거리 이내 학교 레벨별 개수를 매핑하는 함수힙니다.

    """

    unique_apartment_coords = (
        data[["latitude", "longitude"]].drop_duplicates().to_numpy()
    )

    # 학교 레벨별 개수 계산
    school_counts_df = count_schools_by_level_within_radius(
        unique_apartment_coords, school_info, distance_kms, n_jobs=n_jobs
    )

    result = data[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    result = pd.concat([result, school_counts_df], axis=1)

    # 결과를 원본 데이터에 병합
    data = data.merge(result, on=["latitude", "longitude"], how="left")

    return data

def nearest_subway_is_transfer(
    apart_coords: np.ndarray, subway_coords: np.ndarray, is_transfer_station: np.ndarray
) -> pd.DataFrame:
    """
    아파트 데이터에 대해 가장 가까운 지하철역이 환승역인지 아닌지를 매핑하는 함수.

    Args:
        apart_coords (np.ndarray): 아파트 좌표 배열 (latitude, longitude)
        subway_coords (np.ndarray): 지하철 좌표 배열 (latitude, longitude)
        is_transfer_station (np.ndarray): 각 지하철역이 환승역인지 여부 (1: 환승역, 0: 비환승역)

    Returns:
        np.ndarray: 아파트에 대해 가장 가까운 지하철역의 환승 여부 배열 (0 또는 1)
    """
    # cKDTree를 사용해 가장 가까운 지하철역 찾기
    tree = cKDTree(subway_coords)
    distances, indices = tree.query(apart_coords, k=1, workers=-1)

    # 가장 가까운 지하철역의 환승 여부 매핑
    nearest_subway_is_transfer = is_transfer_station[indices]

    return nearest_subway_is_transfer
