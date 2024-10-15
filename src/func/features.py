import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from typing import Dict


def categorize_area(x):
    range_start = (x // 50) * 50
    range_end = range_start + 49
    return f"{range_start}~{range_end}"


def categorize_date(x):
    if 1 <= x <= 10:
        return 10
    elif 11 <= x <= 20:
        return 20
    else:
        return 30


def categorize_price(x):
    scale = 10000
    range_start = (x // scale) * scale
    range_end = range_start + scale - 1
    return f"{range_start}~{range_end}"


def haversine(lonlat1: np.ndarray, lonlat2: np.ndarray) -> np.ndarray:
    """
    두 개의 위도/경도 배열을 받아서, 각 좌표 간의 거리를 계산하는 함수입니다.
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


def calculate_nearest(
    apart_coords: np.ndarray, reference_coords: np.ndarray, k: int = 1
) -> np.ndarray:
    """
    각 행의 좌표와 추가 데이터의 건축물들의 좌표를 받아서 가장 가까운 k개의 좌표를 찾고, 거리를 계산하는 함수입니다.
    kdtree를 이용하여 가장 근접한 건축물의 distances(유클리드 거리)와 인덱스를 반환합니다.
    kfmf
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
