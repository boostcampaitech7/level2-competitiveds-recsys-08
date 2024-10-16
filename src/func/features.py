import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from typing import Dict
from joblib import Parallel, delayed


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


def haversine_vectorized(coords1, coords2):
    """
    2차원의 coords1와 1차원의 coords2에 대한 harversine 연산을 수행하기 위한 함수입니다.
    """
    R = 6371  
    
    coords1_rad = np.radians(coords1)
    coords2_rad = np.radians(coords2)
    
    dlat = coords2_rad[:, 0] - coords1_rad[:, 0]
    dlon = coords2_rad[:, 1] - coords1_rad[:, 1]
    
    a = np.sin(dlat / 2)**2 + np.cos(coords1_rad[:, 0]) * np.cos(coords2_rad[:, 0]) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def calculate_item_density_single_with_area(apartment_coord: np.ndarray, tree, item_coords: np.ndarray, item_areas: np.ndarray, radius_km: float, zone_area:float) -> np.ndarray:
    """
    아파트와 주어진 대상의 밀도를 계산하는 함수입니다.
    """

    distances, indices = tree.query(apartment_coord, k=len(item_coords))

    # 하버사인 연산 후
    distances_haversine = haversine_vectorized(np.tile(apartment_coord, (len(item_coords), 1)), item_coords)
        
    # 특정 반경 내에 있는 아이템들의 면적 합계
    nearby_areas = np.sum(item_areas[indices][distances_haversine <= radius_km])
    
    # 총면적/반경
    return nearby_areas / zone_area


def calculate_item_density_with_area(apartment_coords: np.ndarray, item_info: pd.DataFrame, radius_km: float, n_jobs=8) -> np.ndarray:
    """
    아파트와 주어진 대상의 밀도를 계산할 때 빠른 실행을 위해 병렬처리 하는 함수입니다.
    """

    item_coordinates = item_info[['latitude', 'longitude']].to_numpy()
    item_areas = item_info['area'].to_numpy() 
    tree = cKDTree(item_coordinates)
    zone_area = np.pi * (radius_km ** 2)

    item_densities = Parallel(n_jobs=n_jobs)(
        delayed(calculate_item_density_single_with_area)(apartment_coord, tree, item_coordinates, item_areas, radius_km, zone_area)
        for apartment_coord in tqdm(apartment_coords)
    )
    
    return np.array(item_densities)


def map_item_count_or_density_with_area(data: pd.DataFrame, item_info: pd.DataFrame, distance_km:float, item_name: str, n_jobs=8) -> pd.DataFrame:
    """
    아파트 데이터에 주어진 대상의 밀도를 매핑하는 함수입니다.
    """

    # 아파트 데이터에 유니크한 좌표를 가진 데이터를 남깁니다.
    unique_apartment_coords = data[['latitude', 'longitude']].drop_duplicates().to_numpy()

    # 유니크한 데이터에 대해 주어진 대상의 밀도를 계산합니다.
    item_densities = calculate_item_density_with_area(unique_apartment_coords, item_info, distance_km, n_jobs=n_jobs)
    
    # 결과를 원래 데이터에 반영합니다.
    result = data[['latitude', 'longitude']].drop_duplicates()
    result[f'{item_name}_density'] = item_densities
    data = data.merge(result, on=['latitude', 'longitude'], how='left')
    
    return data


def count_schools_by_level_single(apartment_coord: np.ndarray, tree, school_coords: np.ndarray, school_levels: np.ndarray, distance_kms: list, levels:list) -> dict:
    """
    각 아파트에 대해 레벨별로 주어진 거리 범위 이내의 학교 수를 세는 함수입니다.
    """
    
    distances, indices = tree.query(apartment_coord, k=len(school_coords))
    
    # 학교 레벨별로 지정해준 거리를 필터링
    level_counts = {}
    for i, level in enumerate(levels):
        if level == 'elementary':
            distance_km = distance_kms['elementary']
        elif level == 'middle':
            distance_km = distance_kms['middle']
        elif level == 'high':
            distance_km = distance_kms['high']
        
        # 연산량을 줄이기 위해 1차적으로 유클리드 거리로 radius_kms 이내의 거리를 가진 인덱스만 필터링
        nearby_indices = indices[distances <= distance_km]
        
        # 해당 인덱스들에 대해 하버사인 거리로 다시 계산
        nearby_school_coords = school_coords[nearby_indices]
        distances_haversine = haversine_vectorized(np.tile(apartment_coord, (len(nearby_school_coords), 1)), nearby_school_coords)
        
        # 하버사인 거리로 레벨별 범위 이내에 있는 학교들 필터링
        nearby_schools = (distances_haversine <= distance_km)
        
        # 학교의 레벨별로 정리
        nearby_school_levels = school_levels[nearby_indices][nearby_schools]
        
        # 학교 수 세기
        level_counts[level] = np.sum(nearby_school_levels == level) if np.sum(nearby_school_levels == level) > 0 else 0
    
    return level_counts


def count_schools_by_level_within_radius(apartment_coords: np.ndarray, school_info: pd.DataFrame, distance_kms: list, n_jobs=8) -> pd.DataFrame:
    """
    학교 레벨 별로 특정 거리 내에 있는 학교의 개수를 세는 함수입니다.
    추가적으로, 연산량을 줄이기 위해 병렬 처리를 사용합니다.
    """
    school_coordinates = school_info[['latitude', 'longitude']].to_numpy()
    school_levels = school_info['schoolLevel'].to_numpy()
    levels = ['elementary', 'middle', 'high']
    
    
    tree = cKDTree(school_coordinates)
    
    school_counts = Parallel(n_jobs=n_jobs)(
        delayed(count_schools_by_level_single)(apartment_coord, tree, school_coordinates, school_levels, distance_kms, levels)
        for apartment_coord in tqdm(apartment_coords)
    )
    
    return pd.DataFrame(school_counts)


def map_school_level_counts(data: pd.DataFrame, school_info: np.ndarray, distance_kms: list, n_jobs=8) -> pd.DataFrame:
    """
    아파트 데이터에 특정 거리 이내 학교 레벨별 개수를 매핑하는 함수힙니다.

    """
    
    unique_apartment_coords = data[['latitude', 'longitude']].drop_duplicates().to_numpy()

    # 학교 레벨별 개수 계산
    school_counts_df = count_schools_by_level_within_radius(unique_apartment_coords, school_info, distance_kms, n_jobs=n_jobs)
    
    result = data[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    result = pd.concat([result, school_counts_df], axis=1)

    # 결과를 원본 데이터에 병합
    data = data.merge(result, on=['latitude', 'longitude'], how='left')
    
    return data
