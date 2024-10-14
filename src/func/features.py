import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

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
    '''
        두 개의 위도/경도 배열을 받아서, 각 좌표 간의 거리를 계산하는 함수입니다.
    '''
    # 지구의 반지름 (km)
    R = 6371.0

    # 경도와 위도를 분리
    lon1, lat1 = np.radians(lonlat1[:, 0]), np.radians(lonlat1[:, 1])
    lon2, lat2 = np.radians(lonlat2[:, 0]), np.radians(lonlat2[:, 1])

    # 차이 계산
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine 공식 적용
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # 결과적으로 거리는 R * c
    distance = R * c

    return distance

def calculate_nearest(apart_coords: np.ndarray, reference_coords: np.ndarray, k: int = 1) -> np.ndarray:
    '''
        각 행의 좌표와 추가 데이터의 건축물들의 좌표를 받아서 가장 가까운 k개의 좌표를 찾고, 거리를 계산하는 함수입니다.
        kdtree를 이용하여 가장 근접한 건축물의 distances(유클리드 거리)와 인덱스를 반환합니다.
        kfmf 
    '''
    tree = cKDTree(reference_coords)
    
    distances, indices = tree.query(apart_coords, k=k)
    
    return distances, tree.data[indices]

def calculate_nearest_subway_distance(apart_coords: np.ndarray, subway_coords: np.ndarray) -> np.ndarray:
    '''
        각 행에 대해 가장 가까운 지하철역까지의 거리를 계산하는 함수입니다.
        calculate_nearest 함수를 통해 가장 가까운 지하철의 좌표를 받아 실제 거리를 계산한 후 반환합니다.
    '''
    distances, nearest_subway_coords = calculate_nearest(apart_coords, subway_coords)
    
    # haversine을 이용해 실제 거리 계산
    return haversine(nearest_subway_coords, apart_coords)