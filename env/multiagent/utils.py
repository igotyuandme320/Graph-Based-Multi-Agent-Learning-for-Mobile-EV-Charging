import math
import numpy as np

def haversine(lon1, lat1, lon2, lat2):
    # distance between latitudes
    # and longitudes
    lon1 = float(lon1)
    lon2 = float(lon2)
    lat1 = float(lat1)
    lat2 = float(lat2)
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
         math.cos(lat1) * math.cos(lat2))
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c


def min_max_norm(X, feature_range=(0, 1)):
    X = np.asarray(X, dtype=float)
    if np.all(X == 0):
        return X
    r1, r2 = feature_range   # 将数据归一化到 [r1, r2] 之间
    xmin, xmax = X.min(axis=0), X.max(axis=0)  # 得到数据的最大最小值
    span = xmax - xmin
    X_std = (X - xmin) / np.where(span == 0, 1, span)    # 标准化到 [0, 1]
    X_scaled = X_std * (r2 - r1) + r1   # 数据平移缩放到 [r1, r2]
    return X_scaled
