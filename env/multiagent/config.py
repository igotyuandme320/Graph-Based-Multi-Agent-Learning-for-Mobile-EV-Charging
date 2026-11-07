from marl_ev_mcs.paths import data_path


# LOW_POWER = 10
# MAX_WAIT_TIME = 6
LOWEST_POWER = 1
# POWER_MEAN = 60
POWER_UNIT = 0.3  # 每km损耗的电量
NUM_TRAFFIC_NODES = 338
AREA = [103.9787, 30.5965, 104.1631, 30.7309]
MCS_SPEED = 11
CHARGE_PRICE = 1.6
MCS_CHARGE_PRICE = 0.5
COMM_REGION_R = 2
# CHARGE_SPEED = 60

PATH_TN_NODE = data_path("hash_point_remove.csv")
PATH_TN_ADJ = data_path("adj_remove.npy")
PATH_INIT_EV = data_path("track")


# NUM_MCS = 10
# NUM_EV = 150
#
# PARAMS_NUM = 1
# PARAMS1 = 200
# PARAMS2 = 20
