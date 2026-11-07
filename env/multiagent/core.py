import random

import numpy as np
from env.multiagent.config import *
from env.multiagent.utils import haversine, min_max_norm
import pandas as pd
import geohash
import torch
# from sklearn.cluster import KMeans


# 电车
class Vehicle(object):
    def __init__(self, id, remain, distance, pos, conf):
        self.conf = conf
        self.ID = id  # 车辆编号
        self.remain = remain  # 车辆剩余电量，单位kwh
        self.pos = pos   # 车辆当前位置
        self.total_distance = distance
        self.track = None  # 车辆轨迹，从历史数据中获取
        self.track_index = 0  # 车辆位于轨迹的哪一步
        self.destination = []  # 车辆终点经纬度
        self.wait_time = 0  # 充电等待时间
        self.last_dist = 0
        self.last_pos = []

        self.need_charge = False  # 是否发出充电请求
        # Keep outcome states separate: a vehicle can be assigned to an MCS
        # long before the charging service actually completes.
        self.requested_charge = False
        self.assigned_charge = False
        self.charge_completed = False
        self.need_power = max(distance * POWER_UNIT - remain, 0) # 电车到达终点的缺电量
        self.set_charge()

        self.is_charged = False  # 是否有充电桩服务
        self.nearest_mcs_dist = 99
        self.charge_pos = None
        self.charge_mcs = None

        self.near_task_MCS = []
        self.near_IEV = []
        self.near_quasi_IEV = []
        self.near_point = []
        self.near_idle_MCS = []
        self.reward = 0

        self.total_wait_time = 0
        self.total_reward = 0
        self.total_extra_dist = 0

        self.action_next = None
        self.fail_charge = False

        self.expense = 0


    def set_charge(self):
        if self.remain < self.conf['LOW_POWER']:
            if self.need_charge:  # 如果电车原来就在需要充电的状态
                self.wait_time += 1  # 充电等待时间+1
                if self.wait_time > self.conf['MAX_WAIT_TIME']*5 or self.remain < LOWEST_POWER:
                    self.remain = self.conf['MEAN']  # 假设超过最大等待时间后以其他方式充上了电
                    self.need_charge = False
                    self.need_power = 0
                    self.fail_charge = True
            else: # 如果电车刚转入需要充电的状态
                self.need_charge = True
                self.requested_charge = True
                self.wait_time = 0
        else:
            self.need_charge = False
            self.wait_time = 0

    def set_new_mcs(self, mcs):  # 为IEV设置充电的MCS
        self.charge_pos = [(float(self.pos[0]) + float(mcs.pos[0])) / 2,
                           (float(self.pos[1]) + float(mcs.pos[1])) / 2]
        charge_dist = haversine(self.charge_pos[0],self.charge_pos[1],self.charge_pos[0],self.pos[1]) + \
                          haversine(self.charge_pos[0],self.pos[1], self.pos[0],self.pos[1])

        if self.charge_mcs is not None:
            self.charge_mcs.is_idle = True
            self.charge_mcs.is_arrive = False
            self.charge_mcs.charge_ev = None
            self.charge_mcs.charge_pos = None  # 若MCS有任务在身，则设置充电地点和充电时间
            self.charge_mcs.charge_time = None
            self.charge_mcs.charge_power = None
            self.charge_mcs.reward = 0
            self.charge_mcs.cost -= self.charge_mcs.charge_dist*POWER_UNIT*MCS_CHARGE_PRICE
            self.charge_mcs.charge_dist = 0

        mcs.cost += charge_dist*POWER_UNIT*MCS_CHARGE_PRICE
        mcs.charge_dist = charge_dist
        self.charge_mcs = mcs
        # self.reward = 1
        self.action_next = self.charge_pos

    def reset(self, remain, distance, pos):
        self.remain = remain
        self.pos = pos
        self.total_distance = distance
        self.need_power = max(distance * POWER_UNIT - remain, 0)
        self.track_index = 0
        self.wait_time = 0  # 充电等待时间
        self.need_charge = False  # 是否发出充电请求
        self.requested_charge = False
        self.assigned_charge = False
        self.charge_completed = False
        self.is_charged = False  # 是否有充电桩服务
        self.last_dist = 0
        self.last_pos = []
        self.fail_charge = False

        self.near_task_MCS = []
        self.near_IEV = []
        self.near_quasi_IEV = []
        self.near_idle_MCS = []
        self.near_point = []
        self.nearest_mcs_dist = 99
        self.set_charge()

        self.track = None  # 车辆轨迹，从历史数据中获取
        self.track_index = 0  # 车辆位于轨迹的哪一步

        self.charge_pos = None
        self.charge_mcs = None

        self.near_task_MCS = []
        self.near_IEV = []
        self.near_point = []
        self.reward = 0

        self.total_wait_time = 0
        self.total_reward = 0
        self.total_extra_dist = 0

        self.action_next = None
        self.expense = 0

    def step_finish(self):
        self.near_task_MCS = []
        self.near_IEV = []
        self.near_point = []
        self.near_quasi_IEV = []
        self.near_idle_MCS = []
        self.nearest_mcs_dist = 99
        # self.reward = 0
        # self.last_dist = 0

# 移动充电桩，假设为一直电量充足的状态
class MCS(object):
    def __init__(self, id, pos, conf):
        self.conf = conf
        self.ID = id
        self.is_idle = True
        self.pos = pos  # 当前位置
        self.remain = 300

        self.charge_ev = None
        self.charge_pos = None  # 若MCS有任务在身，则设置充电地点和充电时间
        self.is_arrive = False
        self.charge_time = None  # 时间单位为分钟
        self.charge_power = None
        self.last_dist = 0

        self.total_profit = 0  # 一天总的充电收益
        self.total_reward = 0
        self.total_idle_time = 0  # 一天总的空闲巡航时间
        # self.cur_profit = 0  # 本次充电收益
        # self.cur_dist = 0 # 本次移动的距离

        self.near_task_MCS = []
        self.near_idle_MCS = []
        self.near_quasi_IEV = []
        self.near_IEV = []
        self.near_point = []

        self.reward = 0
        self.cost = 0
        self.charge_dist = 0
        self.last_move_cost = 0
        self.completed_service_count = 0

    def reset(self, pos):
        self.is_idle = True
        self.pos = pos
        self.remain = 300

        self.charge_ev = None
        self.charge_pos = None  # 若MCS有任务在身，则设置充电地点和充电时间
        self.is_arrive = False
        self.charge_time = None  # 时间单位为分钟
        self.charge_power = None

        self.total_profit = 0  # 一天总的充电收益
        self.total_reward = 0
        self.total_idle_time = 0  # 一天总的空闲巡航时间

        self.near_task_MCS = []
        self.near_idle_MCS = []
        self.near_quasi_IEV = []
        self.near_IEV = []
        self.near_point = []

        self.reward = 0
        self.last_dist = 0
        self.cost = 0
        self.charge_dist = 0
        self.last_move_cost = 0
        self.completed_service_count = 0

    def set_new_iev(self, iev):
        self.is_idle = False
        self.charge_ev = iev
        self.charge_pos = iev.charge_pos

        time = (iev.need_power / self.conf['CHARGE_SPEED']) * 60  # 充电多少分钟
        if time > 20:
            self.charge_time = 20
        else:
            self.charge_time = time

        self.charge_power = (self.charge_time/60) * self.conf['CHARGE_SPEED']

    def finish_charge(self):
        served_ev = self.charge_ev
        self.is_arrive = False
        self.is_idle = True
        self.charge_time = None
        self.charge_pos = None
        self.charge_ev = None

        revenue = self.charge_power*CHARGE_PRICE
        self.total_profit += revenue
        self.completed_service_count += 1
        if served_ev is not None:
            served_ev.charge_completed = True
            served_ev.need_charge = False
        self.remain -= self.charge_power
        if self.remain < 10:
            self.remain = 300

        self.charge_power = None
        self.charge_dist = 0
        return revenue

    def step_finish(self):
        self.near_task_MCS = []
        self.near_idle_MCS = []
        self.near_quasi_IEV = []
        self.near_IEV = []
        self.near_point = []


# 路网中的调度点
class TrafficNode(object):
    def __init__(self, id, pos):
        self.ID = id
        self.pos = pos


# 路网
class TrafficNet(object):
    def __init__(self):
        self.n = NUM_TRAFFIC_NODES
        self.nodes = []  # 调度点
        self.max_lat = AREA[3]
        self.max_lon = AREA[2]
        self.min_lat = AREA[1]
        self.min_lon = AREA[0]




