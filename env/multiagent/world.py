import numpy as np
import pandas as pd
from env.multiagent.core import Vehicle,MCS,TrafficNode,TrafficNet
from env.multiagent.utils import haversine, min_max_norm
from env.multiagent.config import *


class World(object):
    def __init__(self, conf):
        self.conf = conf
        self.EVs = []
        self.MCSs = []
        self.agents = [] # idle_MCS以及没有MCS服务的IEV构成agents
        self.last_agents = []
        self.quasi_iev = []
        self.completed_service_events = []
        self.tnet = TrafficNet()  # 路网
        self.MCS_init_pos = [np.random.uniform(self.tnet.min_lat, self.tnet.max_lat, self.conf['NUM_MCS']),
                             np.random.uniform(self.tnet.min_lon, self.tnet.max_lon, self.conf['NUM_MCS'])]

        # 整理数据时可以把下面两个文件合为一个文件
        self.ev_init_data = None  # 电车的初始化数据，格式为[id,remain,pos]
        self.set_data()

        self.tn_data = pd.read_csv(PATH_TN_NODE)
        self.init_world()


    def set_data(self):
        # 电车的初始化数据，格式为[id,remain,pos]
        date = self.conf.get('TRACK_DAY', 5)
        # date = random.randint(3, 7)
        self.ev_init_data = pd.read_csv(PATH_INIT_EV / f'2014080{date}.csv')

    def init_world(self):
        # 读路网数据
        for index,row in self.tn_data.iterrows():
            node = TrafficNode(row['id'], [row['lng'],row['lat']])
            self.tnet.nodes.append(node)

        # MCS的初始化，均匀分布
        # init_mcs_pos = [np.random.uniform(self.tnet.min_lat, self.tnet.max_lat, NUM_MCS),
        #                  np.random.uniform(self.tnet.min_lon, self.tnet.max_lon, NUM_MCS)]
        for i in range(self.conf['NUM_MCS']):
            pos = [self.MCS_init_pos[1][i], self.MCS_init_pos[0][i]]
            mcs = MCS(i, pos, self.conf)
            self.MCSs.append(mcs)

        # ev的初始化
        remain = np.random.normal(loc=self.conf['MEAN'], scale=self.conf['STAND'], size=self.conf['NUM_EV'])
        for index, row in self.ev_init_data.iterrows():

            if index >= self.conf['NUM_EV']:
                break
            ev = Vehicle(row['id'], remain[index], row['distance'],[row['lng'], row['lat']], self.conf)
            # ev = Vehicle(row['id'], row['remain'], row['distance'], [row['lng'], row['lat']], self.conf)
            tracks = row['track'].split(',')
            ev.track = [track.split(' ') for track in tracks]
            ev.track_index = 0
            ev.destination = ev.track[-1]
            ev.set_charge()
            self.EVs.append(ev)
        self.ev_init_data = None  # 每个ev的轨迹赋值之后就释放内存
        self.match_and_get_neibor()
        # self.agents = []

    def step_finish(self):
        self.last_agents = self.agents
        self.agents = []
        self.quasi_iev = []
        for mcs in self.MCSs:
            mcs.step_finish()
        for ev in self.EVs:
            ev.step_finish()

    def begin_update(self):
        """Clear the one-step service-completion event queue."""
        self.completed_service_events = []

    def record_completed_service(self, mcs):
        """Finish one MCS service and expose its ownership to the runner."""
        revenue = mcs.finish_charge()
        self.completed_service_events.append((mcs.ID, revenue))
        return revenue

    def reset_mcs(self):
        for i in range(len(self.MCSs)):
            pos = [self.MCS_init_pos[1][i], self.MCS_init_pos[0][i]]
            self.MCSs[i].reset(pos)

    def reset_vehicle(self):
        self.set_data()
        remain = np.random.normal(loc=self.conf['MEAN'], scale=self.conf['STAND'], size=self.conf['NUM_EV'])
        for index,row in self.ev_init_data.iterrows():
            if index >= self.conf['NUM_EV']:
                break
            self.EVs[index].reset(remain[index], row['distance'], [row['lng'], row['lat']])
            # self.EVs[index].reset(row['remain'],row['distance'], [row['lng'], row['lat']])
            tracks = row['track'].split(',')
            self.EVs[index].track = [track.split( ) for track in tracks]  # 从ev_data读取每量ev的轨迹
            self.EVs[index].track_index = 0
        self.ev_init_data = None

    def reset_world(self):
        self.agents = []
        self.last_agents = []
        self.quasi_iev = []
        self.completed_service_events = []
        self.reset_mcs()
        self.reset_vehicle()

    def get_agent_obs(self, agent):
        obs = {
            'feat': [],
            'pos': [],
            'type': 'MCS' if isinstance(agent, MCS) else 'IEV',
            'id': agent.ID
        }
        if isinstance(agent, MCS):
            for point in agent.near_point:
                dist_agent = haversine(agent.pos[0], agent.pos[1], point.pos[0], point.pos[1])
                Mcs = []
                quasiIev = []

                for task_mcs in agent.near_task_MCS:
                    if task_mcs.charge_time < 10:
                        dist_task_mcs = haversine(point.pos[0], point.pos[1], task_mcs.pos[0],
                                                  task_mcs.pos[1])
                        dist = dist_task_mcs + (task_mcs.charge_time * 60 * MCS_SPEED)/1000
                        Mcs.append(dist)

                for idle_mcs in agent.near_idle_MCS:
                    dist = haversine(point.pos[0], point.pos[1], idle_mcs.pos[0], idle_mcs.pos[1])
                    Mcs.append(dist)

                for quasi_iev in agent.near_quasi_IEV:
                    for tracking_point in quasi_iev.near_point:
                        if tracking_point is point:
                            dist = haversine(point.pos[0], point.pos[1], quasi_iev.pos[0], quasi_iev.pos[1])
                            # remain = quasi_iev.remain - dist*POWER_UNIT
                            num = max(quasi_iev.remain-self.conf['LOW_POWER'], 0) * MCS_SPEED /1000 + dist
                            quasiIev.append(quasi_iev.need_power/(len(quasi_iev.near_point)*(num+1)))
                            break

                idleMcs_avg_res = sum(Mcs)
                quasiIev_avg_res = sum(quasiIev)

                if len(Mcs) > 1 and (sum(Mcs) != 0):
                    idleMcs_avg_res = 0
                    dist_sum = sum(Mcs)
                    for dist in Mcs:
                        idleMcs_avg_res += dist * (dist_sum-dist)/dist_sum

                obs['feat'].append([idleMcs_avg_res, quasiIev_avg_res])
                obs['pos'].append(point.pos)

        # 是IEV
        else:
            for point in agent.near_point:
                dist_agent = haversine(agent.pos[0], agent.pos[1], point.pos[0], point.pos[1])
                taskMcs = []
                otherIev = []
                for task_mcs in agent.near_task_MCS:

                    dist_task_mcs = haversine(task_mcs.pos[0], task_mcs.pos[1], task_mcs.charge_pos[0],
                                              task_mcs.charge_pos[1])

                    remain = task_mcs.remain - task_mcs.charge_power
                    if remain >= min(agent.need_power, self.conf['CHARGE_SPEED']/3) and task_mcs.charge_time<10:
                        # 充电地点与路口的距离
                        dist = haversine(point.pos[0], point.pos[1], task_mcs.charge_pos[0], task_mcs.charge_pos[1])

                        if task_mcs.is_arrive:
                            dist += (task_mcs.charge_time * 60 * MCS_SPEED)/1000
                        else:
                            dist += dist_task_mcs + (task_mcs.charge_time * 60 * MCS_SPEED)/1000

                        taskMcs.append(remain/(dist+1))

                for other_iev in agent.near_IEV:
                    dist = haversine(point.pos[0], point.pos[1], other_iev.pos[0], other_iev.pos[1])
                    otherIev.append(dist)

                taskMcs_avg_res = sum (taskMcs)
                otherIev_avg_res = sum(otherIev)

                if len(otherIev) > 1 and (sum(otherIev) != 0):
                    otherIev_avg_res = 0
                    dist_sum = sum(otherIev)
                    for dist in otherIev:
                        otherIev_avg_res += dist * (dist_sum - dist) / dist_sum

                obs['feat'].append([taskMcs_avg_res, otherIev_avg_res])
                obs['pos'].append(point.pos)
        return obs

    # 获取IEV和所有MCS的局部观测，IEV格式为:[{'taskMcs':[[dist,time,remain]...], 'iev':[[dist, shortage]...]}...]
    # MCS格式为:[{'taskMcs':[[dist,time,remain]...], 'idleMcs':[[dist,remain]...], 'quasi-iev':[[dist, shortage]...]}...]
    def get_obs_n(self):
        new_obs_n = []
        old_obs_n = []
        for agent in self.agents:
            new_obs_n.append(self.get_agent_obs(agent))

        for agent in self.last_agents:
            old_obs_n.append(self.get_agent_obs(agent))

        return new_obs_n, old_obs_n

    # def get_reward_n(self):
    #     reward_n = []
    #     for agent in self.last_agents:
    #         reward = 0
    #         if isinstance(agent, MCS):
    #
    #             for quasi_iev in agent.near_quasi_IEV:
    #                 dist = haversine(quasi_iev.pos[0], quasi_iev.pos[1], agent.pos[0], agent.pos[1])
    #                 if dist != 0:
    #                     # reward += quasi_iev.need_power / (dist + (quasi_iev.remain-10)/POWER_UNIT)
    #                     reward += quasi_iev.need_power / (dist + 1 +  (quasi_iev.remain - self.conf['LOW_POWER']) / POWER_UNIT)
    #                 # 加上周围其他mcs的影响
    #
    #                 for idle_mcs in quasi_iev.near_idle_MCS:
    #                     dist_i = haversine(idle_mcs.pos[0], idle_mcs.pos[1], quasi_iev.pos[0], quasi_iev.pos[1])
    #                     if dist_i <= dist and idle_mcs.remain >= quasi_iev.need_power:
    #                         reward -= (quasi_iev.need_power / (dist_i + 1+ (quasi_iev.remain - self.conf['LOW_POWER']) / POWER_UNIT))
    #
    #                 for task_mcs in agent.near_task_MCS:
    #                     dist_t = haversine(task_mcs.pos[0], task_mcs.pos[1], agent.pos[0], agent.pos[1])
    #                     if task_mcs.charge_time < 5 and dist_t <= dist:
    #                         reward -= (task_mcs.remain-task_mcs.charge_power) / (dist_t + (task_mcs.charge_time * 60 * MCS_SPEED) / 1000 + 1)
    #
    #         else:
    #             for task_mcs in agent.near_task_MCS:
    #                 if task_mcs.remain - task_mcs.charge_power > min(agent.need_power, self.conf['CHARGE_SPEED']/3) and task_mcs.charge_time<=10:
    #                     dist = haversine(task_mcs.pos[0], task_mcs.pos[1], agent.pos[0], agent.pos[1])
    #                     dist_m = haversine(task_mcs.pos[0], task_mcs.pos[1], task_mcs.charge_pos[0], task_mcs.charge_pos[1])
    #                     reward += (task_mcs.remain - task_mcs.charge_power) / ((dist+1) if task_mcs.charge_time <= 5 else (
    #                                     dist + 1 + (task_mcs.charge_time - 5) * 60 * MCS_SPEED / 1000) + dist_m)
    #
    #                     for other_iev in task_mcs.near_IEV:
    #                         if task_mcs.remain - task_mcs.charge_power > min(other_iev.need_power, self.conf['CHARGE_SPEED']/3):
    #                             dist_o = haversine(other_iev.pos[0], other_iev.pos[1], task_mcs.pos[0], task_mcs.pos[1])
    #                             if dist_o <= dist:
    #                                 reward -= (task_mcs.remain - task_mcs.charge_power) / ((dist_o+1) if task_mcs.charge_time <= 5 else (
    #                                     dist_o + 1+ (task_mcs.charge_time - 5) * 60 * MCS_SPEED / 1000) + dist_m)
    #
    #         reward_n.append(reward)
    #
    #     reward_n = min_max_norm(reward_n)
    #
    #     for index, agent in enumerate(self.last_agents):
    #         agent.total_reward += reward_n[index]
    #
    #     return reward_n

    def get_reward_n(self):
        reward_n = []
        for agent in self.last_agents:
            reward = 0
            last_dist = haversine(agent.pos[0], agent.pos[1], agent.last_pos[0], agent.last_pos[1])
            if isinstance(agent, MCS):
                reward -= self.conf.get('MCS_MOVE_COST_PENALTY', 0) * agent.last_move_cost

                for quasi_iev in agent.near_quasi_IEV:
                    dist = haversine(quasi_iev.pos[0], quasi_iev.pos[1], agent.pos[0], agent.pos[1])
                    if min(quasi_iev.need_power, self.conf['CHARGE_SPEED']/3) <= agent.remain:
                        reward += min(quasi_iev.need_power, self.conf['CHARGE_SPEED']/3)/(dist+1)

                        for idle_mcs in quasi_iev.near_idle_MCS:
                            if idle_mcs is not agent:
                                dist_i = haversine(idle_mcs.pos[0], idle_mcs.pos[1], quasi_iev.pos[0], quasi_iev.pos[1])
                                if idle_mcs.remain >= min(quasi_iev.need_power, self.conf['CHARGE_SPEED']/3):
                                    reward -= 0.1 * (idle_mcs.remain) / (dist_i+1)

                        for task_mcs in quasi_iev.near_task_MCS:
                            dist_t = haversine(task_mcs.pos[0], task_mcs.pos[1], quasi_iev.pos[0], quasi_iev.pos[1])
                            if task_mcs.remain >= min(quasi_iev.need_power, self.conf['CHARGE_SPEED'] / 3) and task_mcs.charge_time <= 5:
                                remain_power = task_mcs.remain - task_mcs.charge_power
                                reward -= 0.1 * (remain_power) / (dist_t + 1)


            else:
                reward -= self.conf.get('IEV_WAIT_PENALTY', 0) * 5
                for task_mcs in agent.near_task_MCS:
                    dist_m = haversine(task_mcs.pos[0], task_mcs.pos[1], task_mcs.charge_pos[0], task_mcs.charge_pos[1])
                    remain_power = task_mcs.remain - task_mcs.charge_power - dist_m*POWER_UNIT
                    if remain_power > min(agent.need_power, self.conf['CHARGE_SPEED']/3) \
                            and agent.total_wait_time + task_mcs.charge_time <= self.conf['MAX_WAIT_TIME']*5:

                        dist = haversine(task_mcs.charge_pos[0], task_mcs.charge_pos[1], agent.pos[0], agent.pos[1])
                        reward += remain_power/(dist+1)

                        for other_iev in task_mcs.near_IEV:
                            if other_iev is not agent:
                                dist_o = haversine(other_iev.pos[0], other_iev.pos[1], task_mcs.pos[0], task_mcs.pos[1])
                                reward -= 0.5*other_iev.need_power/(dist_o+1)


            # reward -= POWER_UNIT*last_dist
            reward_n.append(reward)

        reward_n = min_max_norm(reward_n)

        for index, agent in enumerate(self.last_agents):
            agent.total_reward += reward_n[index]

        return reward_n

    def update(self, action_n):
        self.begin_update()
        # idle_MCS和IEV更新
        for i in range(len(self.agents)):
            agent = self.agents[i]
            agent.last_pos = agent.pos
            # 行驶的曼哈顿距离
            dist = haversine(action_n[i][0], action_n[i][1], action_n[i][0], agent.pos[1]) + \
                   haversine(action_n[i][0], agent.pos[1], agent.pos[0], agent.pos[1])

            if isinstance(agent, MCS):
                # MCS位置、剩余电量
                agent.pos = action_n[i]
                agent.remain -= dist*POWER_UNIT
                if agent.remain < 0:
                    agent.remain += 200
                agent.total_idle_time += 5
                agent.last_move_cost = dist * POWER_UNIT * MCS_CHARGE_PRICE
                agent.cost += agent.last_move_cost

            else:
                agent.pos = action_n[i]
                agent.remain -= dist * POWER_UNIT
                agent.set_charge()
                agent.total_wait_time += 5
                agent.total_extra_dist += dist

        # task_MCS位置、剩余电量、剩余充电时间等更新
        for mcs in self.MCSs:
            if mcs.is_idle is False and mcs.is_arrive is False: # 任务中的MCS还未到达充电地点
                ability = 300 * MCS_SPEED / 1000
                dist = haversine(mcs.pos[0],mcs.pos[1],mcs.charge_pos[0],mcs.charge_pos[1])
                if ability < dist:  # 无法到达充电地点
                    mcs.pos[0] += (ability / dist) * (mcs.charge_pos[0] - mcs.pos[0])
                    mcs.pos[1] += (ability / dist) * (mcs.charge_pos[1] - mcs.pos[1])
                else:  # 能够达到充电地点
                    mcs.pos = mcs.charge_pos
                    mcs.is_arrive = True
                    charge_time = max(0, (300-(dist*1000)/MCS_SPEED)/60)
                    if charge_time > mcs.charge_time: # 电车能够在这个时间步完成充电
                        self.record_completed_service(mcs)
                    else:
                        mcs.charge_time -= charge_time
            elif mcs.is_idle is False and mcs.is_arrive is True:  # 在充电地点的任务中的MCS
                if mcs.charge_time < 5:
                    self.record_completed_service(mcs)
                else:
                    mcs.charge_time -= 5

        # 其他ev位置、剩余电量更新
        for ev in self.EVs:   # 不会缺电的电车不用考虑
            if ev.need_charge is False and ev.need_power > 0 and ev.is_charged is False:  # 没有发出充电请求的车，按轨迹行驶
                last_pos = ev.pos
                ev.track_index += 1
                ev.pos = ev.track[ev.track_index]
                dist = haversine(ev.pos[0], ev.pos[1], last_pos[0], last_pos[1])
                ev.remain -= dist*POWER_UNIT
                ev.set_charge()

            elif ev.is_charged is True:  # 不用再考虑这辆电车了，不用更新了
                pass

    def is_in_area(self, pos):
        if 104.1631 > float(pos[0]) > 103.9787 and 30.7309 > float(pos[1]) > 30.5965:
            return True
        else:
            return False

    def match_and_get_neibor(self):
        for ev in self.EVs:
            if self.is_in_area(ev.pos):
                if ev.need_charge is False and ev.need_power > 0 and ev.is_charged is False:
                    self.quasi_iev.append(ev)

                if (ev.need_charge and ev.is_charged is False) or (ev.need_power > 0 and ev.is_charged is False) \
                        or (ev in self.last_agents):  # IEV+quasi_IEV与MCS相互添加邻居信息并匹配

                    for mcs in self.MCSs:
                        # 周围的task_MCS
                        dist = haversine(mcs.pos[0], mcs.pos[1], ev.pos[0], ev.pos[1])
                        if dist < COMM_REGION_R:
                            if mcs.is_idle is False:  # task_mcs
                                ev.near_task_MCS.append(mcs)

                                if ev.need_charge and ev.is_charged is False:
                                    mcs.near_IEV.append(ev)
                            else:   # 有空闲MCS
                                if ev.need_charge and mcs.remain > min(ev.need_power, self.conf['CHARGE_SPEED']/3) and ev.is_charged is False:  # 是发出充电请求的IEV
                                    ev.is_charged = True   # 此IEV一定有MCS服务
                                    ev.assigned_charge = True
                                    ev.expense = min(ev.need_power, self.conf['CHARGE_SPEED']/3) * CHARGE_PRICE

                                    if dist < ev.nearest_mcs_dist:  # 此idle_MCS距离最近
                                        ev.nearest_mcs_dist = dist
                                        ev.set_new_mcs(mcs)
                                        mcs.set_new_iev(ev)

                                if ev.need_charge is False and ev.need_power > 0:  # 没有发出充电请求的EV
                                    mcs.near_quasi_IEV.append(ev)
                                    ev.near_idle_MCS.append(mcs)

        for i in range(len(self.EVs)):
            # 匹配后，剩余的IEV与IEV相互添加邻居信息
            if self.EVs[i].need_charge and self.EVs[i].is_charged is False and self.is_in_area(self.EVs[i].pos):
                self.agents.append(self.EVs[i])
                for j in range(i + 1, len(self.EVs)):
                    if self.is_in_area(self.EVs[j].pos):
                        if self.EVs[j].need_charge and self.EVs[j].is_charged is False and \
                                haversine(self.EVs[j].pos[0], self.EVs[j].pos[1], self.EVs[i].pos[0], self.EVs[i].pos[1]) < COMM_REGION_R:
                            self.EVs[i].near_IEV.append(self.EVs[j])
                            self.EVs[j].near_IEV.append(self.EVs[i])
                        elif self.EVs[j].need_charge is False and self.EVs[j].need_power > 0 and \
                                haversine(self.EVs[j].pos[0], self.EVs[j].pos[1], self.EVs[i].pos[0], self.EVs[i].pos[1]) < COMM_REGION_R:
                            self.EVs[i].near_quasi_IEV.append(self.EVs[j])
                            self.EVs[j].near_quasi_IEV.append(self.EVs[i])

            # 上一批IEV添加邻居信息
            elif self.EVs[i] in self.last_agents and self.is_in_area(self.EVs[i].pos):
                for j in range(i + 1, len(self.EVs)):
                    if self.is_in_area(self.EVs[j].pos):
                        dist = haversine(self.EVs[j].pos[0], self.EVs[j].pos[1], self.EVs[i].pos[0], self.EVs[i].pos[1])
                        if self.EVs[j].need_charge and self.EVs[j].is_charged is False and dist < COMM_REGION_R:
                            self.EVs[i].near_IEV.append(self.EVs[j])
                            self.EVs[j].near_IEV.append(self.EVs[i])
                        elif self.EVs[j].need_charge is False and self.EVs[j].need_power > 0 and dist < COMM_REGION_R:
                            self.EVs[i].near_quasi_IEV.append(self.EVs[j])
                            self.EVs[j].near_quasi_IEV.append(self.EVs[i])

        # 获取MCS相互的邻居关系
        for i in range(len(self.MCSs)):
            if self.MCSs[i].is_idle:
                self.agents.append(self.MCSs[i])
            for j in range(i + 1, len(self.MCSs)):
                if haversine(self.MCSs[i].pos[0], self.MCSs[i].pos[1], self.MCSs[j].pos[0], self.MCSs[j].pos[1]) < COMM_REGION_R:
                    if self.MCSs[j].is_idle:
                        self.MCSs[i].near_idle_MCS.append(self.MCSs[j])
                    else:
                        self.MCSs[i].near_task_MCS.append(self.MCSs[j])
                    if self.MCSs[i].is_idle:
                        self.MCSs[j].near_idle_MCS.append(self.MCSs[i])
                    else:
                        self.MCSs[j].near_task_MCS.append(self.MCSs[i])

    # 获取所有agent周围的路口
    def get_neibor_point(self):
        for agent in self.agents:
            for point in self.tnet.nodes:
                dist = haversine(agent.pos[0],agent.pos[1],point.pos[0],point.pos[1])
                if dist < COMM_REGION_R:
                    agent.near_point.append(point)

        for agent in self.last_agents:
            for point in self.tnet.nodes:
                dist = haversine(agent.pos[0],agent.pos[1],point.pos[0],point.pos[1])
                if dist < COMM_REGION_R:
                    agent.near_point.append(point)

        for ev in self.quasi_iev:
            for point in self.tnet.nodes:
                dist = haversine(ev.pos[0],ev.pos[1],point.pos[0],point.pos[1])
                if dist < COMM_REGION_R:
                    ev.near_point.append(point)
