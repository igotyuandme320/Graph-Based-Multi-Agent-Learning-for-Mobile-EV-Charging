
class MultiAgentEnv():
    def __init__(self, world):
        self.world = world
        # self.agents = self.world.agents
        # self.cur_step = 0  # 当前执行步数，用于更新日期

    def step(self, action_n):
        # 1、所有成员位置更新，状态更新
        self.world.update(action_n)
        self.world.step_finish()
        # 2、匹配，获取邻居关系
        self.world.match_and_get_neibor()
        # 3、剩余的IEV，空闲MCS，获取周围待选路口
        self.world.get_neibor_point()
        # 4、获取局部观测以及奖励
        new_obs_n, old_obs_n = self.world.get_obs_n()
        reward_n = self.world.get_reward_n()

        return new_obs_n, old_obs_n, reward_n

    def reset(self):
        self.world.reset_world()
        # self.cur_step = 0
        self.world.match_and_get_neibor()
        self.world.get_neibor_point()
        new_obs_n, old_obs_n= self.world.get_obs_n()

        return new_obs_n

    def render(self):
        pass




