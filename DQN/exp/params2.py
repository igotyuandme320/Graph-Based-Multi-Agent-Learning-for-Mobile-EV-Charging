# EV初始电量正态分布均值+方差
# 需要额外处理数据

from DQN.dqn import train

conf = {
    'LOW_POWER' : 8,
    'MAX_WAIT_TIME' : 2,
    'CHARGE_SPEED' : 120,
    'NUM_MCS' : 20,
    'NUM_EV' : 500,
    # 'POWER_MEAN' : 60,
    'PARAMS_NUM' : 2,
    'PARAMS1' : 25,
    'PARAMS2' : 8,
    'MEAN': 40,
    'STAND' : 8
}

POWER_MEAN = [25,30,35,40,45,50,55]
POWER_STAN = [8,10,12,14,16]


# 每次读的数据不一样 额外写
for i in range(7):
    for j in range(5):
        conf['MEAN'] = POWER_MEAN[i]
        conf['STAND'] = POWER_STAN[j]
        conf['PARAMS1'] = POWER_MEAN[i]
        conf['PARAMS2'] = POWER_STAN[j]
        train(conf)