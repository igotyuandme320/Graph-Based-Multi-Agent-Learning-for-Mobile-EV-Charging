# MCS数量+EV最大充电等待时间阈值

from DQN.dqn import train

conf = {
    'LOW_POWER' : 8,
    'MAX_WAIT_TIME' : 2,
    'CHARGE_SPEED' : 120,
    'NUM_MCS' : 20,
    'NUM_EV' : 500,
    'POWER_MEAN' : 40,
    'PARAMS_NUM' : 3,
    'PARAMS1' : 1,
    'PARAMS2' : 20,
    'MEAN': 40,
    'STAND' : 16
}

MAX_WAIT_TIME = [1,2,3,4,5]
NUM_MCS = [10,20,30,40,50]

for i in range(4,5):
    for j in range(3,5):
        conf['MAX_WAIT_TIME'] = MAX_WAIT_TIME[i]
        conf['NUM_MCS'] = NUM_MCS[j]
        conf['PARAMS1'] = MAX_WAIT_TIME[i]
        conf['PARAMS2'] = NUM_MCS[j]
        train(conf)