# EV发出充电请求的剩余电量阈值+充电速度

from DQN.dqn import train

conf = {
    'LOW_POWER' : 8,
    'MAX_WAIT_TIME' : 2,
    'CHARGE_SPEED' : 120,
    'NUM_MCS' : 20,
    'NUM_EV' : 500,
    'POWER_MEAN' : 60,
    'PARAMS_NUM' : 4,
    'PARAMS1' : 5,
    'PARAMS2' : 80,
    'MEAN': 60,
    'STAND' : 8
}

LOW_POWER = [5,6,7,8,9,10,11,12]
CHARGE_SPEED = [80,90,100,110,120]

for i in range(8):
    for j in range(5):
        conf['LOW_POWER'] = LOW_POWER[i]
        conf['CHARGE_SPEED'] = CHARGE_SPEED[j]
        conf['PARAMS1'] = LOW_POWER[i]
        conf['PARAMS2'] = CHARGE_SPEED[j]
        train(conf)