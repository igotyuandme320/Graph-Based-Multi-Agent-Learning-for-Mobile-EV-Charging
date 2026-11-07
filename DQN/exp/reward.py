from DQN.dqn import train

conf = {
    'LOW_POWER' : 10,
    'MAX_WAIT_TIME' : 4,
    'CHARGE_SPEED' : 60,
    'NUM_MCS' : 30,
    'NUM_EV' : 400,
    # 'POWER_MEAN' : 50,
    'PARAMS_NUM' : 5,
    'PARAMS1' : 0,
    'PARAMS2' : 0,
    'MEAN': 60,
    'STAND' : 8
}


train(conf)