from DQN.dqn import train


conf = {
    'LOW_POWER' : 8,
    'MAX_WAIT_TIME' : 2,
    'CHARGE_SPEED' : 120,
    'NUM_MCS' : 20,
    'NUM_EV' : 500,
    'POWER_MEAN' : 40,
    'PARAMS_NUM' : 3,  # 1、随机  2、只调度MCS  3、只调度IEV 4\train
    # 'PARAMS1' : 200,
    # 'PARAMS2' : 20,
    'MEAN': 40,
    'STAND' : 16,
    'MCS_RANDOM':False,
    'IEV_RANDOM':False
}

NUM_EV = [200,300,400,500,600,700,800,900]
name = ['random', 'train_mcs', 'train_iev','train']
iftrain = [[True,True], [False,True],[True,False],[False,False]]

# for j in range(4):
#     for i in range(8):
#         conf['NUM_EV'] = NUM_EV[i]
#         conf['PARAMS1'] = NUM_EV[i]
#         conf['PARAMS_NUM'] = name[j]
#         conf['MCS_RANDOM'] = iftrain[j][0]
#         conf['IEV_RANDOM'] = iftrain[j][1]
#         train(conf)
train(conf)