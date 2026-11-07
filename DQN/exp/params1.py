from DQN.dqn import train

conf = {
    'LOW_POWER' : 8,
    'MAX_WAIT_TIME' : 2,
    'CHARGE_SPEED' : 120,
    'NUM_MCS' : 20,
    'NUM_EV' : 200,
    'POWER_MEAN' : 60,
    'PARAMS_NUM' : 1,
    'PARAMS1' : 200,
    'PARAMS2' : 20,
    'MEAN': 40,
    'STAND': 16,
    'MCS_RANDOM': False,
    'IEV_RANDOM': False
}

NUM_EV = [200,300,400,500,600,700,800,900]
NUM_MCS = [10,20,30,40,50]

# for i in range(8):
#     for j in range(5):
#         conf['NUM_EV'] = NUM_EV[i]
#         conf['NUM_MCS'] = NUM_MCS[j]
#         conf['PARAMS1'] = NUM_EV[i]
#         conf['PARAMS2'] = NUM_MCS[j]
#         train(conf)

conf['NUM_EV'] = 500
conf['NUM_MCS'] = 50
conf['PARAMS1'] = 500
conf['PARAMS2'] = 50
train(conf)