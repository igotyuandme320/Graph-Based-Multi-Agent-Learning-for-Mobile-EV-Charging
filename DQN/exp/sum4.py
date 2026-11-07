# params4
import pandas as pd

LOW_POWER = [5,6,7,8,9,10,11,12]
CHARGE_SPEED = [80,90,100,110,120]
N = len(LOW_POWER)
M = len(CHARGE_SPEED)

chargedProp = { str(LOW_POWER[i]):[] for i in range(N)}
waitTime = { str(LOW_POWER[i]):[] for i in range(N)}
extraDist = { str(LOW_POWER[i]):[] for i in range(N)}
avgExpense = { str(LOW_POWER[i]):[] for i in range(N)}
idleTime ={ str(LOW_POWER[i]):[] for i in range(N)}
mcsCost = { str(LOW_POWER[i]):[] for i in range(N)}
profit = { str(LOW_POWER[i]):[] for i in range(N)}

for i in range(N):
    for j in range(M):
        data = pd.read_csv('./results2/params4_'+ str(LOW_POWER[i]) + '_'
                           + str(CHARGE_SPEED[j]) + '.csv')

        a = data['IevChargedProp'].values[4:]
        chargedProp[str(LOW_POWER[i])].append(sum(a)/len(a))

        b = data['IevWaitTime'].values[4:]
        waitTime[str(LOW_POWER[i])].append(sum(b) / len(b))

        c = data['IevExtraDist'].values[4:]
        extraDist[str(LOW_POWER[i])].append(sum(c) / len(c))

        d = data['IevAvgExpense'].values[4:]
        avgExpense[str(LOW_POWER[i])].append((sum(d)+sum(c)) / len(d))

        e = data['mcsIdleTime'].values[4:]
        idleTime[str(LOW_POWER[i])].append(sum(e) / len(e))

        f = data['mcsCost'].values[4:]
        mcsCost[str(LOW_POWER[i])].append(sum(f) / len(f))

        g = data['mcsProfit'].values[4:]
        profit[str(LOW_POWER[i])].append(sum(g) / len(g))


pd.DataFrame(chargedProp, index=CHARGE_SPEED).to_csv('./sum_results/params4_chargedProp.csv', header=True)
pd.DataFrame(waitTime, index=CHARGE_SPEED).to_csv('./sum_results/params4_waitTime.csv', header=True)
pd.DataFrame(extraDist, index=CHARGE_SPEED).to_csv('./sum_results/params4_extraDist.csv', header=True)
pd.DataFrame(avgExpense, index=CHARGE_SPEED).to_csv('./sum_results/params4_avgExpense.csv', header=True)
pd.DataFrame(idleTime, index=CHARGE_SPEED).to_csv('./sum_results/params4_idleTime.csv', header=True)
pd.DataFrame(mcsCost, index=CHARGE_SPEED).to_csv('./sum_results/params4_mcsCost.csv', header=True)
pd.DataFrame(profit, index=CHARGE_SPEED).to_csv('./sum_results/params4_profit.csv', header=True)
