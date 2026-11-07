# params1
import pandas as pd

POWER_MEAN = [40,45,50,55,60,65,70]
POWER_STAN = [8,9,10,11,12]
N = len(POWER_MEAN)
M = len(POWER_STAN)

chargedProp = { str(POWER_MEAN[i]):[] for i in range(N)}
waitTime = { str(POWER_MEAN[i]):[] for i in range(N)}
extraDist = { str(POWER_MEAN[i]):[] for i in range(N)}
avgExpense = { str(POWER_MEAN[i]):[] for i in range(N)}
idleTime ={ str(POWER_MEAN[i]):[] for i in range(N)}
mcsCost = { str(POWER_MEAN[i]):[] for i in range(N)}
profit = { str(POWER_MEAN[i]):[] for i in range(N)}

for i in range(N):
    for j in range(M):
        data = pd.read_csv('./results2/params2_'+ str(POWER_MEAN[i]) + '_'
                           + str(POWER_STAN[j]) + '.csv')

        a = data['IevChargedProp'].values[4:]
        chargedProp[str(POWER_MEAN[i])].append(sum(a)/len(a))

        b = data['IevWaitTime'].values[4:]
        waitTime[str(POWER_MEAN[i])].append(sum(b) / len(b))

        c = data['IevExtraDist'].values[4:]
        extraDist[str(POWER_MEAN[i])].append(sum(c) / len(c))

        d = data['IevAvgExpense'].values[4:]
        avgExpense[str(POWER_MEAN[i])].append((sum(d)+sum(c)) / len(d))

        e = data['mcsIdleTime'].values[4:]
        idleTime[str(POWER_MEAN[i])].append(sum(e) / len(e))

        f = data['mcsCost'].values[4:]
        mcsCost[str(POWER_MEAN[i])].append(sum(f) / len(f))

        g = data['mcsProfit'].values[4:]
        profit[str(POWER_MEAN[i])].append(sum(g) / len(g))


pd.DataFrame(chargedProp, index=POWER_STAN).to_csv('./sum_results/params2_chargedProp.csv', header=True)
pd.DataFrame(waitTime, index=POWER_STAN).to_csv('./sum_results/params2_waitTime.csv',  header=True)
pd.DataFrame(extraDist, index=POWER_STAN).to_csv('./sum_results/params2_extraDist.csv',  header=True)
pd.DataFrame(avgExpense, index=POWER_STAN).to_csv('./sum_results/params2_avgExpense.csv',  header=True)
pd.DataFrame(idleTime, index=POWER_STAN).to_csv('./sum_results/params2_idleTime.csv',  header=True)
pd.DataFrame(mcsCost, index=POWER_STAN).to_csv('./sum_results/params2_mcsCost.csv',  header=True)
pd.DataFrame(profit, index=POWER_STAN).to_csv('./sum_results/params2_profit.csv',  header=True)
