# params3
import pandas as pd

MAX_WAIT_TIME = [1,2,3,4,5]
NUM_MCS = [10,20,30,40,50]

N = len(MAX_WAIT_TIME)
M = len(NUM_MCS)

chargedProp = { str(MAX_WAIT_TIME[i]):[] for i in range(N)}
waitTime = { str(MAX_WAIT_TIME[i]):[] for i in range(N)}
extraDist = { str(MAX_WAIT_TIME[i]):[] for i in range(N)}
# avgExpense = { str(MAX_WAIT_TIME[i]):[] for i in range(N)}
idleTime ={ str(MAX_WAIT_TIME[i]):[] for i in range(N)}
mcsCost = { str(MAX_WAIT_TIME[i]):[] for i in range(N)}
profit = { str(MAX_WAIT_TIME[i]):[] for i in range(N)}

for i in range(N):
    for j in range(M):
        data = pd.read_csv('./results3/params3_'+ str(MAX_WAIT_TIME[i]) + '_'
                           + str(NUM_MCS[j]) + "_19"+ '.csv')

        a = data['IevChargedProp'].values[8:]
        chargedProp[str(MAX_WAIT_TIME[i])].append(sum(a)/len(a))

        b = data['IevWaitTime'].values[8:]
        waitTime[str(MAX_WAIT_TIME[i])].append(sum(b) / len(b))

        c = data['IevExtraDist'].values[8:]
        extraDist[str(MAX_WAIT_TIME[i])].append(sum(c) / len(c))

        # d = data['IevAvgExpense'].values[8:]
        # avgExpense[str(MAX_WAIT_TIME[i])].append((sum(d)+sum(c)) / len(d))

        e = data['mcsIdleTime'].values[8:]
        idleTime[str(MAX_WAIT_TIME[i])].append(sum(e) / len(e))

        f = data['mcsCost'].values[8:]
        mcsCost[str(MAX_WAIT_TIME[i])].append(sum(f) / len(f))

        g = data['mcsProfit'].values[8:]
        profit[str(MAX_WAIT_TIME[i])].append(sum(g) / len(g))


pd.DataFrame(chargedProp, index=NUM_MCS).to_csv('./sum_results3/params3_chargedProp.csv', header=True)
pd.DataFrame(waitTime, index=NUM_MCS).to_csv('./sum_results3/params3_waitTime.csv',  header=True)
pd.DataFrame(extraDist, index=NUM_MCS).to_csv('./sum_results3/params3_extraDist.csv',  header=True)
# pd.DataFrame(avgExpense, index=NUM_MCS).to_csv('./sum_results3/params3_avgExpense.csv',  header=True)
pd.DataFrame(idleTime, index=NUM_MCS).to_csv('./sum_results3/params3_idleTime.csv',  header=True)
pd.DataFrame(mcsCost, index=NUM_MCS).to_csv('./sum_results3/params3_mcsCost.csv',  header=True)
pd.DataFrame(profit, index=NUM_MCS).to_csv('./sum_results3/params3_profit.csv',  header=True)
