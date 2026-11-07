# params1
import pandas as pd

NUM_EV = [200,300,400,500,600,700,800,900]
NUM_MCS = [10, 20,30,40,50,60]
N = len(NUM_EV)
M = len(NUM_MCS)

chargedProp = { str(NUM_EV[i]):[] for i in range(N)}
waitTime = { str(NUM_EV[i]):[] for i in range(N)}
extraDist = { str(NUM_EV[i]):[] for i in range(N)}
avgExpense = { str(NUM_EV[i]):[] for i in range(N)}
idleTime ={ str(NUM_EV[i]):[] for i in range(N)}
mcsCost = { str(NUM_EV[i]):[] for i in range(N)}
profit = { str(NUM_EV[i]):[] for i in range(N)}

for i in range(N):
    for j in range(M):
        data = pd.read_csv('./results2/params1_'+ str(NUM_EV[i]) + '_'
                           + str(NUM_MCS[j]) + '.csv')

        a = data['IevChargedProp'].values[3:]
        chargedProp[str(NUM_EV[i])].append(sum(a)/len(a))

        b = data['IevWaitTime'].values[3:]
        waitTime[str(NUM_EV[i])].append(sum(b) / len(b))

        c = data['IevExtraDist'].values[3:]
        extraDist[str(NUM_EV[i])].append(sum(c) / len(c))

        d = data['IevAvgExpense'].values[3:]
        avgExpense[str(NUM_EV[i])].append((sum(d)+sum(c)) / len(d))

        e = data['mcsIdleTime'].values[3:]
        idleTime[str(NUM_EV[i])].append(sum(e) / len(e))

        f = data['mcsCost'].values[3:]
        mcsCost[str(NUM_EV[i])].append(sum(f) / len(f))

        g = data['mcsProfit'].values[3:]
        profit[str(NUM_EV[i])].append(sum(g) / len(g))


pd.DataFrame(chargedProp, index=NUM_MCS).to_csv('./sum_results/params1_chargedProp.csv', header=True)
pd.DataFrame(waitTime, index=NUM_MCS).to_csv('./sum_results/params1_waitTime.csv',  header=True)
pd.DataFrame(extraDist, index=NUM_MCS).to_csv('./sum_results/params1_extraDist.csv',  header=True)
pd.DataFrame(avgExpense, index=NUM_MCS).to_csv('./sum_results/params1_avgExpense.csv',  header=True)
pd.DataFrame(idleTime, index=NUM_MCS).to_csv('./sum_results/params1_idleTime.csv',  header=True)
pd.DataFrame(mcsCost, index=NUM_MCS).to_csv('./sum_results/params1_mcsCost.csv',  header=True)
pd.DataFrame(profit, index=NUM_MCS).to_csv('./sum_results/params1_profit.csv',  header=True)
