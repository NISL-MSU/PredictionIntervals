import pickle
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Box-plot
datasets = ["Energy"]  # ["Boston", "Concrete", "Energy", "Kin8nm", "Power", "Synth", "Wine", "Yacht"]

for dataset in datasets:
    for metric in ['MPIW', 'PICP', 'MSE']:
        cvfDualAQD = np.load('CVResults/' + dataset + '/DualAQD/' + 'validation_' + metric + '-DualAQD-' + dataset + '.npy')
        # cvfDualAQDNBS = np.load('CVResults/' + dataset + '/DualAQD/without_batch_sorting/' + 'validation_' +
        #                     metric + '-DualAQD-' + dataset + '.npy')
        cvfQDp = np.load('CVResults/' + dataset + '/QD+/' + 'validation_' + metric + '-QD+-' + dataset + '.npy')
        cvfQD = np.load('CVResults/' + dataset + '/QD/' + 'validation_' + metric + '-QD-' + dataset + '.npy')
        cvfMC = np.load('CVResults/' + dataset + '/MCDropout/' + 'validation_' + metric + '-MCDropout-' + dataset + '.npy')

        df = pd.DataFrame({'DualAQD': cvfDualAQD, 'QD+': cvfQDp, 'QD': cvfQD, 'MC-Dropout-PI': cvfMC})
        ax = df[['DualAQD', 'QD+', 'QD', 'MC-Dropout-PI']].plot(kind='box', showmeans=True)
        ax.tick_params(labelsize=9)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        # Perform t-test
        # compnoBS = stats.ttest_rel(df['DualAQD'], df['DualAQD_noBS'])
        compqdp = stats.ttest_rel(df['DualAQD'], df['QD+'])
        compqd = stats.ttest_rel(df['DualAQD'], df['QD'])
        compmc = stats.ttest_rel(df['DualAQD'], df['MC-Dropout-PI'])
        plt.savefig('CVResults/' + dataset + '/' + 'comparison_' + metric + '.jpg')
        with open('CVResults/' + dataset + '/t-test.txt', 'a') as x_file:
            x_file.write(metric + '\n')
            # x_file.write("DualAQD vs DualAQD_noBS: t-value = %.6f. p-value = %s)" % (float(compnoBS.statistic), str(compnoBS.pvalue)))
            # x_file.write('\n')
            x_file.write("DualAQD vs QD+: t-value = %.6f. p-value = %s)" % (float(compqdp.statistic), str(compqdp.pvalue)))
            x_file.write('\n')
            x_file.write("DualAQD vs QD: t-value = %.6f. p-value = %s)" % (float(compqd.statistic), str(compqd.pvalue)))
            x_file.write('\n')
            x_file.write("DualAQD vs MC-Dropout-PI: t-value = %.6f. p-value = %s)" % (float(compmc.statistic), str(compmc.pvalue)))
            x_file.write('\n')


# dataset = 'Power'
# method = 'DualAQD'
# MPIWtr = np.load('CVResults/' + dataset + '/' + method + '/weights-' + method + '-' + dataset + '-1_historyMPIWtr.npy')
# MPIW = np.load('CVResults/' + dataset + '/' + method + '/weights-' + method + '-' + dataset + '-1_historyMPIW.npy')
# PICPtr = np.load('CVResults/' + dataset + '/' + method + '/weights-' + method + '-' + dataset + '-1_historyPICPtr.npy')
# PICP = np.load('CVResults/' + dataset + '/' + method + '/weights-' + method + '-' + dataset + '-1_historyPICP.npy')
#
# x = np.arange(0, 1000, 4)
#
# fig, ax = plt.subplots()
# ax.set_ylabel('$PICP$', fontsize=17)
# ax.set_xlabel('Epoch', fontsize=17)
# ax.set_ylim([0.6, 1.01])
# plt.xlim(0, 1000)
# plt.xticks(fontsize=17)
# plt.yticks(fontsize=17)
# plt3 = ax.plot(x, PICPtr[0:1000:4], label=r'$PICP_{train}$', color='r')
# plt4 = ax.plot(x, PICP[0:1000:4], label=r'$PICP_{val}$', color='g')
# plt.legend(loc="upper right")
# val_picp, val_mpiw, best_epoch = 0, np.infty, 0
# for epochs in range(1000):
#     picp = PICP[epochs]
#     width = MPIW[epochs]
#     # Criteria 1: If <95, choose max picp, if picp>95, choose any picp if width<minimum width
#     if picp >= 0.9499 and width < val_mpiw:
#         val_picp = picp
#         val_mpiw = width
#         best_epoch = epochs
#
# # best_epoch = 776
#
# ax.vlines(best_epoch, 0,  PICP[best_epoch], linestyle="dashed")
# ax.hlines(PICP[best_epoch], 0, best_epoch, linestyle="dashed")
# ax.scatter(best_epoch, PICP[best_epoch], s=100, c='black')
#
# ax2 = ax.twinx()  # position of the xticklabels in the old x-axis
# ax2.set_ylabel('$MPIW$', fontsize=17)
# plt.yticks(fontsize=17)
# plt1 = ax2.plot(x, MPIWtr[0:1000:4], label=r'$MPIW_{train}$')
# plt2 = ax2.plot(x, MPIW[0:1000:4], label=r'$MPIW_{val}$')
# plts = plt1 + plt2 + plt3 + plt4
# labs = [p.get_label() for p in plts]
# ax.legend(plts, labs, loc="lower left", fontsize=17, bbox_to_anchor=(.5, 0.3))  # , bbox_to_anchor=(.5, 0.35)
