import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



directory = '/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_one/03102022_1023_evaluatehyperopt3/'

data = pd.read_excel(directory + 'Trials_excel.xlsx')

# # figure 1 performance histograms

# plt.figure(1)
# plt.hist(data['f1_rest'], bins=60, alpha=0.5, color='tab:blue', label='F1 rest')
# plt.hist(data['accuracy'], bins=60, alpha=0.5, color='tab:pink', label='Accuracy')
# plt.xlabel("Data")
# plt.ylabel("Count")
# plt.title("Distribution of performance metrics")
# plt.legend()
# plt.savefig(dir + 'performance_hists.eps', format='eps') # it can't be transparant...

# plt.rcParams['legend.title_fontsize'] = 'x-small'

# # figure 2 features vs batch size
# sns.set(font_scale=1.4)

# f, ax1 = plt.subplots(figsize=(15,6), num=2)
# sns.boxplot(x = data['features'], 
#            y = data['f1_rest'], 
#             hue = data['batch_size'], 
#             hue_order=[64, 256], 
#             palette='rocket',
#             ax=ax1)
    
# handles, labels = ax1.get_legend_handles_labels()
# labels = ['Size 64', 'Size 256']
# ax1.set_xlabel("Features")
# ax1.set_ylabel("F1 rest")
# ax1.legend(handles[:len(labels)], labels, title='Batch size', loc='lower left')
# plt.savefig(directory + '/boxplot_features_batchsize.eps', format='eps')

# plt.show()

# figure 3 layers 
sns.set(font_scale=1.2)

f, ax1 = plt.subplots(figsize=(8,6), num=3)
sns.boxplot(x = data['layers'], 
            y = data['f1_rest'], 
            # hue = data['layers'], 
            # hue_order=['threelayers', 'fourlayers', 'fivelayers'], 
            palette='rocket',
            order=['threelayers', 'fourlayers', 'fivelayers'],
            ax=ax1)
    
handles, labels = ax1.get_legend_handles_labels()
labels = ['3 layers', '4 layers', '5 layers']
ax1.set_xlabel("")
ax1.set_ylabel("F1 rest")
# ax1.legend(handles[:len(labels)], labels, title='Number of layers', loc='lower left')
plt.savefig(directory + '/boxplot_layers.eps', format='eps')

# figure 4 clusters 
sns.set(font_scale=1.2)

f, ax1 = plt.subplots(figsize=(8,6), num=4)
sns.boxplot(x = data['clusters'], 
            y = data['f1_rest'], 
            # hue = data['layers'], 
            # hue_order=['threelayers', 'fourlayers', 'fivelayers'], 
            palette='rocket',
            order=[5, 6, 7],
            ax=ax1)
    
handles, labels = ax1.get_legend_handles_labels()
labels = ['3 layers', '4 layers', '5 layers']
ax1.set_xlabel("Number of clusters k")
ax1.set_ylabel("F1 rest")
# ax1.legend(handles[:len(labels)], labels, title='Number of layers', loc='lower left')
plt.savefig(directory + '/boxplot_clusters.eps', format='eps')



#     # # figure 4 learning rate?
#     # plt.rcParams['legend.title_fontsize'] = 'x-small'

#     # f, ax1 = plt.subplots(figsize=(15,5), num=2)
#     # sns.boxplot(x = df_trials['features'], 
#     #             y = df_trials['f1_rest'], 
#     #             hue = df_trials['learning_rate'], 
#     #             hue_order=['64 constant 1e-06', '64 constant 5e-05', '64 constant 1e-05', '64 constant 0.0005', '64 constant 0.0001', '64 constant 0.001', 
#     #                     '64 clr1 (1e-07, 0.0001)', '64 clr2 (1e-07, 0.0001)'], 
#     #             ax=ax1)
#     # sns.stripplot(x = df_trials['features'], 
#     #             y = df_trials['f1_rest'], 
#     #             hue = df_trials['learning_rate'], 
#     #             hue_order=['64 constant 1e-06', '64 constant 5e-05', '64 constant 1e-05', '64 constant 0.0005', '64 constant 0.0001', '64 constant 0.001', 
#     #                     '64 clr1 (1e-07, 0.0001)', '64 clr2 (1e-07, 0.0001)'], 
#     #             dodge=True, ec='k', linewidth=1)
#     # handles, labels = ax1.get_legend_handles_labels()
    
#     # labels = ['LR 1e-6', 'LR 5e-5', 'LR 1e-5', 'LR 5e-4', 'LR 1e-4', 'LR 0.001', 
#     #         'CLR without scaling ', 'CLR with scaling']
#     # ax1.set_xlabel("Features")
#     # ax1.set_ylabel("F1 rest")
#     # ax1.legend(handles[:len(labels)], labels, title='Batch size', loc='lower left', fontsize='x-small')
#     # ax1.set_ylim([0.4, 1.0])
#     # plt.savefig(dir + '/boxplot_features_bs64_lr.eps', format='eps')

#     # filtered box plots
#     df_filtered = df_trials[(df_trials.layers == 'threelayers') 
#                             # & (df_trials.activation == 'ReLU') 
#                             # & (df_trials.batchnorm == True)
#                             # & (df_trials.clusters == 6)
#                             ]
#     df_filtered = df_filtered[
#                             (df_filtered.learning_rate == '64 constant 0.0001')
#                             | (df_filtered.learning_rate == '64 constant 0.0005')
#                             | (df_filtered.learning_rate == '256 clr2 (1e-05, 0.001)')
#                             | (df_filtered.learning_rate == '256 constant 0.001')
#                             ]
#     df_low = df_filtered[(df_filtered.features <= 20)]
#     df_low['features'] = '10, 12, 14, 16, 18, 20'
#     df_mediumlow = df_filtered[(df_filtered.features > 20)
#                                 & (df_filtered.features <= 40)]
#     df_mediumlow['features'] = '24, 28, 32, 36, 40'
#     df_mediumhigh = df_filtered[(df_filtered.features > 40)
#                                 & (df_filtered.features <= 128)]
#     df_mediumhigh['features'] = '48, 56, 64, 128'
#     df_high = df_filtered[(df_filtered.features > 128)]
#     df_high['features'] = '256, 512, 1024, 2048, 4096'
#     df_merge = pd.concat([df_low, df_mediumlow, df_mediumhigh, df_high])

#     f, ax2 = plt.subplots(num=4, figsize=(15,5))
#     sns.boxplot(x = df_merge['features'], 
#                 y = df_merge['f1_rest'], 
#                 hue = df_merge['learning_rate'], 
#                 hue_order=['64 constant 0.0005', '64 constant 0.0001','256 constant 0.001', '256 clr2 (1e-05, 0.001)'], 
#                 ax=ax2)
#     # sns.stripplot(x = df_filtered['features'], 
#     #             y = df_filtered['f1_rest'], 
#     #             hue = df_filtered['learning_rate'], 
#     #             hue_order=['64 constant 0.0005', '64 constant 0.0001', '256 constant 0.001', '256 clr2 (1e-05, 0.001)'], 
#     #             dodge=True, ec='k', linewidth=1)
#     ax2.set_xlabel("Features")
#     ax2.set_ylabel("F1 rest")
#     handles, labels = ax2.get_legend_handles_labels()
#     labels = ['bs 64 lr 0.0005', 'bs 64 lr 0.0001', 'bs 256 lr 0.001', 'bs 256 clr with scaling']
#     ax2.legend(handles[:len(labels)], labels, title='Batch size and learning rate', loc='lower left')
#     plt.savefig(dir + '/boxplot_best_learningrate_threelayers_merged.eps', format='eps')
