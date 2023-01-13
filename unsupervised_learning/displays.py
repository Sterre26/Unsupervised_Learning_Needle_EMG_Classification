import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
import itertools
import seaborn as sns
import pickle
from hyperopt import hp, space_eval
import pylab

"""
Results can be displayed in various plots.

@author: Sterre de Jonge
"""

## From TESTING >>
def display_clusters(features, y_true, y_pred, show, save_dir, soft, num):
    """
    This function reduces the dimensionality of the embedding space to a 2d 
    dimension, so that the datapoints can be visualised. Visualisation as t-SNE & PCA.
    """

    # t-SNE 2d
    # tsne2d_1 = TSNE(n_components=2, perplexity=30) #default
    # projection2d_1 = tsne2d_1.fit_transform(features)
    # tsne2d_2 = TSNE(n_components=2, perplexity=50) #default
    # projection2d_2 = tsne2d_2.fit_transform(features)
    # tsne2d_3 = TSNE(n_components=2, perplexity=100) #default
    # projection2d_3 = tsne2d_3.fit_transform(features)
    # tsne2d_4 = TSNE(n_components=2, perplexity=150) #default
    # projection2d_4 = tsne2d_4.fit_transform(features)
    # tsne2d_5 = TSNE(n_components=2, perplexity=200) #default
    # projection2d_5 = tsne2d_5.fit_transform(features)
    # tsne2d_6 = TSNE(n_components=2, perplexity=250) #default
    # projection2d_6 = tsne2d_6.fit_transform(features)

    tsne2d = TSNE(
        n_components=2,
        perplexity=150, 
        init='pca',
        learning_rate='auto',
        n_iter=2000,
        n_iter_without_progress=150,
        n_jobs=2,
        random_state=0,
        )
    projection2d = tsne2d.fit_transform(features)

    fig, ax = plt.subplots(num=num, figsize=(12,8))
    df = pd.DataFrame(dict(x=projection2d[:, 0], y=projection2d[:, 1], label=y_true))
    groups = df.groupby('label')
    for name, group in groups:
        if name == 0: label_name, color = "Needle/artefact", 'palegreen'
        if name == 1: label_name, color = "Contraction", 'deepskyblue'
        if name == 2: label_name, color = "Rest", 'tab:pink'
        ax.scatter(group.x, group.y, label=label_name, color=color)
    ax.set_title("t-SNE results with true labels")
    ax.legend()

    if soft is True: plt.savefig(save_dir + '/scatter-tsne-2d-true-soft.eps', format='eps')
    if soft is False: plt.savefig(save_dir + '/scatter-tsne-2d-true.eps', format='eps')

    # tsne3d = TSNE(
    #     n_components=3,
    #     perplexity=150, 
    #     init='pca',
    #     learning_rate='auto',
    #     n_iter=2000,
    #     n_iter_without_progress=150,
    #     n_jobs=2,
    #     random_state=0,
    #     )
    # projection3d = tsne3d.fit_transform(features)
    # x, y, z = list(zip(*projection3d))
    # result3d = pd.DataFrame(dict(x=x, y=y, z=z, label=y_true))
    # groups = result3d.groupby('label')
    # fig = pylab.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # for name, group in groups:
    #     if name == 0: label_name, color = "Needle/artefact", 'tab:green'
    #     if name == 1: label_name, color = "Contraction", 'tab:purple'
    #     if name == 2: label_name, color = "Rest", 'tab:orange'
    #     ax.scatter(group.x, group.y, group.z, label=label_name, color=color)
    # ax.set_title("t-SNE results with true labels")
    # ax.legend()

    # if soft is True: pylab.savefig(save_dir + '/scatter-tsne-3d-true-soft.eps', format='eps')
    # if soft is False: pylab.savefig(save_dir + '/scatter-tsne-3d-true.eps', format='eps')
        
    if show: plt.show()
     
def display_confusionmatrix(cm_original, cm_small, labels, cluster, save_dir, show, soft, normalize=True):

    accuracy = np.trace(cm_small) / np.sum(cm_small).astype('float')
    cmap = plt.get_cmap('Blues')

    # # plot confusion matrix for combined clusters
    # plt.figure(figsize=(4, 4))
    # plt.imshow(cm_small, interpolation='nearest', cmap=cmap)
    # plt.title('Confusion matrix (accuracy={:0.4f})'.format(accuracy))

    # tick_marks = np.arange(len(labels))
    # plt.xticks(tick_marks, labels, rotation=45)
    # plt.yticks(tick_marks, labels, rotation=45)

    # cm_normalized = cm_small.astype('float') / cm_small.sum(axis=1)[:, np.newaxis]

    # thresh = cm_small.max() / 2
    # for i, j in itertools.product(range(cm_small.shape[0]), range(cm_small.shape[1])):
    #         plt.text(j, i, "{} \n({:0.4f})".format(cm_small[i,j], cm_normalized[i, j]),
    #                  horizontalalignment="center",
    #                  color="white" if cm_small[i, j] > thresh else "black")

    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # if soft is True: plt.savefig(save_dir + '/cm_softlabels.eps', format='eps')
    # if soft is False: plt.savefig(save_dir + '/cm.eps', format='eps')
    # if show: plt.show()

    # plot confusion matrix for individual clusters
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_original, interpolation='nearest', cmap=cmap)
    # plt.title('Confusion matrix per cluster')
    plt.title('Confusion matrix (accuracy={:0.4f})'.format(accuracy))
    for i in range(len(labels)+1, cluster+1): labels.append('Empty')
    cluster_lst = []
    for i in range(1, cluster+1): cluster_lst.append('Cluster %s' % i)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, cluster_lst, rotation=45)
    plt.yticks(tick_marks, labels, rotation=45)

    cm_normalized = cm_original.astype('float') / cm_original.sum(axis=1)[:, np.newaxis]
    
    thresh = cm_original.max() / 2
    for i, j in itertools.product(range(cm_original.shape[0]), range(cm_original.shape[1])):
            plt.text(j, i, "{} \n({:0.4f})".format(cm_original[i,j], cm_normalized[i, j]),
                     horizontalalignment="center",
                     color="white" if cm_original[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if soft is True: plt.savefig(save_dir + '/cm_%s_clusters_softlabels.eps' % (cluster), format='eps')
    if soft is False: plt.savefig(save_dir + '/cm_%s_clusters.eps' % (cluster), format='eps')
    if show: plt.show()

def display_example_predictions(array1, array2, label, show, save_dir, fig):
    """
    Displays ten random images from each one of the supplied arrays
    """

    n = 10

    # for item1, item2 in array1, array2:

    if label == "random":
        indices = np.random.randint(len(array1), size=n)
        images1 = array1[indices, :]
        images2 = array2[indices, :]
        save = '/example-predictions-random.png'
    if label == "rest":
        indices = np.random.randint(len(array1), size=n)
        images1 = array1[indices, :]
        images2 = array2[indices, :]
        save = '/example-predictions-rest.png'
    if label == "contraction":
        indices = np.random.randint(len(array1), size=n)
        images1 = array1[indices, :]
        images2 = array2[indices, :]
        save = '/example-predictions-contraction.png'
    if label == "artefact":
        indices = np.random.randint(len(array1), size=n)
        images1 = array1[indices, :]
        images2 = array2[indices, :]
        save = '/example-predictions-artefact.png'

    plt.figure(fig, figsize=(20,4))
    plt.title("Prediction with label {}".format(label))

    for i, (image1, image2) in enumerate(zip(images1, images2)):    
        ax = plt.subplot(2, n, i+1)
        plt.imshow(image1.reshape(128,128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(128, 128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(save_dir + save)
    if show: plt.show()

def hyperopt_boxplots(dir):

    df_trials = pd.read_excel(dir + 'Trials_excel_model2.xlsx')

    # palette = 'Greens' # model 1
    palette = 'Blues' # model 2

    # # figure 1 performance histograms
    # plt.figure(1)
    # plt.hist(df_trials['f1_rest'], bins=60, alpha=0.5, color='tab:blue', label='F1 rest')
    # plt.hist(df_trials['accuracy'], bins=60, alpha=0.5, color='tab:pink', label='Accuracy')
    # plt.xlabel("Data")
    # plt.ylabel("Count")
    # plt.title("Distribution of performance metrics")
    # plt.legend()
    # plt.savefig(dir + 'performance_hists.eps', format='eps') # it can't be transparant...

    # plt.rcParams['legend.title_fontsize'] = 'x-small'

    sns.set_style("whitegrid")
    sns.set(font_scale = 1.6)


    # figure 2 features vs batch size
    f, ax1 = plt.subplots(figsize=(20,5), num=2)
    sns.boxplot(x = df_trials['features'], 
                y = df_trials['f1_rest'], 
                hue = df_trials['batch_size'], 
                hue_order=[64, 256], 
                palette=palette,
                ax=ax1)
    
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['Size 64', 'Size 256']
    ax1.set_xlabel("Features")
    ax1.set_ylabel("F1 rest")
    ax1.legend(handles[:len(labels)], labels, title='Batch size', loc='lower left')
    plt.savefig(dir + '/boxplot_features_batchsize.eps', format='eps')

    # figure 3 features vs layers
    f, ax1 = plt.subplots(figsize=(15,5), num=3)
    sns.boxplot(x = df_trials['features'], 
                y = df_trials['f1_rest'], 
                hue = df_trials['layers'], 
                hue_order=['threelayers', 'fourlayers', 'fivelayers'], 
                palette = palette,
                ax=ax1)
    
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['3 layers', '4 layers', '5 layers']
    ax1.set_xlabel("Features")
    ax1.set_ylabel("F1 rest")
    ax1.legend(handles[:len(labels)], labels, title='Number of layers', loc='lower left')
    plt.savefig(dir + '/boxplot_features_layers.eps', format='eps')

    # figure 4 features 
    f, ax1 = plt.subplots(figsize=(18,6), num=4)
    sns.boxplot(x = df_trials['features'], 
                y = df_trials['f1_rest'], 
                palette = palette,
                ax=ax1)
    ax1.set_xlabel("Features")
    ax1.set_ylabel("F1 rest")
    plt.savefig(dir + '/boxplot_features.eps', format='eps')

    # figure 5 clusters 
    f, ax1 = plt.subplots(figsize=(8,7), num=5)
    sns.boxplot(x = df_trials['clusters'], 
                y = df_trials['f1_rest'], 
                palette = palette,
                ax=ax1)
    ax1.set_xlabel("Clusters")
    ax1.set_ylabel("F1 rest")
    plt.savefig(dir + '/boxplot_clusters.eps', format='eps')

    # figure 6 gamma 
    f, ax1 = plt.subplots(figsize=(8,6), num=6)
    sns.boxplot(x = df_trials['gamma'], 
                y = df_trials['f1_rest'], 
                palette = palette,
                ax=ax1)
    ax1.set_xlabel("Gamma")
    ax1.set_ylabel("F1 rest")
    plt.savefig(dir + '/boxplot_gamma.eps', format='eps')

    # Figure 7 layers
    f, ax1 = plt.subplots(figsize=(8,7), num=7)
    sns.boxplot(x = df_trials['layers'], 
                y = df_trials['f1_rest'], 
                palette = palette,
                ax=ax1)
    ax1.set_xlabel("Layers")
    ax1.set_ylabel("F1 rest")
    ax1.set_xticklabels(['3', '4', '5'])
    plt.savefig(dir + '/boxplot_layers.eps', format='eps')

    # Figure 7 BS
    f, ax1 = plt.subplots(figsize=(8,7), num=8)
    sns.boxplot(x = df_trials['batch_size'], 
                y = df_trials['f1_rest'], 
                palette = palette,
                ax=ax1)
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("F1 rest")
    ax1.set_xticklabels(['64', '256'])
    plt.savefig(dir + '/boxplot_bs.eps', format='eps')

    # # Figure 7 LR schedule
    # f, ax1 = plt.subplots(figsize=(12,7), num=9)
    # sns.boxplot(x = df_trials['learning_schedule'], 
    #             y = df_trials['f1_rest'], 
    #             palette = palette,
    #             ax=ax1)
    # ax1.set_xlabel("Batch size and learning rate schedule")
    # ax1.set_ylabel("F1 rest")
    # ax1.set_xticklabels(['64-constant', '64-clr1', '64-clr2', '256-clr2', '256-constant', '256-clr1'])
    # plt.savefig(dir + '/boxplot_LR.eps', format='eps')


    # # figure 4 learning rate?
    # plt.rcParams['legend.title_fontsize'] = 'x-small'

    # f, ax1 = plt.subplots(figsize=(15,5), num=2)
    # sns.boxplot(x = df_trials['features'], 
    #             y = df_trials['f1_rest'], 
    #             hue = df_trials['learning_rate'], 
    #             hue_order=['64 constant 1e-06', '64 constant 5e-05', '64 constant 1e-05', '64 constant 0.0005', '64 constant 0.0001', '64 constant 0.001', 
    #                     '64 clr1 (1e-07, 0.0001)', '64 clr2 (1e-07, 0.0001)'], 
    #             ax=ax1)
    # sns.stripplot(x = df_trials['features'], 
    #             y = df_trials['f1_rest'], 
    #             hue = df_trials['learning_rate'], 
    #             hue_order=['64 constant 1e-06', '64 constant 5e-05', '64 constant 1e-05', '64 constant 0.0005', '64 constant 0.0001', '64 constant 0.001', 
    #                     '64 clr1 (1e-07, 0.0001)', '64 clr2 (1e-07, 0.0001)'], 
    #             dodge=True, ec='k', linewidth=1)
    # handles, labels = ax1.get_legend_handles_labels()
    
    # labels = ['LR 1e-6', 'LR 5e-5', 'LR 1e-5', 'LR 5e-4', 'LR 1e-4', 'LR 0.001', 
    #         'CLR without scaling ', 'CLR with scaling']
    # ax1.set_xlabel("Features")
    # ax1.set_ylabel("F1 rest")
    # ax1.legend(handles[:len(labels)], labels, title='Batch size', loc='lower left', fontsize='x-small')
    # ax1.set_ylim([0.4, 1.0])
    # plt.savefig(dir + '/boxplot_features_bs64_lr.eps', format='eps')

    # # filtered box plots
    # df_filtered = df_trials[(df_trials.layers == 'threelayers') 
    #                         # & (df_trials.activation == 'ReLU') 
    #                         # & (df_trials.batchnorm == True)
    #                         # & (df_trials.clusters == 6)
    #                         ]
    # df_filtered = df_filtered[
    #                         (df_filtered.learning_rate == '64 constant 0.0001')
    #                         | (df_filtered.learning_rate == '64 constant 0.0005')
    #                         | (df_filtered.learning_rate == '256 clr2 (1e-05, 0.001)')
    #                         | (df_filtered.learning_rate == '256 constant 0.001')
    #                         ]
    # df_low = df_filtered[(df_filtered.features <= 20)]
    # df_low['features'] = '10, 12, 14, 16, 18, 20'
    # df_mediumlow = df_filtered[(df_filtered.features > 20)
    #                             & (df_filtered.features <= 40)]
    # df_mediumlow['features'] = '24, 28, 32, 36, 40'
    # df_mediumhigh = df_filtered[(df_filtered.features > 40)
    #                             & (df_filtered.features <= 128)]
    # df_mediumhigh['features'] = '48, 56, 64, 128'
    # df_high = df_filtered[(df_filtered.features > 128)]
    # df_high['features'] = '256, 512, 1024, 2048, 4096'
    # df_merge = pd.concat([df_low, df_mediumlow, df_mediumhigh, df_high])

    # f, ax2 = plt.subplots(num=4, figsize=(15,5))
    # sns.boxplot(x = df_merge['features'], 
    #             y = df_merge['f1_rest'], 
    #             hue = df_merge['learning_rate'], 
    #             hue_order=['64 constant 0.0005', '64 constant 0.0001','256 constant 0.001', '256 clr2 (1e-05, 0.001)'], 
    #             ax=ax2)
    # # sns.stripplot(x = df_filtered['features'], 
    # #             y = df_filtered['f1_rest'], 
    # #             hue = df_filtered['learning_rate'], 
    # #             hue_order=['64 constant 0.0005', '64 constant 0.0001', '256 constant 0.001', '256 clr2 (1e-05, 0.001)'], 
    # #             dodge=True, ec='k', linewidth=1)
    # ax2.set_xlabel("Features")
    # ax2.set_ylabel("F1 rest")
    # handles, labels = ax2.get_legend_handles_labels()
    # labels = ['bs 64 lr 0.0005', 'bs 64 lr 0.0001', 'bs 256 lr 0.001', 'bs 256 clr with scaling']
    # ax2.legend(handles[:len(labels)], labels, title='Batch size and learning rate', loc='lower left')
    # plt.savefig(dir + '/boxplot_best_learningrate_threelayers_merged.eps', format='eps')

    print("")

def performance_per_epoch_graph(location_csv_file, save_dir):
    results_df = pd.read_csv(location_csv_file)

    columns = results_df.columns[1:]
    lst_acc = [item for item in columns if "acc" in item]
    lst_precision_rest = [item for item in columns if "precision" in item]
    lst_recall_rest = [item for item in columns if "recall" in item]
    lst_f1_rest = [item for item in columns if "f1" in item]

    lst_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # TODO: hier even een for loopje van maken. 

    fig1, ax = plt.subplots()
    for i, (cluster, color) in enumerate(zip(lst_acc, lst_colors[:len(lst_acc)])):
        ax.plot(results_df[cluster], 
        color=color,
        linestyle = '-',
        label=cluster)
    ax.set_xlim(0,100)
    ax.set_ylim(0.5,1.00)
    ax.legend(loc='lower right')    
    ax.set_ylabel("Accuracy [%]")
    ax.set_xlabel("Epochs")
    ax.set_title("Evaluation of accuracy on validation set")
    fig1.savefig(save_dir + 'hyperopt_boxplot{}.png'.format("_acc"))

    fig2, ax = plt.subplots()
    for i, (cluster, color) in enumerate(zip(lst_precision_rest, lst_colors[:len(lst_precision_rest)])):
        ax.plot(results_df[cluster], 
        color=color,
        linestyle = '-',
        label=cluster)
    ax.set_xlim(0,100)
    ax.set_ylim(0.5,1.00)
    ax.legend(loc='lower right')    
    ax.set_ylabel("Precision [%]")
    ax.set_xlabel("Epochs")
    ax.set_title("Evaluation of precision for rest on validation set")
    fig2.savefig(save_dir + 'hyperopt_boxplot{}.png'.format("_precisionrest"))

    fig3, ax = plt.subplots()
    for i, (cluster, color) in enumerate(zip(lst_recall_rest, lst_colors[:len(lst_recall_rest)])):
        ax.plot(results_df[cluster], 
        color=color,
        linestyle = '-',
        label=cluster)
    ax.set_xlim(0,100)
    ax.set_ylim(0.5,1.00)
    ax.legend(loc='lower right')    
    ax.set_ylabel("Recall [%]")
    ax.set_xlabel("Epochs")
    ax.set_title("Evaluation of recall for rest on validation set")
    fig3.savefig(save_dir + 'hyperopt_boxplot{}.png'.format("_recallrest"))

    fig4, ax = plt.subplots()
    for i, (cluster, color) in enumerate(zip(lst_f1_rest, lst_colors[:len(lst_f1_rest)])):
        ax.plot(results_df[cluster], 
        color=color,
        linestyle = '-',
        label=cluster)
    ax.set_xlim(0,100)
    ax.set_ylim(0.5,1.00)
    ax.legend(loc='lower right')    
    ax.set_ylabel("F1 [%]")
    ax.set_xlabel("Epochs")
    ax.set_title("Evaluation of F1 for rest on validation set")
    fig4.savefig(save_dir + 'hyperopt_boxplot{}.png'.format("_f1rest"))

def display_performance_convae(directory, test_acc, test_f1rest, test_acc_soft, test_f1rest_soft, show):

    csv_location_test = directory + 'convae_log_test.csv'
    csv_location_train = directory + 'convae_log_train.csv'
    csv_location_confidence = directory + 'confidence.csv'

    
    csv_test_df = pd.read_csv(csv_location_test)
    csv_train_df = pd.read_csv(csv_location_train)
    csv_confidence_df = pd.read_csv(csv_location_confidence)

    plt.rcParams['axes.labelsize'] = 7

    fig1, axs = plt.subplots(1,3, figsize=(14,3))
    fig1.tight_layout()
    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    csv_train_df.plot(y='train_loss', x='epoch', color='tab:green', linestyle = '-', 
                    label='training $L_{{reconstruction}}$ ({} @ 100)'.format(round(csv_train_df['train_loss'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='val_loss', x='epoch', color='tab:orange', linestyle = '-', 
                    label='validation $L_{{reconstruction}}$ ({} @ 100)'.format(round(csv_train_df['val_loss'].iat[-1],4)), ax=ax1)
    ax1.set(ylabel='losses')
    ax1.set_xlim(1,len(csv_train_df['train_loss']))
    ax1.set_ylim(0,0.009)
    ax1.set(xlabel='')
    ax1.set_axisbelow(True)
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1, fontsize=7)
    ax1.yaxis.grid(color='whitesmoke')
    ax1.xaxis.grid(color='whitesmoke')
    ax1.set(xlabel='epochs')
    ax1.xaxis.set_tick_params(labelsize=7)
    ax1.yaxis.set_tick_params(labelsize=7)

    csv_train_df.plot(y='deltalabel', x='epoch', color='tab:green', linestyle = '-', 
                    label='delta label ({} @ 100)'.format(round(csv_train_df['deltalabel'].iat[-1],4)), ax=ax2)
    ax2.set(ylabel='delta label')
    ax2.set_xlim(1,len(csv_train_df['train_loss']))
    ax2.set_ylim(0,0.03)
    ax2.set(xlabel='')
    ax2.set_axisbelow(True)
    ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1, fontsize=7)
    ax2.yaxis.grid(color='whitesmoke')
    ax2.xaxis.grid(color='whitesmoke')
    ax2.set(xlabel='epochs')
    ax2.xaxis.set_tick_params(labelsize=7)
    ax2.yaxis.set_tick_params(labelsize=7)

    csv_test_df.plot(y='f1_rest', x='epoch', color='tab:orange', linestyle='-', 
                    label='F1 rest ({} @ 100)'.format(round(csv_test_df['f1_rest'].iat[-1],4)), ax=ax3)
    csv_test_df.plot(y='acc', x='epoch', color=plt.cm.tab20c(7), linestyle = '-', 
                    label='Accuracy ({} @ 100)'.format(round(csv_test_df['acc'].iat[-1],4)), ax=ax3)
    csv_test_df.plot(y='f1_rest_soft', x='epoch', color='tab:orange', linestyle='--', 
                    label='F1 rest soft metric ({} @ 100)'.format(round(csv_test_df['f1_rest_soft'].iat[-1],4)), ax=ax3)
    csv_test_df.plot(y='acc_soft', x='epoch', color=plt.cm.tab20c(7), linestyle = '--', 
                    label='Accuracy soft metric ({} @ 100)'.format(round(csv_test_df['acc_soft'].iat[-1],4)), ax=ax3)
   
    # if test_acc is not None:
    
    #     ax3.scatter([91], [test_acc], marker="D", color='mediumpurple', label='Test accuracy: {}'.format(round(test_acc, 4)))
    #     ax3.scatter([91], [test_f1rest], marker="D", color='mediumpurple', label='Test F1-score for rest: {}'.format(round(test_f1rest, 4)))
    #     ax3.scatter([91], [test_acc_soft], marker="*", color='mediumpurple', label='Test accuracy soft metric: {}'.format(round(test_acc_soft, 4)))
    #     ax3.scatter([91], [test_f1rest_soft], marker="*", color='mediumpurple', label='Test F1-score for rest soft emtric: {}'.format(round(test_f1rest_soft, 4)))
    
    ax3.set(ylabel='accuracy and F1 rest')
    ax3.set_xlim(1,len(csv_train_df['train_loss']))
    ax3.set_ylim(0.814, 1.00)
    ax3.set_axisbelow(True)
    ax3.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7)
    ax3.yaxis.grid(color='whitesmoke')
    ax3.xaxis.grid(color='whitesmoke')
    ax3.set(xlabel='epochs')
    ax3.xaxis.set_tick_params(labelsize=7)
    ax3.yaxis.set_tick_params(labelsize=7)


    # csv_confidence_df.plot(y='confidence', x='epoch', color='k', linestyle = '-', 
    #                 label='Confidence label last value: {}'.format(round(csv_confidence_df['confidence'].iat[-1],4)), ax=ax4)
    # csv_confidence_df.plot(y='perc_removed_train', x='epoch', color='tab:green', linestyle = '-', 
    #                 label='Confidence label last value: {}'.format(round(csv_confidence_df['perc_removed_train'].iat[-1],4)), ax=ax4, secondary_y=True)
    # csv_confidence_df.plot(y='perc_removed_val', x='epoch', color='tab:orange', linestyle = '-', 
    #                 label='Confidence label last value: {}'.format(round(csv_confidence_df['perc_removed_val'].iat[-1],4)), ax=ax4, secondary_y=True)
    # # ax4.set(ylabel='Confidence label')
    # ax4.set_xlim(1,len(csv_train_df['train_loss']))
    # ax4.set_ylim(0.52,0.70)
    # ax4.right_ax.set_ylim(10, 30)
    # ax4.set(xlabel='Epochs')
    # ax4.set(ylabel='Confidence level')
    # ax4.right_ax.set(ylabel='percentage removed')
    # ax4.legend([ax4.get_lines()[0], ax4.right_ax.get_lines()[0], ax4.right_ax.get_lines()[1]], ['confidence ({} @ 100)'.format(round(csv_confidence_df['confidence'].iat[-1],4)), 
    #                                                                                             'train data ({} @ 100)'.format(round(csv_confidence_df['perc_removed_train'].iat[-1],1)), 
    #                                                                                             'validation data ({} @ 100)'.format(round(csv_confidence_df['perc_removed_val'].iat[-1],1))], fontsize=8)
    # ax4.set_axisbelow(True)
    # ax4.yaxis.grid(color='whitesmoke')
    # ax4.xaxis.grid(color='whitesmoke')

    plt.savefig(directory + 'performance_graph_convae.eps', format='eps', bbox_inches="tight")
    if show is True: plt.show()

def display_performance_dcec(directory, gamma, show):

    csv_pretrain_cae = directory + 'logs-convae.csv'
    csv_dcec = directory + 'logs-dcec.csv'

    csv_pretrain_cae_df = pd.read_csv(csv_pretrain_cae)
    csv_dcec_df = pd.read_csv(csv_dcec)

    csv_pretrain_cae_df['reconstruction_loss'] = csv_pretrain_cae_df['loss']
    csv_pretrain_cae_df['val_reconstruction_loss'] = csv_pretrain_cae_df['val_loss']
    csv_dcec_df['val_loss_2'] = csv_dcec_df['val_loss']
    csv_pretrain_cae_df['loss'] = np.nan
    csv_dcec_df['clustering_loss'] = csv_dcec_df['clustering_loss']*gamma
    csv_dcec_df['val_clustering_loss'] = csv_dcec_df['val_clustering_loss']*gamma
    csv_dcec_df['perc_removed_train'] = csv_dcec_df['percentage_removed_train']
    csv_dcec_df['perc_removed_val'] = csv_dcec_df['percentage_removed_val']

    csv_train_df = pd.concat([csv_pretrain_cae_df, csv_dcec_df], ignore_index=True)

    plt.rcParams['axes.labelsize'] = 7

    fig1, axs = plt.subplots(1,3, figsize=(14,3))
    fig1.tight_layout()
    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    csv_train_df.plot(y='loss', color='tab:green', linestyle = 'solid', 
                    label='training $L_{{total}}$ ({} @ 100)'.format(round(csv_train_df['loss'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='clustering_loss', color=plt.cm.tab20c(9), linestyle = 'dashed', 
                    label='training $ \gamma * L_{{clustering}}$ ({} @ 100)'.format(round(csv_train_df['clustering_loss'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='reconstruction_loss', color=plt.cm.tab20c(9), linestyle = 'dotted', 
                    label='training $L_{{reconstruction}}$ ({} @ 100)'.format(round(csv_train_df['reconstruction_loss'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='val_loss_2', color='tab:orange', linestyle = 'solid', 
                    label='validation $L_{{t}}$ ({} @ 100)'.format(round(csv_train_df['val_loss_2'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='val_clustering_loss', color=plt.cm.tab20c(5), linestyle = 'dashed', 
                    label='validation $ \gamma * L_{{c}}$ ({} @ 100)'.format(round(csv_train_df['val_clustering_loss'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='val_reconstruction_loss', color=plt.cm.tab20c(5), linestyle = 'dotted', 
                    label='validation $L_{{r}}$ ({} @ 100)'.format(round(csv_train_df['val_reconstruction_loss'].iat[-1],4)), ax=ax1)

    ax1.set(ylabel='losses')
    ax1.set_xlim(1,len(csv_train_df['epoch']))
    ax1.set_ylim(0,0.009)
    ax1.vlines(len(csv_pretrain_cae_df['epoch']), 0, 1, color='dimgrey', linestyle='--')
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7)
    ax1.yaxis.grid(color='whitesmoke')
    ax1.xaxis.grid(color='whitesmoke')
    ax1.set(xlabel='epochs')
    ax1.text(1.5, 0.0002, '\N{LEFTWARDS ARROW} pre-training', fontsize=6)
    ax1.text(16, 0.0002, 'training \N{RIGHTWARDS ARROW}', fontsize=6)
    ax1.xaxis.set_tick_params(labelsize=7)
    ax1.yaxis.set_tick_params(labelsize=7)
    


    csv_train_df.plot(y='deltalabel', color='tab:green', linestyle = '-', 
                    label='delta label ({} @ 100)'.format(round(csv_train_df['deltalabel'].iat[-1],4)), ax=ax2)
    ax2.set(ylabel='delta label')
    ax2.set_xlim(1,len(csv_train_df['train_loss']))
    ax2.set_ylim(0,0.03)
    ax2.set(xlabel='')
    ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1, fontsize=7)
    ax2.vlines(len(csv_pretrain_cae_df['epoch']), 0, 1, color='dimgrey', linestyle='--')
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='whitesmoke')
    ax2.xaxis.grid(color='whitesmoke')
    ax2.set(xlabel='epochs')
    ax2.text(1.5, 0.0005, '\N{LEFTWARDS ARROW} pre-training', fontsize=6)
    ax2.text(16, 0.0005, 'training \N{RIGHTWARDS ARROW}', fontsize=6)
    ax2.xaxis.set_tick_params(labelsize=7)
    ax2.yaxis.set_tick_params(labelsize=7)


    csv_train_df.plot(y='f1_rest', color='tab:orange', linestyle='solid', 
                    label='F1 rest ({} @ 100)'.format(round(csv_train_df['f1_rest'].iat[-1],4)), ax=ax3)
    csv_train_df.plot(y='acc', color=plt.cm.tab20c(7), linestyle = 'solid', 
                    label='accuracy ({} @ 100)'.format(round(csv_train_df['acc'].iat[-1],4)), ax=ax3)
    csv_train_df.plot(y='f1_rest_soft', color='tab:orange', linestyle='dashed', 
                    label='F1 rest soft metric ({} @ 100)'.format(round(csv_train_df['f1_rest_soft'].iat[-1],4)), ax=ax3)
    csv_train_df.plot(y='acc_soft', color=plt.cm.tab20c(7), linestyle = 'dashed', 
                    label='accuracy soft metric ({} @ 100)'.format(round(csv_train_df['acc_soft'].iat[-1],4)), ax=ax3)
   
    ax3.set(ylabel='accuracy and F1 rest')
    ax3.set_xlim(1,len(csv_train_df['train_loss']))
    ax3.set(xlabel='epochs')
    ax3.set_ylim(0.814, 1.00)
    ax3.vlines(len(csv_pretrain_cae_df['epoch']), 0, 1, color='dimgrey', linestyle='--')
    ax3.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7)
    ax3.set_axisbelow(True)
    ax3.yaxis.grid(color='whitesmoke')
    ax3.xaxis.grid(color='whitesmoke')
    ax3.text(1.5, 0.818, '\N{LEFTWARDS ARROW} pre-training', fontsize=6)
    ax3.text(16, 0.818, 'training \N{RIGHTWARDS ARROW}', fontsize=6)
    ax3.xaxis.set_tick_params(labelsize=7)
    ax3.yaxis.set_tick_params(labelsize=7)

    # csv_train_df.plot(y='confidence',  color='k', linestyle = '-', 
    #                 label='Confidence label last value: {}'.format(round(csv_train_df['confidence'].iat[-1],4)), ax=ax4)
    # csv_train_df.plot(y='perc_removed_train', color='tab:green', linestyle = '-', 
    #                 label='Confidence label last value: {}'.format(round(csv_train_df['perc_removed_train'].iat[-1],4)), ax=ax4, secondary_y=True)
    # csv_train_df.plot(y='perc_removed_val', color='tab:orange', linestyle = '-', 
    #                 label='Confidence label last value: {}'.format(round(csv_train_df['perc_removed_val'].iat[-1],4)), ax=ax4, secondary_y=True)
    # ax4.set_xlim(1,len(csv_train_df['train_loss']))
    # ax4.set_ylim(0.52,1.00)
    # ax4.right_ax.set_ylim(10, 30)
    # ax4.set(xlabel='Epochs')
    # ax4.vlines(len(csv_pretrain_cae_df['epoch']), 0, 1, color='k', linestyle='--')
    # ax4.set(ylabel='Confidence level')
    # ax4.right_ax.set(ylabel='percentage removed')
    # ax4.legend([ax4.get_lines()[0], ax4.right_ax.get_lines()[0], ax4.right_ax.get_lines()[1]], ['confidence ({} @ 100)'.format(round(csv_train_df['confidence'].iat[-1],4)), 
    #                                                                                             'train data ({} @ 100)'.format(round(csv_train_df['perc_removed_train'].iat[-1],1)), 
    #                                                                                             'validation data ({} @ 100)'.format(round(csv_train_df['perc_removed_val'].iat[-1],1))], fontsize=8)
    # ax4.set_axisbelow(True)
    # ax4.yaxis.grid(color='whitesmoke')
    # ax4.xaxis.grid(color='whitesmoke')

    plt.savefig(directory + '/performance_graph_dcec.eps', format='eps', bbox_inches="tight")
    if show: plt.show()

def display_performance_convae_rest(directory):

    csv_df = pd.read_csv(directory + 'logs-convae.csv')

    plt.rcParams['axes.labelsize'] = 7

    fig1, axs = plt.subplots(1,3, figsize=(14,3))
    fig1.tight_layout()
    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    csv_df.plot(y='train_loss', x='epoch', color='tab:green', linestyle = '-', 
                    label='training $L_{{reconstruction}}$ ({} @ 100)'.format(round(csv_df['train_loss'].iat[-1],4)), ax=ax1)
    csv_df.plot(y='val_loss', x='epoch', color='tab:red', linestyle = '-', 
                    label='validation $L_{{reconstruction}}$ ({} @ 100)'.format(round(csv_df['val_loss'].iat[-1],4)), ax=ax1)
    ax1.set(ylabel='losses')
    ax1.set_xlim(1,len(csv_df['train_loss']))
    ax1.set_ylim(0,0.009)
    ax1.set(xlabel='')
    ax1.set_axisbelow(True)
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1, fontsize=7)
    ax1.yaxis.grid(color='whitesmoke')
    ax1.xaxis.grid(color='whitesmoke')
    ax1.set(xlabel='epochs')
    ax1.xaxis.set_tick_params(labelsize=7)
    ax1.yaxis.set_tick_params(labelsize=7)

    csv_df.plot(y='deltalabel', x='epoch', color='tab:green', linestyle = '-', 
                    label='delta label ({} @ 100)'.format(round(csv_df['deltalabel'].iat[-1],4)), ax=ax2)
    ax2.set(ylabel='delta label')
    ax2.set_xlim(1,len(csv_df['train_loss']))
    ax2.set_ylim(0,0.03)
    ax2.set(xlabel='')
    ax2.set_axisbelow(True)
    ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1, fontsize=7)
    ax2.yaxis.grid(color='whitesmoke')
    ax2.xaxis.grid(color='whitesmoke')
    ax2.set(xlabel='epochs')
    ax2.xaxis.set_tick_params(labelsize=7)
    ax2.yaxis.set_tick_params(labelsize=7)

    csv_df.plot(y='acc', x='epoch', color='tab:red', linestyle='-', 
                    label='Accuracy ({} @ 100)'.format(round(csv_df['acc'].iat[-1],4)), ax=ax3)
    

    ax3.set(ylabel='accuracy and F1 rest')
    ax3.set_xlim(1,len(csv_df['train_loss']))
    ax3.set_ylim(0, 0.35)
    ax3.set_axisbelow(True)
    ax3.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7)
    ax3.yaxis.grid(color='whitesmoke')
    ax3.xaxis.grid(color='whitesmoke')
    ax3.set(xlabel='epochs')
    ax3.xaxis.set_tick_params(labelsize=7)
    ax3.yaxis.set_tick_params(labelsize=7)

    plt.show()
    fig1.savefig(directory + 'performance_graph_convae.eps', format='eps', bbox_inches="tight")

    fig2, axs = plt.subplots(2,3, figsize=(20,6))
    fig2.tight_layout()

    classes = ['f1_rest', 'f1_fibrillation', 'f1_PSW', 'f1_Fib_PSW', 'f1_CRD', 'f1_Myotonic_discharge']
    color = ['tab:red', 'moccasin', 'mediumseagreen', 'cornflowerblue', 'mediumvioletred', 'yellowgreen']

    axs = [axs[0][0], axs[0][1], axs[0][2], axs[1][0], axs[1][1], axs[1][2]]

    for i, ax in enumerate(axs): 

        csv_df.plot(y=classes[i], x='epoch', linestyle = '-', color=color[i],
                        label='{} ({} @ 100)'.format(classes[i], round(csv_df[classes[i]].iat[-1],4)), ax=ax)
        ax.set(ylabel='F1')
        ax.set_xlim(1,len(csv_df['train_loss']))
        ax.set_ylim(0, 0.5)
        ax.set_axisbelow(True)
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=2, fontsize=7)
        ax.yaxis.grid(color='whitesmoke')
        ax.xaxis.grid(color='whitesmoke')
        ax.set(xlabel='epochs')
        ax.xaxis.set_tick_params(labelsize=7)
        ax.yaxis.set_tick_params(labelsize=7)

    fig2.savefig(directory + 'performance_graph_f1.eps', format='eps', bbox_inches="tight")

    plt.show()

def display_performance_dcec_rest(directory, gamma):

    csv_pretrain_cae = directory + 'logs-convae.csv'
    csv_dcec = directory + 'logs-dcec.csv'

    csv_pretrain_cae_df = pd.read_csv(csv_pretrain_cae)
    csv_dcec_df = pd.read_csv(csv_dcec)

    csv_pretrain_cae_df['reconstruction_loss'] = csv_pretrain_cae_df['loss']
    csv_pretrain_cae_df['val_reconstruction_loss'] = csv_pretrain_cae_df['val_loss']
    csv_dcec_df['val_loss_2'] = csv_dcec_df['val_loss']
    csv_pretrain_cae_df['loss'] = np.nan
    csv_dcec_df['clustering_loss'] = csv_dcec_df['clustering_loss']*gamma
    csv_dcec_df['val_clustering_loss'] = csv_dcec_df['val_clustering_loss']*gamma

    csv_dcec_df['epoch'] = csv_dcec_df['epoch'] + 15


    csv_train_df = pd.concat([csv_pretrain_cae_df, csv_dcec_df], ignore_index=True)

    print(csv_train_df['f1_rest'])

    plt.rcParams['axes.labelsize'] = 7

    fig1, axs = plt.subplots(1,3, figsize=(14,3))
    fig1.tight_layout()
    
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    csv_train_df.plot(y='loss', color='tab:green', linestyle = 'solid', 
                    label='training $L_{{total}}$ ({} @ 100)'.format(round(csv_train_df['loss'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='clustering_loss', color=plt.cm.tab20c(9), linestyle = 'dashed', 
                    label='training $ \gamma * L_{{clustering}}$ ({} @ 100)'.format(round(csv_train_df['clustering_loss'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='reconstruction_loss', color=plt.cm.tab20c(9), linestyle = 'dotted', 
                    label='training $L_{{reconstruction}}$ ({} @ 100)'.format(round(csv_train_df['reconstruction_loss'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='val_loss_2', color='tab:red', linestyle = 'solid', 
                    label='validation $L_{{t}}$ ({} @ 100)'.format(round(csv_train_df['val_loss_2'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='val_clustering_loss', color='lightcoral', linestyle = 'dashed', 
                    label='validation $ \gamma * L_{{c}}$ ({} @ 100)'.format(round(csv_train_df['val_clustering_loss'].iat[-1],4)), ax=ax1)
    csv_train_df.plot(y='val_reconstruction_loss', color='lightcoral', linestyle = 'dotted', 
                    label='validation $L_{{r}}$ ({} @ 100)'.format(round(csv_train_df['val_reconstruction_loss'].iat[-1],4)), ax=ax1)

    ax1.set(ylabel='losses')
    ax1.set_xlim(1,len(csv_train_df['epoch']))
    ax1.set_ylim(0,0.009)
    ax1.vlines(len(csv_pretrain_cae_df['epoch']), 0, 1, color='dimgrey', linestyle='--')
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7)
    ax1.yaxis.grid(color='whitesmoke')
    ax1.xaxis.grid(color='whitesmoke')
    ax1.set(xlabel='epochs')
    ax1.text(1.5, 0.0002, '\N{LEFTWARDS ARROW} pre-training', fontsize=5)
    ax1.text(16, 0.0002, 'training \N{RIGHTWARDS ARROW}', fontsize=5)
    ax1.xaxis.set_tick_params(labelsize=7)
    ax1.yaxis.set_tick_params(labelsize=7)
    
    csv_train_df.plot(y='deltalabel', color='tab:green', linestyle = '-', 
                    label='delta label ({} @ 100)'.format(round(csv_train_df['deltalabel'].iat[-1],4)), ax=ax2)
    ax2.set(ylabel='delta label')
    ax2.set_xlim(1,len(csv_train_df['train_loss']))
    ax2.set_ylim(0,0.03)
    ax2.set(xlabel='')
    ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=1, fontsize=7)
    ax2.vlines(len(csv_pretrain_cae_df['epoch']), 0, 1, color='dimgrey', linestyle='--')
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(color='whitesmoke')
    ax2.xaxis.grid(color='whitesmoke')
    ax2.set(xlabel='epochs')
    ax2.text(1.5, 0.0005, '\N{LEFTWARDS ARROW} pre-training', fontsize=5)
    ax2.text(16, 0.0005, 'training \N{RIGHTWARDS ARROW}', fontsize=5)
    ax2.xaxis.set_tick_params(labelsize=7)
    ax2.yaxis.set_tick_params(labelsize=7)

    csv_train_df.plot(y='acc', color='tab:red', linestyle = 'solid', 
                    label='accuracy ({} @ 100)'.format(round(csv_train_df['acc'].iat[-1],4)), ax=ax3)
   
    ax3.set(ylabel='accuracy')
    ax3.set_xlim(1,len(csv_train_df['train_loss']))
    ax3.set(xlabel='epochs')
    ax3.set_ylim(0, 0.35)
    ax3.vlines(len(csv_pretrain_cae_df['epoch']), 0, 1, color='dimgrey', linestyle='--')
    ax3.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2, fontsize=7)
    ax3.set_axisbelow(True)
    ax3.yaxis.grid(color='whitesmoke')
    ax3.xaxis.grid(color='whitesmoke')
    ax3.text(1.5, 0.005, '\N{LEFTWARDS ARROW} pre-training', fontsize=5)
    ax3.text(16, 0.005, 'training \N{RIGHTWARDS ARROW}', fontsize=5)
    ax3.xaxis.set_tick_params(labelsize=7)
    ax3.yaxis.set_tick_params(labelsize=7)

    plt.show()
    fig1.savefig(directory + '/performance_graph_dcec.eps', format='eps', bbox_inches="tight")

    fig2, axs = plt.subplots(2,3, figsize=(20,6))
    fig2.tight_layout()

    classes = ['f1_rest', 'f1_fibrillation', 'f1_PSW', 'f1_Fib_PSW', 'f1_CRD', 'f1_Myotonic_discharge']
    color = ['tab:red', 'moccasin', 'mediumseagreen', 'cornflowerblue', 'mediumvioletred', 'yellowgreen']

    axs = [axs[0][0], axs[0][1], axs[0][2], axs[1][0], axs[1][1], axs[1][2]]

    for i, ax in enumerate(axs): 

        csv_train_df.plot(y=classes[i], x='epoch', linestyle = '-', color=color[i],
                        label='{} ({} @ 100)'.format(classes[i], round(csv_train_df[classes[i]].iat[-1],4)), ax=ax)
        ax.set(ylabel='F1')
        ax.set_xlim(1,len(csv_train_df['train_loss']))
        ax.set_ylim(0, 0.5)
        ax.set_axisbelow(True)
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=2, fontsize=7)
        ax.yaxis.grid(color='whitesmoke')
        ax.xaxis.grid(color='whitesmoke')
        ax.set(xlabel='epochs')
        ax.xaxis.set_tick_params(labelsize=7)
        ax.yaxis.set_tick_params(labelsize=7)
        ax.vlines(len(csv_pretrain_cae_df['epoch']), 0, 1, color='dimgrey', linestyle='--')
        ax.text(1.5, 0.4, '\N{LEFTWARDS ARROW} pre-training', fontsize=5)
        ax.text(16, 0.4, 'training \N{RIGHTWARDS ARROW}', fontsize=5)

    plt.show()
    fig2.savefig(directory + 'performance_graph_f1.eps', format='eps', bbox_inches="tight")

    

    





def display_performance_rest(directory):

    csv_location = directory + 'logs-convae.csv'
    
    csv_df = pd.read_csv(csv_location)

    fig1, ax1 = plt.subplots(1,1, figsize=(7,9))

    csv_df.plot(y='acc', x='epoch', color='darkseagreen', linestyle = '-', 
                    label='Accuracy last value: {}'.format(round(csv_df['acc'].iat[-1],4)), ax=ax1)
    csv_df.plot(y='F1-CRD', x='epoch', color='orange', linestyle = '-', 
                    label='F1 score for CRD last value: {}'.format(round(csv_df['F1-CRD'].iat[-1],4)), ax=ax1)
    csv_df.plot(y='F1-rest', x='epoch', color='orange', linestyle = '-', 
                    label='F1 score for rest last value: {}'.format(round(csv_df['F1-CRD'].iat[-1],4)), ax=ax1)
    csv_df.plot(y='F1-fib', x='epoch', color='orange', linestyle = '-', 
                    label='F1 score for fibrillations last value: {}'.format(round(csv_df['F1-CRD'].iat[-1],4)), ax=ax1)
    csv_df.plot(y='F1-PSW', x='epoch', color='orange', linestyle = '-', 
                    label='F1 score for PSW last value: {}'.format(round(csv_df['F1-CRD'].iat[-1],4)), ax=ax1)
    csv_df.plot(y='F1-fib-PSW', x='epoch', color='orange', linestyle = '-', 
                    label='F1 score for PSW and fibrillations last value: {}'.format(round(csv_df['F1-CRD'].iat[-1],4)), ax=ax1)
    csv_df.plot(y='F1-Myotonia', x='epoch', color='orange', linestyle = '-', 
                    label='F1 score for CRD last value: {}'.format(round(csv_df['F1-CRD'].iat[-1],4)), ax=ax1)
    ax1.set(ylabel='Losses')
    ax1.set_xlim(1,len(csv_df['train_loss']))
    # ax1.set_ylim(0,0.02)
    ax1.set(xlabel='')
    ax1.legend()
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='whitesmoke')
    ax1.xaxis.grid(color='whitesmoke')

    plt.savefig(directory + '/performance_graph_convae.eps', format='eps')


if __name__ == '__main__':

    display_performance_dcec('/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_two/14122022_1300_bestmodel/', 
                                gamma=0.05, show=False)
    # display_performance_convae('/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_one/14102022_0820_bestmodelhyperopt_final/', 
    #                             test_acc = 0.8979147214441332, test_f1rest = 0.88285659, test_acc_soft=0.9350537297042906, test_f1rest_soft=0.93227792, show=False)

    # display_performance_convae_rest('/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_one_rest/230103_1923_bestmodel/')

    # display_performance_dcec_rest('/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_two_rest/230105_2006_bestmodel/', gamma=0.05)


    #load trials
    # hyperopt_boxplots('/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_one/03102022_1023_evaluatehyperopt3/') #MODEL1
    # hyperopt_boxplots('/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_two/05122022_1016_evalhyperopt1/') #MODEL2


# RESULTS MODEL 1 ON TESTSET

# 0.8979147214441332
# precision [0.88332464 0.8918568  0.92159252]
# recall [0.94864613 0.89785247 0.84724556]
# fscore [0.91482082 0.89484459 0.88285659]
# support [5355 5355 5355]
# 0.9350537297042906
# precision [0.92674266 0.90133038 0.97889182]
# recall [0.98471009 0.91760722 0.88990166]
# fscore [0.95484741 0.90939597 0.93227792]
# support [5036 3544 4169]