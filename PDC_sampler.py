import numpy as np
import copy as cp
import os.path
import matplotlib.pyplot as plt
import sys
import datetime
import matplotlib.cm as cm
import csv


from scipy import stats
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from PIL import Image
from pylab import *

import argparse



def closs_selection(prev_data, unlabeled_index_list, data_list):
    candidate_index_list = []
    for ui in unlabeled_index_list:
        #print((prev_data == data_list[ui]).any(), prev_data, data_list[ui] )
        if (prev_data == data_list[ui]).any():
            candidate_index_list.append(ui)
    if len(candidate_index_list)>0:
        return candidate_index_list
    else:
        return unlabeled_index_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--estimation", default='LP')
    parser.add_argument("--sampling", default='LC')
    parser.add_argument("--parameter_constraint", help="optional", action="store_true")
    parser.add_argument("--prev_point", help="optional", type=str)
    args = parser.parse_args()
    
    input_data = args.input
    LP_algorithm = args.estimation #'LP', 'LS'                                                                   
    US_strategy = args.sampling  #'LC' ,'MS', 'EA', 'RS'
    is_output_img = True
    parameter_constraint = args.parameter_constraint

    prev_parameter = [float(x) for x in args.prev_point.split(',')]
    #parameter_constraintを使うときは, １つ前のパラメータを入れる
    #python PDC_sampler.py data.csv --parameter_constraint True --prev_parameter [10, 20]  
    
    fig_title = "Sampling by "+LP_algorithm+'+'+ US_strategy
    output_dir = 'next_point.csv'
    out_f = open(output_dir, 'w')
    out_f.close()

    
    #----load data
    f = open(input_data, 'r')

    data_list = []
    label_list = []
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if row[0].isdecimal():
            phase = int(row[0])
        else:
            phase = -1
        label_list.append(phase)
        data_list.append([float(p) for p in row[1:]])
    label_index_list = range(len(data_list))
    labeled_index_list = [index for index in range(len(data_list)) if label_list[index] != -1]
    unlabeled_index_list = [index for index in range(len(data_list)) if label_list[index] == -1]
    dimension = len(data_list[0])

    if parameter_constraint:
        if prev_parameter in data_list:
            prev_point_index = data_list.index(prev_parameter)
        else:
            print('Error')

    data_list = np.array(data_list)
    prev_parameter = np.array(prev_parameter)

    max_label = np.max(list(set(label_list)))
    color_list = [cm.rainbow(float(i)/(max_label)) for i in range(max_label+1)]
    
    ss = StandardScaler()
    ss.fit(data_list)
    data_list_std = ss.transform(data_list)

    

    #----SAMPLING
    label_train = np.copy(label_list)
    #label_train[unlabeled_index_list] = -1

    #estimate phase of each point
    if LP_algorithm == 'LS':
        lp_model = label_propagation.LabelSpreading()
    elif LP_algorithm == 'LP':
        lp_model = label_propagation.LabelPropagation()

    lp_model.fit(data_list_std, label_train)
    predicted_labels = lp_model.transduction_[unlabeled_index_list]
    predicted_all_labels = lp_model.transduction_
    label_distributions = lp_model.label_distributions_[unlabeled_index_list]
    label_distributions_all = lp_model.label_distributions_
    classes = lp_model.classes_

    #print(label_train, classes,  predicted_labels, predicted_all_labels, label_distributions)


    #calculate Uncertainly Score
    if US_strategy == 'E':
        pred_entropies = stats.distributions.entropy(label_distributions.T)
        u_score_list = pred_entropies/np.max(pred_entropies)
        if parameter_constraint:
            cand_index_list = closs_selection(prev_parameter, unlabeled_index_list, data_list)
            pred_entropies_all = stats.distributions.entropy(label_distributions_all.T)
            cand_E_list = []
            for i in cand_index_list:
                cand_E_list.append(pred_entropies_all[i])
            uncertainty_index = [cand_indices[np.argmax(cand_E_list)]]
        else:
            uncertainty_index = [unlabeled_index_list[np.argmax(pred_entropies)]]


    elif US_strategy == 'LC':
        u_score_list = 1- np.max(label_distributions, axis = 1)
        if parameter_constraint:
            cand_index_list = closs_selection(prev_parameter, unlabeled_index_list, data_list)
            cand_LC_list = []
            for i in cand_index_list:
                cand_LC_list.append(label_distributions_all[i])

            uncertainty_index = [cand_index_list[np.argmax(1- np.max(np.array(cand_LC_list), axis = 1))]]
        else:
            uncertainty_index = [unlabeled_index_list[np.argmax(1- np.max(label_distributions, axis = 1))]]

    elif US_strategy == 'MS':

        u_score_list = []
        for pro_dist in label_distributions:
            pro_ordered = np.sort(pro_dist)[::-1]
            margin = pro_ordered[0] - pro_ordered[1]
            u_score_list.append(margin)

        if parameter_constraint:
            cand_index_list = closs_selection(prev_parameter, unlabeled_index_list, data_list)
            cand_margin_list = []
            for i in cand_index_list:
                pro_ordered = np.sort(label_distributions_all[i])[::-1]
                margin = pro_ordered[0] - pro_ordered[1]
                cand_margin_list.append(margin)
            uncertainty_index = [cand_index_list[np.argmin(cand_margin_list)]]    
        else:
            uncertainty_index = [unlabeled_index_list[np.argmin(u_score_list)]]
        u_score_list = 1-  np.array(u_score_list)

    elif US_strategy == 'RS':
        if parameter_constraint:
            cand_index_list = closs_selection(prev_parameter, unlabeled_index_list, data_list)
            uncertainty_index = [np.random.permutation(cand_index_list)[0]]
        else:
            uncertainty_index = [np.random.permutation(unlabeled_index_list)[0]]
        u_score_list = [0.5 for i in range(len(label_distributions))]


    dt_now = datetime.datetime.now()
    print(dt_now)
    print('Next point:', data_list[uncertainty_index[0]])
    out_f = open(output_dir, 'a')
    out_f.write('#'+str(dt_now)+'\n')
    for i in range(len(data_list[uncertainty_index[0]])):
        out_f.write(str(data_list[uncertainty_index[0]][i]))
        if i < len(data_list[uncertainty_index[0]])-1:
            out_f.write(',')
    out_f.close()

    if is_output_img and dimension == 2:

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        plt.figure(figsize=(12, 4.5))
        plt.rcParams["font.size"] = 13
        ax = plt.subplot(131)

        for i in labeled_index_list:
            plt.scatter(data_list[i][0], data_list[i][1],  c=color_list[label_list[i]], marker="o")
        for i in unlabeled_index_list:
            plt.scatter(data_list[i][0], data_list[i][1],  c='gray', marker = 's')
        plt.grid()
        plt.xlim([np.min([data_list[:,0]]), np.max([data_list[:,0]])])
        plt.ylim([np.min([data_list[:,1]]), np.max([data_list[:,1]])])
        plt.title('Checked points', size = 16)



        fig = plt.subplot(132)

        u_score_colors = cm.Greens(u_score_list)
        for i in range(len(data_list[unlabeled_index_list])):
            #print(i, data_list[unlabeled_index_list][i], u_score_colors[i])
            plt.scatter(data_list[unlabeled_index_list][i][0], data_list[unlabeled_index_list][i][1],  c= u_score_colors[i], marker = 's')
        plt.xlim([np.min([data_list[:,0]]), np.max([data_list[:,0]])])
        plt.ylim([np.min([data_list[:,1]]), np.max([data_list[:,1]])])
        plt.title('Uncertainty score', size = 16)


        ax = plt.subplot(133)
        x = np.arange(0, 300)/300.
        x_min, x_max = np.min(data_list[:,0]), np.max(data_list[:,0])
        x = x_min+(x_max- x_min)*x
        y = np.arange(0, 300)/300.
        y_min, y_max = np.min(data_list[:,1]), np.max(data_list[:,1])
        y = y_min+(y_max - y_min)*y

        mesh_list = []
        for yy in y:
            for xx in x:
                mesh_list.append([xx, yy])
        pred_label = lp_model.predict(ss.transform(mesh_list))

        X, Y = np.meshgrid(x, y)
        Z = np.reshape(pred_label, (len(y), len(x)))
        z_min = 0
        z_max = np.max(label_list)

        plt.pcolormesh(X, Y, Z, cmap='rainbow', vmin=0, vmax=np.max(label_list))


        plt.title('Estimated', size = 16)


        plt.suptitle(fig_title, fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)

        output_img_dir = 'snapshot/'+LP_algorithm+'_'+US_strategy+'_PC'+str(parameter_constraint)+'_checkedNum'+str(len(labeled_index_list))+'.png'
        plt.savefig(output_img_dir, dpi = 300)
