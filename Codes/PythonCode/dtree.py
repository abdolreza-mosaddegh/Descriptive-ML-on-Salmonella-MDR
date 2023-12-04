##########################################
#Desicion Tree Model
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Email: am2685@cornell.edu
#Create Date: 2022_10_18
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_04_11
##########################################

import pandas
from sklearn import tree
import graphviz
import os
from Common_functions import *

##########################################
##D-Tree Parameters
##########################################
min_impurity_thr = 0.001               #Default min_impurity threshold for Dtree (Default is 0.001)
ccp_thr = 0.001                        #Default ccp_thr threshold for Dtree (Default is 0.001)
max_depth_thr = 15                     #Default max_depth threshold for Dtree (Default is 15)
min_samples_leaf_thr = 30              #Default min_samples_leaf threshold for Dtree (Default is 30)
tree_null_thr = 0.3                    #Default null values threshold for Dtree (Default is 0.3)
deafault_imputed_value = 0             #Default imputed value for  Dtree (Default is 0)


##########################################
##Classes
##########################################

##Class for dtree model
class dtree:
    def __init__(self,  included_transactions, features, chart_folder = 'charts', filename = 'Dtree', test_trans=None):
        self.included_transactions = included_transactions
        self.features = features
        self.test_trans = test_trans
        file_path = os.getcwd() + '\\' + chart_folder
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.filename =  file_path + '\\' + filename
        pass


    # Prepare transaction to feed the Dtree model
    def __prepare_data_for_dtree(self,transactions, feat, test=False):
        feature_names = feat
        list_of_label = []
        list_of_feature = []
        for tran in transactions:
            list_of_feature.append(tran['features'])
            list_of_label.append(tran['label'])

        list_of_feature, feature_names = handel_null_values(list_of_feature, list_of_label, feature_names, tree_null_thr, test)
        feature_data = pandas.DataFrame(list_of_feature, columns=feature_names)
        label_data = pandas.DataFrame(list_of_label, columns=['Outcome'])
        label_names = []
        for lb in list_of_label:
            if lb not in label_names:
                label_names.append(lb)

        return feature_names, label_names, feature_data, label_data


    # Create the Dtree model
    def __get_dtree_model(self, feature_data, label_data, min_imp=min_impurity_thr, ccp=ccp_thr,
                          max_depth=max_depth_thr, min_samples_leaf=min_samples_leaf_thr):
        treemodel = tree.DecisionTreeClassifier(min_impurity_decrease=min_imp, max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf, ccp_alpha=ccp)
        treemodel.fit(feature_data, label_data)
        path = treemodel.cost_complexity_pruning_path(feature_data, label_data)
        return treemodel, path['ccp_alphas']


    # Create the Dtree Graph
    def __create_dtree(self, features, labels):
        text_representation = tree.export_text(self.treemodel, feature_names=features)
        with open(self.filename +".txt", "w") as fout:
            fout.write(text_representation)

        # print(text_representation)
        dot_data = tree.export_graphviz(self.treemodel, out_file=None,
                                        feature_names=features,
                                        class_names=labels,
                                        filled=True)

        graph = graphviz.Source(dot_data, format="png")
        graph.render(self.filename)
        graph.view(self.filename)
        pass


    # Create the Dtree model and draw graph
    def draw_dtree(self):
        features, labels, train_ft, train_lb = self.__prepare_data_for_dtree(self.included_transactions, self.features)
        self.treemodel, _ = self.__get_dtree_model(train_ft, train_lb)
        self.__create_dtree(features, labels)
        if self.test_trans and len(self.test_trans) > 0:
            _, _, test_ft, test_lb = prepare_data_for_dtree(test_trans, features, test=True)
            self.__test_model(self.treemodel, train_ft, train_lb, test_ft, test_lb)



##########################################
##Functions
##########################################

#Returns the most frequent value on a list of values and frequencies
def most_frequent(frequency_list):
    if frequency_list == {}:
        return None
    most_frequent = list(frequency_list.keys())[0]
    max_frequency = frequency_list[most_frequent]
    for val in frequency_list:
        if frequency_list[val] > max_frequency:
            most_frequent = val
            max_frequency = frequency_list[val]
    return most_frequent


#Returns the most frequent values for a feature by all values
def get_imputed_by_all(val_by_label):
    val_by_all = {}
    for lb in val_by_label:
        for vl in val_by_label[lb]:
            val_by_all[vl] =   val_by_all[vl]  + val_by_label[lb][vl] if vl in val_by_all  else val_by_label[lb][vl]
    if val_by_all == {}:
        return deafault_imputed_value
    return most_frequent(val_by_all)



#Returns the most frequent values for a feature by labels
def get_imputed_by_label(transaction_list,label_list,feature_name):
    feat_by_label = [(label_list[ind],transaction_list[ind][feature_name])  for ind in range(len(transaction_list)) if feature_name in transaction_list[ind] and transaction_list[ind][feature_name]!= None]
    val_by_label = {}
    for lb,vl in feat_by_label:
        val_by_label[lb] = {vl:val_by_label[lb][vl] +1 if lb in val_by_label and vl in val_by_label[lb]  else 1}
    imp_by_label = {lb:most_frequent(val_by_label[lb]) if lb in val_by_label else get_imputed_by_all(val_by_label) for lb in label_list}
    return imp_by_label


#Handel Null Values
def handel_null_values(transaction_list,label_list, feature_names,null_thr = 0.3, test=False):
    final_feature_names = []
    final_transaction_list = []
    imputed_by_feature = {}
    for ft in feature_names:
        feature_values = [tr[ft] if ft in tr else None for tr in transaction_list ]
        null_values = [v for v in feature_values if v == None ]
        if len(feature_values) == 0:
            null_ratio = 1
        else:
            null_ratio = len(null_values) / len(feature_values)
        if null_ratio <= null_thr or test:
            final_feature_names.append(ft)
            imputed_by_feature[ft] = get_imputed_by_label(transaction_list,label_list,ft)

    for ind in range(len(transaction_list)):
        tr = transaction_list[ind]
        lb = label_list[ind]
        new_tr = {ft:tr[ft] if ft in tr and tr[ft]!=None else imputed_by_feature[ft][lb] for ft in final_feature_names }
        final_transaction_list.append(new_tr)

    return final_transaction_list,final_feature_names
