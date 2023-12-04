##########################################
#Agglomerative Hierarchical Clustering
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Email: am2685@cornell.edu
#Create Date: 2022_10_18
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_04_11
##########################################

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from Common_functions import *

##########################################
##Parameters
##########################################
number_of_clusters = 10
linkage_type = 'ward'
distance_type = 'euclidean'
significance_threshold = 0.05
dendo_max_desc_lines = 5
default_null_impute_val = 0
figure_size = (19,11)
figure_dpi = 100


##########################################
##Classes
##########################################

#Agglomerative Hierarchical Clustering
class agnes:

    def __init__(self, complete_trans, var_dic, data_base, data_sources, sampling_condition,
                 target_attributes =[], analytical_attribute = None, chart_folder = 'charts', filename = 'Network'):
        self.var_dic = var_dic
        self.data_sources = data_sources
        self.sampling_condition = sampling_condition
        self.analytical_attribute = analytical_attribute
        self.trans = complete_trans
        self.data_base = data_base
        self.cluster_analysis = {}
        self.target_attributes = target_attributes
        file_path = os.getcwd() + '\\' + chart_folder
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.filename =  file_path + '\\' + filename
        all_trans , self.label_meta = label_data_by_target_att(self.trans,self.target_attributes) if self.target_attributes != [] else label_data_by_status(self.trans)
        trans_df = pd.DataFrame(all_trans)
        self.feat_df = trans_df.loc[:,trans_df.columns != 'Label']
        self.labels = np.array(trans_df.loc[:,'Label'])



    def __get_att_stat(self, label_att):
        att_analysis = {}
        for item in label_att:
            for att in item:
                val = str(item[att])
                if  att in att_analysis:
                    if val in att_analysis[att]:
                            att_analysis[att][val] += 1
                    else:
                        att_analysis[att][val] = 1
                else:
                    att_analysis[att] = {val: 1}
        return att_analysis


    def __get_label_stat(self,lb):
        label_att = [tr['data'] for tr in self.label_meta if tr['Label'] == lb]
        label_drug = [tr['data'] for tr in self.label_meta if tr['Label'] == lb]
        return {'number':len(label_drug) , 'attribute_stat':self.__get_att_stat(label_att)}


    def __get_cluster_analysis(self,members):
        att_analysis = {}
        cluster_labels = {lb:self.__get_label_stat(lb) for lb in members}
        all_att_stat = [cluster_labels[lb]['attribute_stat'] for lb in cluster_labels]
        number = sum([cluster_labels[lb]['number'] for lb in cluster_labels])
        for item in all_att_stat:
            for att in item:
                if att in att_analysis:
                    for val in item[att]:
                        if val in att_analysis[att]:
                            att_analysis[att][val] += item[att][val]
                        else:
                            att_analysis[att][val] = item[att][val]
                else:
                    att_analysis[att] = {val: item[att][val] for val in item[att]}
        att_analysis = {att:{val:str(round(att_analysis[att][val] * 100.0 / number,2))+' %' for val in att_analysis[att] if att_analysis[att][val] > significance_threshold * number} for att in att_analysis  }
        return {'number': number , 'att_analysis':att_analysis}



    def __record_node(self,node_no, node,clusterno, analysis_id):
        self.data_base.execute_sql("""
        INSERT INTO ClusterNode ([Node] ,[ClusterNo] ,[Analysis_Id], [Data])  
                         VALUES (\'%s\',\'%s\',\'%s\',\'%s\') 
                          """ % (node_no,str(clusterno),analysis_id,node))
        print(node)


    def __record_cluster(self, cl, analysis_id):
        cluster_size = self.cluster_member[cl]['count']
        self.cluster_analysis[cl] = self.__get_cluster_analysis(self.cluster_member[cl]['members'])
        cluster_analysis_str = dict_to_str( self.cluster_analysis[cl]['att_analysis'])
        self.data_base.execute_sql("""
        INSERT INTO Cluster([ClusterNo]          ,[Analysis_Id]          ,[Cluster_Size]         ,[Cluster_Analysis])
                     VALUES (\'%s\',\'%s\',%.9f,\'%s\')
                      """ % (str(cl),analysis_id,cluster_size,cluster_analysis_str))
        print('')
        print('Cluster No:', cl)
        print('Cluster Size:', cluster_size)
        print('')
        print('Cluster Analysis:')
        print(cluster_analysis_str)
        print('')
        print('Included Types:')
        node_no = 0
        for mem in self.cluster_member[cl]['members']:
            node_no += 1
            self.__record_node(node_no, mem, cl, analysis_id)
        print('------------------------------------------------------------------------------------------------')


    def __record_analysis(self):
        analysis_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        sampling_condition_str = self.sampling_condition.replace("'","")
        datasources_str = list_to_str(self.data_sources)
        self.data_base.execute_sql("""
        INSERT INTO ClusterAnalysis ([DataSet]  ,[Sample_Condition]  ,[Analysis_Id] , [Clusters])     
                             VALUES (\'%s\',\'%s\',\'%s\',%.9f)
                              """ % (datasources_str,sampling_condition_str,analysis_id,number_of_clusters))
        for cl in self.cluster_member:
            self.__record_cluster(cl,analysis_id)
        self.data_base.commit_transactions()


    def cluster_data(self):
        members_count = lambda x: len([cl for (cl,lb) in clusters if cl == x])
        self.hierarchical_cluster = AgglomerativeClustering(n_clusters=number_of_clusters, affinity=distance_type, linkage=linkage_type,compute_distances=True)
        labels = self.hierarchical_cluster.fit_predict(self.feat_df.fillna(default_null_impute_val))
        distinct_labels = [*set(labels)]
        clusters = [(labels[it],self.labels[it]) for it in range(len(labels))]
        distinct_cluster_member = [*set(clusters)]
        self.cluster_member = {cl:{'members':[l for (c,l) in distinct_cluster_member if c == cl],'count':members_count(cl)} for cl in distinct_labels}
        self.__record_analysis()


    def show_clusters(self):
        plt.figure(constrained_layout=True,figsize=figure_size)
        #plt.title('', fontsize=20)
        plot_dendrogram(self.hierarchical_cluster, truncate_mode="lastp", p=number_of_clusters, leaf_rotation=90)
        xlocs, _ = plt.xticks()
        ylocs, ylabels = plt.yticks()
        new_xlabels = []
        new_xlocs = []
        new_ylabels = ['' for i in ylabels]
        for cl in range(number_of_clusters):
            new_xlocs.append(xlocs[cl])
            new_xlabels.append('Cluster '+ str(cl) + ' (n=' + str(self.cluster_analysis[cl]['number'])+ '):')
            loc = 1
            if not self.analytical_attribute:
                self.analytical_attribute = list(self.cluster_analysis[cl]['att_analysis'].keys())[0]

            if self.analytical_attribute == 'drug' and self.target_attributes != []:
                analytical_items = sort_dic( {dr:self.cluster_analysis[cl]['att_analysis'][dr][st]
                                              for dr in self.cluster_analysis[cl]['att_analysis']
                                               for st in self.cluster_analysis[cl]['att_analysis'][dr]
                                                 if st in ['resistant','intermediate']})
            else:
                analytical_items = sort_dic(self.cluster_analysis[cl]['att_analysis'][self.analytical_attribute])

            for analytical_item in analytical_items:
                new_xlocs.append(xlocs[cl]+loc)
                new_xlabels.append(str(analytical_item) + ' '+ str(analytical_items[analytical_item]))
                loc += 1
                if loc > dendo_max_desc_lines:
                    break
        plt.xticks(ticks=new_xlocs,labels=new_xlabels )
        plt.yticks(ticks=ylocs,labels=new_ylabels )

        plt.savefig(self.filename + '.png', dpi=figure_dpi)
        plt.show()


    def clustering(self):
        self.cluster_data()
        self.show_clusters()


##########################################
##Functions
##########################################

#Plot Dendrogram
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, **kwargs)


#Calculate diatance between two sequence types
def seq_types_ham_distance(st1,st2):
    common_genes = [gn for gn in st1 if gn in st2]
    uncommon_genes = [gn for gn in st1 if gn not in st2] + [gn for gn in st2 if gn not in st1]
    dist = len (uncommon_genes)
    for gn in common_genes:
        if st1[gn] != st2[gn]:
            dist +=1
    return dist


#These functions should be defined based on clustering application
def label_data_by_target_att(trans,target_attributes):
    comp = lambda x: [gn + str(x[gn]) for gn in x if gn in self.target_attributes]
    get_label = lambda x: list_to_str(comp(x))
    resistant_trans = [tr for tr in trans if 'resistant' in tr['target'].values() or 'intermediate' in tr['target'].values()]
    gene_trans = [{str(gn) + str(tr['other'][gn]): 1 for gn in target_attributes if gn in tr['other']} | {'Label': get_label(tr['other'])} for tr in resistant_trans]
    label_meta = [{'Label': get_label(tr['other'])} | {'data': tr['target']} for tr in resistant_trans]
    return gene_trans, label_meta


def label_data_by_status(trans):
    amr_status = lambda x: 3 if x == 'resistant' else 2 if x == 'intermediate' else 1 if x == 'susceptible' else 0
    co_resist = lambda x: [dr for dr in x if amr_status(x[dr]) > 1] if [dr for dr in x if amr_status(x[dr]) > 1] != [] else ['All susceptible']
    get_label = lambda x: list_to_str(co_resist(x))
    drug_trans = [{dr: amr_status(tr['target'][dr]) for dr in tr['target']} | {'Label': get_label(tr['target'])} for tr in trans]
    label_meta = [{'Label': get_label(tr['target'])} | {'data': tr['other']} for tr in trans]
    return drug_trans, label_meta


def float_val(val):
    if is_empty(val):
        return val
    if is_numeric(val):
        return float(val)
    str_val = str(val).replace('%','').replace(' ','')
    if is_numeric(str_val):
        return float(str_val)
    return val


def sort_dic(dc):
    numeric_dic = {ky:float_val(dc[ky]) for ky in dc}
    ordered_items =  sorted(numeric_dic.items(), key=lambda x: x[1], reverse=True)
    return {ky:dc[ky] for (ky,vl) in ordered_items}



