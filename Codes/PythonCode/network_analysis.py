##########################################
#Network Analysis of Microbe Resistance
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Create Date: 2023_03_06
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_03_14
##########################################

import json
import sys
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import naive_greedy_modularity_communities
import networkx.algorithms.community.centrality as cm
import networkx.algorithms.community.asyn_fluid as fl
import plotly.express as px
import itertools
import matplotlib.pyplot as plt
import warnings
import statistics
import dimcli
from dimcli.utils import *
from dimcli.utils.networkviz import NetworkViz
import holoviews as hv
import itertools
import webbrowser
from Common_functions import *
from transaction_set import *
from data_base import *


##########################################
##Parameters
##########################################
graph_type = 'Bipartite'              #'Ordinary' and 'Bipartite' (Default is 'Ordinary')
support_thr = 0.01                    #Minimum support for inclusion
lift_thr = 1.01                       #Minimum lift for inclusion
record_results = True                 #Record of results in the dataset
clean_previous_rules = True           #Remove previous recorded analysis in the database



##########################################
##Classes
##########################################

##Class of network
class network:

    def __init__(self,  trans_by_category, data_base, data_sources, classes,sampling_condition= '',
                 clean_previous_net_analysis = False, chart_folder = 'charts', filename = 'Network'):
        self.trans_by_category = trans_by_category
        self.data_base = data_base
        self.data_sources = data_sources
        self.classes = classes
        self.sampling_condition = sampling_condition
        self.clean_previous_net_analysis = clean_previous_net_analysis
        self.Gr = nx.Graph()
        self.node_labels = {}
        self.edge_labels = {}
        self.node_color_dic={}
        self.edge_color_dic= {}
        self.node_sizes = {}
        file_path = os.getcwd() + '\\' + chart_folder
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.filename =  file_path + '\\' + filename
        all_drugs_in_transactions = {drg for t in [tr['target'] for tr in self.trans_by_category] for drg in t}
        self.drugs = [*set(all_drugs_in_transactions)]
        pass



    def __get_stat_two_drugs(self,itm1,itm2):
        if graph_type == 'Bipartite':
            return {'support': -1, 'lift': -1}
        status_in_transactions = [(tr['target'][itm1], tr['target'][itm2]) for tr in self.trans_by_category
                                  if itm1 in tr['target'] and itm2 in tr['target']]
        total_cases = len(status_in_transactions)
        co_resistant = len([(st1, st2) for (st1, st2) in status_in_transactions if (st1, st2) == ('resistant', 'resistant')])
        drg1_resistant = len([(st1, st2) for (st1, st2) in status_in_transactions if st1 == 'resistant'])
        drg2_resistant = len([(st1, st2) for (st1, st2) in status_in_transactions if st2 == 'resistant'])
        support = co_resistant / total_cases if total_cases > 0 else 0
        lift = (co_resistant * total_cases) / (drg1_resistant * drg2_resistant) if drg1_resistant * drg2_resistant > 0 else 1
        return {'support': support, 'lift': lift}



    def __get_stat_drug_with_item(self, drg, itm, itm_val):
        status_in_transactions = [(tr['target'][drg],tr['other'][itm] ) for tr in self.trans_by_category
                                  if drg in tr['target'] and itm in tr['other']]
        total_cases = len(status_in_transactions)
        resistant = len([(st1, st2) for (st1, st2) in status_in_transactions if st1 == 'resistant' and st2 == itm_val])
        non_resistant = len(
            [(st1, st2) for (st1, st2) in status_in_transactions if st1 != 'resistant' and st2 == itm_val])
        drg_resistant = len([(st1, st2) for (st1, st2) in status_in_transactions if st1 == 'resistant'])
        item_cnt = len([(st1, st2) for (st1, st2) in status_in_transactions if st2 == itm_val])
        support = resistant / (resistant + non_resistant) if (resistant + non_resistant) > 0 else 0
        lift = (resistant * total_cases) / (drg_resistant * item_cnt) if drg_resistant * item_cnt > 0 else 1
        return {'support': support, 'lift': lift}



    def __get_stat_two_items(self, itm1, itm1_val, itm2, itm2_val):
        if graph_type == 'Bipartite':
            return {'support': -1, 'lift': -1}
        status_in_transactions = [(tr['other'][itm1],tr['other'][itm2] ) for tr in self.trans_by_category
                                  if itm1 in tr['other'] and itm2 in tr['other']]
        total_cases = len(status_in_transactions)
        co_exist = len([(st1, st2) for (st1, st2) in status_in_transactions if st1 == itm1_val and st2 == itm2_val])
        item1_cnt = len([(st1, st2) for (st1, st2) in status_in_transactions if st1 == itm1_val])
        item2_cnt = len([(st1, st2) for (st1, st2) in status_in_transactions if st2 == itm2_val])
        support = co_exist / total_cases if total_cases > 0 else 0
        lift = (co_exist * total_cases) / (item1_cnt * item2_cnt) if item1_cnt * item2_cnt > 0 else 1
        return {'support': support, 'lift': lift}



    def __calc_co_occur(self):
        all_drug_drug_relation = [*set([(itm1, itm2)  for tr in self.trans_by_category
                                        for itm1 in tr['target']
                                          for itm2 in tr['target']
                                            if itm1 > itm2])]
        all_item_item_relation = [*set([(itm1, tr['other'][itm1], itm2, tr['other'][itm2])
                                        for tr in self.trans_by_category
                                          for itm1 in tr['other']
                                            for itm2 in tr['other']
                                              if itm1 > itm2])]
        all_drug_item_relation = [*set([(itm1, itm2 , tr['other'][itm2] )
                                        for tr in self.trans_by_category
                                          for itm1 in tr['target']
                                            for itm2 in tr['other'] ])]
        self.co_ocuur_stat = {(itm1, itm2): self.__get_stat_two_drugs(itm1, itm2)
                              for (itm1, itm2) in all_drug_drug_relation} | {(itm1+':'+itm1val, itm2+':'+itm2val): self.__get_stat_two_items(itm1,itm1val, itm2,itm2val)
                                for (itm1, itm2) in all_item_item_relation} | {(itm1, itm2+':'+itm2val): self.__get_stat_drug_with_item(itm1, itm2, itm2val)
                                  for (itm1, itm2, itm2val) in all_drug_item_relation}



    def __filter_relations(self):
        self.co_occur_relations = {}
        self.co_occur_relations = {(itm1, itm2): self.co_ocuur_stat[(itm1, itm2)] for (itm1, itm2) in self.co_ocuur_stat
                                    if  self.co_ocuur_stat[(itm1, itm2)]['support'] >= support_thr
                                    and self.co_ocuur_stat[(itm1, itm2)]['lift'] >= lift_thr
                                   }


    #Add a node to association graph
    def __add_node_to_graph(self,nd,size = 5):
        if nd in self.classes:
            desc = nd+' ('+self.classes[nd]+')'
            node_color = self.colors[self.classes[nd]]
        elif nd in [self.classes[drg] for drg in self.classes]:
            desc = nd
            node_color = self.colors[nd]
        else:
            desc = nd
            node_color = 0

        self.Gr.add_nodes_from([nd])
        self.node_color_dic.update({nd: node_color})
        self.node_labels.update({nd: desc})
        self.node_sizes.update({nd: size})



    # Add an edge to association graph
    def __add_edge_to_graph(self,from_node,to_node,wgth, desc = ''):
        if from_node in self.classes:
            edge_color = self.colors[self.classes[from_node]]
        elif from_node in [self.classes[drg] for drg in self.classes]:
            edge_color = self.colors[from_node]
        else:
            edge_color = 0

        self.Gr.add_edge(from_node,to_node,color=edge_color, weight=wgth)
        self.edge_color_dic.update({(from_node, to_node): edge_color})
        self.edge_labels.update({(from_node, to_node): desc})



    def __get_edge_weight(self,itm1, itm2):
        return self.co_occur_relations[(itm1, itm2)]['lift']


    def __get_node_weight(self,itm):
        related_wiehts = sum([self.co_occur_relations[(itm1, itm2)]['lift']
                               for (itm1, itm2) in self.co_occur_relations
                                if itm1 == itm or itm2 == itm])
        return related_wiehts #min(max(related_wiehts / 20.0 , 10) ,50)



    def __create_network(self):
        self.__calc_co_occur()
        self.__filter_relations()
        self.colors = get_colors(self.classes)

        for (itm1, itm2) in self.co_occur_relations:
                self.__add_node_to_graph(itm1,self.__get_node_weight(itm1))
                self.__add_node_to_graph(itm2,self.__get_node_weight(itm2))
                self.__add_edge_to_graph(itm1, itm2, self.__get_edge_weight(itm1, itm2))

        nx.write_graphml(self.Gr, path= self.filename + '.graphml')



    def __visualize_network(self):
        viznet = NetworkViz(notebook=True, width="100%", height="2100px")
        viznet.toggle_hide_edges_on_drag(True)
        viznet.barnes_hut()
        viznet.repulsion(300)
        #viznet.heading = 'Antibacterial Resistance'

        # reuse plotly color palette
        palette = px.colors.diverging.Temps  # 7 colors

        viznet.from_nx(self.Gr)

        for node in viznet.nodes:
            node['size'] = self.node_sizes[node['id']] #self.Gr.nodes[node['id']]['size']
            node['color'] = self.node_color_dic[node['id']] # 10 #palette[3 * score_bucket]  # get color based on score_bucket (1 or 2)
            #node['borderWidthSelected'] = 5
            node['title'] = 'DEG_Centrality: ' + str(round(self.deg_centrality[node['id']],2)) +  ' and BTW_Centrality: ' + str(round(self.bet_centrality[node['id']],2)) #self.node_labels[node['id']] +


        for edge in viznet.edges:
            #dge['value'] = 0.1
            edge['width'] = 0.1
            edge['size'] = 0.1
            #edge['color'] = 1000 #self.edge_color_dic[(edge['from'],edge['to'])]  if (edge['from'],edge['to']) in self.edge_color_dic else 0

        filename = self.filename + '.html'
        viznet.show(filename)
        webbrowser.open_new_tab(filename)




    def __get_fluidc_communities(self):
        comp = fl.asyn_fluidc(self.Gr,k=12)
        print(tuple(sorted(c) for c in next(comp)))


    def __get_newman_communities(self):
        comp = cm.girvan_newman(self.Gr)
        k = 12
        limited = itertools.takewhile(lambda c: len(c) <= k, comp)
        for communities in limited:
            print(tuple(sorted(c) for c in communities))


    def __get_greedy_communities(self):
        self.clusters = {}
        self.cluster_no = 0
        communities = naive_greedy_modularity_communities(self.Gr,weight='weight',resolution=1)
        print('Communities by greedy approach: ')
        for com in communities:
            self.cluster_no +=1
            self.clusters.update({nd:self.cluster_no for nd in com})
            print(list(com))
        self.modularity = nx_comm.modularity(self.Gr,communities=communities)
        print('Modularity: ',self.modularity)


    def __get_class_communities(self):
        communities = []
        if correlation_target == 'Class':
            return
        for cl in self.classes.values():
            community = set([nd for nd in self.Gr.nodes if nd in self.classes and self.classes[nd] == cl])
            if len(community) > 0 and community not in communities:
                communities.append(community)
        communities.append(set([nd for nd in self.Gr.nodes if nd not in self.classes]))
        print('Communities by class: ')
        for com in communities:
            print(list(com))
        modularity = nx_comm.modularity(self.Gr,communities=communities)
        print('Modularity: ',modularity)



    def __get_net_factors(self):
        self.deg_centrality = nx.degree_centrality(self.Gr)
        print('Deg Centrality: ',self.deg_centrality)
        self.bet_centrality = nx.betweenness_centrality(self.Gr)
        print('Bet Centrality: ',self.bet_centrality)
        self.__get_greedy_communities()
        #self.__get_class_communities()
        #clustering_coef = nx.clustering(self.Gr)
        #self.clustering = dict(sorted(clustering_coef.items(), key=lambda x:x[1]))
        #print ('Clustering: ', self.clustering)
        #self.average_clustering = nx.average_clustering(self.Gr)
        #print ('Average Clustering: ', self.average_clustering)
        #self.__get_newman_communities()
        #self.__get_fluidc_communities()


    def __del_analysis(self):
        self.data_base.execute_sql(""" delete from NetworkAnalysis """ )
        self.data_base.execute_sql(""" delete from NetworkNode """ )


    def __record_node(self, node, analysis_id):
        deg_cent = self.deg_centrality[node]
        bet_cent = self.bet_centrality[node]
        cluster = str(self.clusters[node])
        self.data_base.execute_sql("""
        INSERT INTO NetworkNode ([Node] ,[Bet_Cent]  ,[Deg_Cent]  ,[Cluster] ,[Analysis_Id])  
                         VALUES (\'%s\',%.9f,%.9f,\'%s\',\'%s\') 
                          """ % (node,bet_cent,deg_cent,cluster,analysis_id))


    def __record_net(self):
        analysis_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        sampling_condition_str = self.sampling_condition.replace("'","")
        datasources_str = list_to_str(list(self.data_sources.keys()))
        if self.clean_previous_net_analysis:
            self.__del_analysis()
        self.data_base.execute_sql("""
        INSERT INTO NetworkAnalysis ([DataSet]  ,[Sample_Condition]  ,[Analysis_Id] ,[Modularity] ,[Clusters])     
                             VALUES (\'%s\',\'%s\',\'%s\',%.9f,%.9f)
                              """ % (datasources_str,sampling_condition_str,analysis_id,self.modularity,self.cluster_no))
        for nd in self.deg_centrality:
            self.__record_node(nd,analysis_id)
        self.data_base.commit_transactions()



    def analyze_network(self):
        self.__create_network()
        self.__get_net_factors()
        self.__visualize_network()
        if record_results:
            self.__record_net()


##########################################
##Functions
##########################################

def get_color_by_class(node_dic, class_dic,color_list, color_range_in_class):
    distinct_nodes = [*set([nd1 for (nd1,nd2) in node_dic])]
    distinct_classes = [*set([class_dic[nd] for nd in distinct_nodes])]
    class_by_color_index = {}
    node_by_color_index = {}
    it = 0
    for cl in distinct_classes:
        class_by_color_index[cl] = it
        it += color_range_in_class
        if it >= len(color_list):
            it = 0
    it = 0
    for nd in distinct_nodes:
        node_by_color_index[nd] = class_by_color_index[class_dic[nd]] + it
        it += 1
        if it >= color_range_in_class:
            it = 0
    return {nd1:color_list[node_by_color_index[nd1]] for (nd1,nd2) in node_dic }



def get_colors(dict):
    colors = {}
    ind = 0
    hv.extension('bokeh')
    color_list = list(hv.Cycle('Category20').values)
    for itm in dict:
        colors.update({dict[itm]: color_list[ind]})
        ind += 1
        if ind >= len(color_list):
            ind = 0
    return colors
