##########################################
#Visualize association rules
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Email: am2685@cornell.edu
#Create Date: 2022_10_18
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_04_11
##########################################

import pandas
import numpy
import re
import networkx as nx
import matplotlib.pyplot as plt
import random
import statistics
import holoviews as hv
from holoviews import opts, dim
from bokeh.themes.theme import Theme
import webbrowser
import os
from Common_functions import *

##########################################
##Parameters
##########################################
print_associations = True
default_graph_size = 250
default_graph_font = '10pt'
default_label_margin = 0.1

##########################################
##Classes
##########################################

##Class for managing graphs of rules
class rule_graph:
    def __init__(self, title, rules, var_dic, attribute_classes={},
                 graph_types=['3dScatter','Chord','Arrows'], graph_parameters={}, rules_to_visualize_cond={},
                 hidden_vars_in_graph=[], grouping_conditions = {}, rule_weight_factor = 'lift', chart_folder='charts'):
        get_att_name = lambda nd: get_attribute_name(nd, var_dic)[2]
        class_by_node = lambda nd, cls: cls[get_att_name(nd)] if get_att_name(nd) in cls else 'no_class'
        self.title = title
        self.graph_types = graph_types
        self.graph_parameters = graph_parameters
        self.rules_to_visualize_cond = rules_to_visualize_cond
        self.hidden_vars_in_graph = hidden_vars_in_graph
        self.rule_weight_factor = rule_weight_factor
        self.rules = self.__select_rules_to_visualize(rules)
        self.var_dic = var_dic
        all_nodes = set([nd for rule in self.rules for nd in self.rules[rule]['antecedent'] + self.rules[rule]['consequent']])
        if grouping_conditions == {}:
            self.classes = {nd:class_by_node(nd,attribute_classes) for nd in all_nodes}
        else:
            self.classes = {nd:get_grouping(nd,grouping_conditions) for nd in all_nodes}
        self.chart_files_path =  os.getcwd() + '\\'+ chart_folder
        if not os.path.exists(self.chart_files_path):
            os.makedirs(self.chart_files_path)
        self.chart_files_path =  self.chart_files_path + '\\'
        pass


    #check if a rule should be visualized
    def __rule_to_visualize(self,rule):
        if 'antecedent' in self.rules_to_visualize_cond:
            condition_type = self.rules_to_visualize_cond['antecedent']['condition_type']
            condition = self.rules_to_visualize_cond['antecedent']['condition']
            for ant in rule['antecedent']:
                if not check_item_by_condition(ant,condition_type,condition):
                    return False

        if 'consequent' in self.rules_to_visualize_cond:
            condition_type = self.rules_to_visualize_cond['consequent']['condition_type']
            condition = self.rules_to_visualize_cond['consequent']['condition']
            for con in rule['consequent']:
                if not check_item_by_condition(con,condition_type,condition):
                    return False

        return True


    #Select rules should be visualized
    def __select_rules_to_visualize(self,rules):
        if self.rules_to_visualize_cond == {}:
            return rules
        visulize_rules = {}
        for rule_id in rules:
            if self.__rule_to_visualize(rules[rule_id]):
                visulize_rules[rule_id] = rules[rule_id]
        return visulize_rules



    # Create data structure of the graph
    def __create_data_structure(self, chart_type):
        get_att_name = lambda nd: get_attribute_name(nd, self.var_dic)[2]
        title_changes = self.graph_parameters[chart_type]['title_changes'] \
                        if chart_type in self.graph_parameters and 'title_changes' in self.graph_parameters[chart_type] \
                        else {}
        data_dict = {}
        classes = {}
        for rule_id in self.rules:
            rule = self.rules[rule_id]
            for con in rule['consequent']:
                con_att = get_att_name(con)
                if con_att not in self.hidden_vars_in_graph:
                    weight = float(rule[self.rule_weight_factor])
                    for ant in rule['antecedent']:
                        ant_att = get_att_name(ant)
                        if ant_att not in self.hidden_vars_in_graph:
                            weight_local = weight
                            ant_class = self.__get_class(ant)
                            con_class = self.__get_class(con)
                            ant = change_name(ant, title_changes)
                            con = change_name(con, title_changes)
                            if (ant, con) in data_dict:
                                weight_local += data_dict[(ant, con)]
                            data_dict[(ant, con)] = weight_local
                            if ant not in classes:
                                classes[ant] = ant_class
                            if con not in classes:
                                classes[con] = con_class
        return data_dict,classes



    # Create Arrows graph
    def __create_arrows_graph(self):
        data_dict, classes = self.__create_data_structure('Arrows')
        edge_labels = {}
        node_color_dic={}
        edge_color_dic= {}
        G1 = nx.DiGraph()
        colors = get_colors(classes)
        for (ant, con) in data_dict:
            G1.add_nodes_from([con])
            con_color = colors[classes[con]] if con in classes and classes[con] in colors else 0
            node_color_dic.update({con:con_color})
            G1.add_nodes_from([ant])
            ant_color = colors[classes[ant]] if ant in classes and classes[ant] in colors else 0
            node_color_dic.update({ant: ant_color})
            G1.add_edge(ant, con, color=ant_color, weight=2)
            edge_color_dic.update({(ant, con): ant_color})
            edge_labels.update({ (ant, con): str(format(data_dict[(ant, con)], ".2f")) })
        node_labels = {ant:ant for (ant, con) in data_dict} | {con:con for (ant, con) in data_dict}
        return G1,node_labels,edge_labels,node_color_dic,edge_color_dic



    # Draw Arrow Graph of Association Rules
    def draw_arrows_graph(self):
        G1, node_labels, edge_labels, node_color_dic, edge_color_dic = self.__create_arrows_graph()
        edges = G1.edges()
        nodes = G1.nodes()
        #edge_colors = [edge_color_dic[nd] for nd in edge_color_dic]
        node_colors = [node_color_dic[nd] for nd in nodes]
        weights = [G1[u][v]['weight'] for u, v in edges]

        pos = nx.spring_layout(G1, k=16, scale=1)
        nx.draw(G1, pos , node_color=node_colors,  width=weights, font_size=16,with_labels=False) #edge_color=edge_colors,
        nx.draw_networkx_edge_labels(G1,pos,edge_labels=edge_labels)

        avg_ver,avg_hor = average_pos(pos)
        margin = self.graph_parameters['Arrows']['label_margin'] \
                 if 'Arrows' in self.graph_parameters and 'label_margin' in self.graph_parameters['Arrows'] \
                 else default_label_margin
        for p in pos:
            pos[p][1] += margin if pos[p][1] > avg_ver else (-1*margin)

        nx.draw_networkx_labels(G1, pos,labels=node_labels)
        filename = self.chart_files_path + 'Associations_Arrows_' + self.title +'.png'
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(filename, dpi=100)
        plt.show()
        pass


    # Get class of a node
    def __get_class(self,nd):
        if nd in self.classes:
            return self.classes[nd]
        return 'no_class'


    # Rotate labels in a HV graph
    def __rotate_label(self,plot, element):
        if element.group in self.graph_parameters:
            if 'rotation' in self.graph_parameters[element.group]:
                rotation = self.graph_parameters[element.group]['rotation']
                text_cds = plot.handles['text_1_source']
                length = len(text_cds.data['angle'])
                text_cds.data['angle'] = [rotation] * length
                xs = text_cds.data['x']
                ys = text_cds.data['y']
                text = numpy.array(text_cds.data['text'])
                xs[xs < 0] -= numpy.array([len(t) * 0.019 for t in text[xs < 0]])
                ys[ys > 0] += numpy.array([0.02 for t in text[ys > 0]])
                ys[ys < 0] -= numpy.array([0.02 for t in text[ys < 0]])
                near_ys = numpy.array([y for y in ys for ny in ys if y - ny < 0.03 and y - ny > 0])
                ys[[i for i in range(len(ys)) if ys[i] in near_ys]] += numpy.array([0.01 for y in near_ys])



    # Draw Chord Graph of Association Rules
    def draw_chord_graph(self):
        class_and_name = lambda nd: classes[nd] + '_' + nd
        gr_size = self.graph_parameters['Chord']['size'] \
                  if 'Chord' in self.graph_parameters and 'size' in self.graph_parameters['Chord'] \
                  else default_graph_size
        label_font_size = self.graph_parameters['Chord']['label_font_size'] \
                          if 'Chord' in self.graph_parameters and 'label_font_size' in self.graph_parameters['Chord'] \
                          else default_graph_font
        data_dict, classes = self.__create_data_structure('Chord')

        if print_associations:
            print('Aggregated %s of associations : ' %(self.rule_weight_factor,))
            print_association_dic_by_order(data_dict)

        data_dict_by_class = {( class_and_name(ant), class_and_name(con)):data_dict[(ant,con)] for (ant,con) in data_dict}

        hv.extension('bokeh')
        color_by_class = get_color_by_class(data_dict,classes,get_color_list_by_category_bokeh(),4)
        #print('Colorset for antimicrobials: ',color_by_class)
        links=pandas.DataFrame( [[ant,con,data_dict_by_class[(ant,con)] ] for (ant,con) in data_dict_by_class],
                                columns = ['source', 'target','value'])
        all_nodes = [nd1 for (nd1, nd2) in data_dict] + [nd2 for (nd1, nd2) in data_dict]
        distinct_nodes = [*set(all_nodes)]
        hv.output(size=gr_size)
        nodes = hv.Dataset(pandas.DataFrame([[class_and_name(nd), nd] for nd in distinct_nodes ],columns=['index','label']), 'index')
        nodes.data = nodes.data.sort_values(by=['index'])
        chord = hv.Chord((links,nodes))
        chord_items = [item for item in chord.nodes.data['label']]
        color_list = [color_by_class[it] for it in  chord_items]
        chd_option = opts.Chord(cmap=color_list, edge_cmap=color_list, edge_color=dim('source').str(),label_text_font_size=label_font_size,
                              labels=dim('label').str(), node_color= dim('index').str() ,hooks=[self.__rotate_label]
                              )
        chord.opts(chd_option)
        filename = self.chart_files_path + 'Associations_Chord_' + self.title + '.html'
        hv.save(chord,filename, backend='bokeh')
        webbrowser.open_new_tab(filename)
        pass



    # Draw 3dScatter plot of Association Rules' Measures
    def draw_rule_suport_conf_lift(self,show_label=False):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        support = numpy.array([self.rules[rule_id]['support'] for rule_id in self.rules])
        confidence = numpy.array([self.rules[rule_id]['confidence'] for rule_id in self.rules])
        lift = numpy.array([self.rules[rule_id]['lift'] for rule_id in self.rules])

        ant = {(self.rules[rule_id]['support'],
                self.rules[rule_id]['confidence'],
                self.rules[rule_id]['lift']):(list_to_str(self.rules[rule_id]['antecedent'])+ '->' +
                                              list_to_str(self.rules[rule_id]['consequent'])) for rule_id in self.rules}

        ax.scatter(support, confidence, lift,  marker="*")

        #plt.xticks(numpy.arange(0.1, 0.7, 0.1))

        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
        ax.set_zlabel('Lift')

        if show_label:
            for x, y, z in ant:
                ax.text(x ,y ,z,ant[(x, y, z)])

        filename = self.chart_files_path + 'Interestingness_3dScatter_' + self.title +'.png'
        plt.savefig(filename)
        plt.show()



    def visualize_rules(self):
        ##Show Arrow graph of rules
        if 'Arrows' in self.graph_types:
            self.draw_arrows_graph()

        ##Show 3dScatter graph of rules
        if '3dScatter' in self.graph_types:
            self.draw_rule_suport_conf_lift()

        ##Chord_Graph
        if 'Chord' in self.graph_types:
            self.draw_chord_graph()


##########################################
##Functions
##########################################

#Returns Average Vertical and Horizontal position of a POS
def average_pos(ps):
    if len(ps) == 0:
        return 0,0
    ver = [float(ps[itm][0]) for itm in ps]
    hor = [float(ps[itm][1]) for itm in ps]
    return statistics.mean(ver),statistics.mean(hor)


#Returns random colors for each item of a dict
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


#Change a name based on a dictionary of names and substitues
def change_name(name,dict):
    for ky in dict:
        name = name.replace(ky,dict[ky])
    return name


#Print associations
def print_association_dic_by_order(assocciation_dict):
    order_associations = [(i,j) for (i,j) in sorted(assocciation_dict.keys())]
    for (i,j) in order_associations:
        print(i,',', j , ':',assocciation_dict[(i,j)])


#Check if an item meet the condition (New condition types can be defined)
def check_item_by_condition(item,condition_type,condition_value):
    if condition_type == 'contain':
        if item.find(condition_value) >= 0:
            return True
        return False
    return False


#Get grouping of an item meet the condition (New condition types can be defined)
def get_grouping(item,conditions):
    grouping = ''
    for condition_type in conditions:
        condition_value = conditions[condition_type]
        if condition_type == 'right':
            grp = item[-condition_value:]
        elif condition_type == 'left':
            grp = item[:condition_value]
        else:
            grp = ''
        grouping += grp
    return grouping


#Returns random colors within the range of class for each item
def get_color_list_by_category_bokeh():
    hv.extension('bokeh')
    color_list= list(hv.Cycle('Category20c').values)
    color_list.extend(list(hv.Cycle('YlOrRd').values)[0:4])
    color_list.extend(list(hv.Cycle('PuBuGn').values)[0:4])
    color_list.extend(list(hv.Cycle('YlOrBr').values)[4:8])
    cat20b = list(hv.Cycle('Category20b').values)
    color_list.extend(cat20b[8:20])
    color_list.extend(cat20b[0:8])
    return color_list



#Returns random colors within the range of class for each item
def get_color_by_class(node_dic, class_dic,color_list,color_range_in_class):
    all_nodes = [nd1 for (nd1,nd2) in node_dic]
    all_nodes.extend([nd2 for (nd1,nd2) in node_dic])
    distinct_nodes = [*set(all_nodes)]
    distinct_nodes.sort()
    distinct_classes = [*set([class_dic[nd] if nd in class_dic else 'no_class' for nd in distinct_nodes])]
    distinct_classes.sort()
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
        if nd in class_dic:
            node_by_color_index[nd] = class_by_color_index[class_dic[nd]] + it
        else:
            node_by_color_index[nd] = class_by_color_index['no_class'] + it
        it += 1
        if it >= color_range_in_class:
            it = 0

    return {nd1:color_list[node_by_color_index[nd1]] for nd1 in distinct_nodes }
"""
    #Customize Colors
    col_by_class = {nd1:color_list[node_by_color_index[nd1]] for nd1 in distinct_nodes }
    col_by_class.update({'Sulfisoxazole':'#9e9ac8', 'Tetracycline': '#bdbdbd','Ciprofloxacin':'#fc4e2a'} )
    return col_by_class
"""



