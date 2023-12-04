##########################################
#Managing Rules resulted by Association Rule Mining
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Email: am2685@cornell.edu
#Create Date: 2022_10_18
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_04_11
##########################################

import pandas
import numpy
import datetime
from pyspark.sql.functions import *
from sklearn.cluster import KMeans
import statistics
from Common_functions import *

##########################################
##Parameters
##########################################
rules_limit = 2000000                  #Maximum rules selected by association mining process (Default is 1000000)
support_approach = 'RAR'               #Use a support approach of RAR or CROSS to filter rules (Default is RAR)
cluster_rules = True                   #Cluster rules based on interestingness measures (Default is True)

##########################################
##Classes
##########################################

##Class for managing rules resulted by FP_Growth
class rule_set:

    # Constructor of rule_set class
    def __init__(self,included_cohorts, allrules_df, config, transaction_number, complete_trans, var_dic, data_base,
                 pruning_rules=False,apply_real_support=False, only_one_to_one_association=False,combine_rules=False,
                 association_antecedent={},association_consequent={},clean_previous_rules=False,record_in_database=True):
        self.included_cohorts = included_cohorts
        self.rule_mining_spec = config
        self.transaction_number = transaction_number
        self.used_parameters = config['thresholds']
        self.allrules_df = allrules_df
        self.var_dic = var_dic
        self.data_base = data_base
        self.pruning_rules = pruning_rules
        self.apply_real_support = apply_real_support
        self.only_one_to_one_association = only_one_to_one_association
        self.combine_rules = combine_rules
        self.association_antecedent = association_antecedent
        self.association_consequent = association_consequent
        self.clean_previous_rules =  clean_previous_rules
        self.record_in_database =  record_in_database
        self.included_att_in_trans = [tr['target'] | tr['other'] for tr in complete_trans]
        pass


    # Record a rule as a dictionary of {antecedent: [List of antecedents],consequent:[List of consequents],lift: Lift_value , confidence: Conf_vallue}
    def __record_rule(self, rule, ruleid):
        antecedent = rule['antecedent']
        consequent = rule['consequent']
        support = rule['support']
        real_support = rule['real_support']
        lift = rule['lift']
        real_lift = rule['real_lift']
        confidence = rule['confidence']
        cluster = rule['cluster']
        rule_string = list_to_str(antecedent) + ' -> ' + list_to_str(consequent)
        correlation_target = get_param_value(self.rule_mining_spec, 'correlation_target')
        included_datasources_str = list_to_str(get_param_value(self.rule_mining_spec, 'data_sources').keys())
        used_parameters_str = dict_to_str(self.used_parameters)
        focused_patterns_str = list_to_str(get_param_value(self.rule_mining_spec, 'focused_patterns'))
        interested_antecedent_str = list_to_str(get_param_value(self.rule_mining_spec, 'association_antecedent', dafault_value = {'All':'*'}).keys(), exclude_brackets=True)
        interested_consequent_str = list_to_str(get_param_value(self.rule_mining_spec, 'association_consequent', dafault_value = {'All':'*'}).keys(), exclude_brackets=True)
        included_var_str = list_to_str(self.var_dic.keys())
        test_data_condition_str = get_param_value(self.rule_mining_spec,'test_data_condition', dafault_value ='').replace("'","")
        test_data_condition_str = str(get_param_value(self.rule_mining_spec,'test_data_percentage')) + ' Percent of Input Data Used as Test Data' if test_data_condition_str == '' else test_data_condition_str
        sampling_condition_str = get_param_value(self.rule_mining_spec,'sampling_condition').replace("'","")
        included_cohorts_str =  'All Cohorts' if get_param_value(self.rule_mining_spec,'rule_by_cohort', dafault_value = False) else list_to_str(self.included_cohorts)

        self.data_base.execute_sql(""" 
        INSERT INTO AssociationRule(association ,confidence , support  , lift, ruleid  ,status  , cohort, 
                                    included_transactions, included_datasources, mining_parameters, focused_patterns,
                                    correlation_target,interested_antecedent,interested_consequent,included_var,
                                    sampling_condition,test_condition, rule_cluster,real_support,real_lift) 
        VALUES (\'%s\',%.9f,%.9f,%.9f,\'%s\',\'provisional\',\'%s\',%.9f,\'%s\',\'%s\',\'%s\',\'%s\',\'%s\',\'%s\',\'%s\',\'%s\',\'%s\',\'%s\',%.9f,%.9f)
        """ % (rule_string, confidence, support, lift, ruleid, included_cohorts_str,
               self.transaction_number, included_datasources_str, used_parameters_str,focused_patterns_str,
               correlation_target,interested_antecedent_str,interested_consequent_str,included_var_str,
               sampling_condition_str, test_data_condition_str,cluster,real_support,real_lift))

        for ant in antecedent:
            correlation, datasource_name, var_name = get_attribute_name(ant, self.var_dic)
            self.data_base.execute_sql(
                "INSERT INTO Attribute(entity, attribute,ruleid,correlation,type) VALUES (\'%s\',\'%s\',\'%s\',\'%s\',\'antecedent\')"
                % (datasource_name, var_name, ruleid,correlation))

        for con in consequent:
            correlation, datasource_name, var_name = get_attribute_name(con, self.var_dic)
            self.data_base.execute_sql(
                "INSERT INTO Attribute(entity, attribute,ruleid,correlation,type) VALUES (\'%s\',\'%s\',\'%s\',\'%s\',\'consequent\')"
                % (datasource_name, var_name, ruleid,correlation))

        pass


    # Combine two rules with same antecedents
    def __combine_two_rules(self,rule1,rule2):
        if set(rule1['antecedent']) != set(rule2['antecedent']):
            return None

        if set(rule1['consequent']) > set(rule2['consequent']):
            return rule1

        if set(rule2['consequent']) > set(rule1['consequent']):
            return rule2

        combined_rule = {}
        combined_rule['antecedent'] =rule1['antecedent']
        combined_rule['consequent'] =rule1['consequent']+rule2['consequent']
        combined_rule['support'] =rule1['support'] if rule1['support'] < rule2['support'] else rule2['support']
        combined_rule['real_support'] =rule1['real_support'] if rule1['real_support'] < rule2['real_support'] else rule2['real_support']
        combined_rule['confidence'] =rule1['confidence'] if rule1['confidence'] < rule2['confidence'] else rule2['confidence']
        combined_rule['lift'] =rule1['lift'] if rule1['lift'] < rule2['lift'] else rule2['lift']
        combined_rule['real_lift'] =rule1['real_lift'] if rule1['real_lift'] < rule2['real_lift'] else rule2['real_lift']
        combined_rule['cluster'] = ''
        return combined_rule



    def __combine_all_rules(self):
        rules_after_combination = {}
        for ruleid in self.recorded_rules:
            combined = False
            rule = self.recorded_rules[ruleid]

            for comp_ruleid in rules_after_combination:
                compared_rule = rules_after_combination[comp_ruleid]
                if set(rule['antecedent']) == set(compared_rule['antecedent']) and \
                   set(rule['consequent']) <= set(compared_rule['consequent']):
                    combined = True
                    break

            if not combined:
                combined_rule = rule
                for compared_ruleid in self.recorded_rules:
                    compared_rule = self.recorded_rules[compared_ruleid]
                    if set(combined_rule['antecedent']) == set(compared_rule['antecedent']) and \
                       set(combined_rule['consequent']) != set(compared_rule['consequent']):
                        combined_rule = self.__combine_two_rules(combined_rule, compared_rule)

                rules_after_combination[ruleid] = combined_rule

        print_with_timestamp(str(len(rules_after_combination)) + ' rules are remained after combination' )
        return rules_after_combination



    def __get_rar_lift(self,rule):
        itemset = list(rule['antecedent']) + list(rule['consequent'])
        itemset_att = [get_attribute_name(item, self.var_dic)[2] for item in itemset]
        related_trans = [tr for tr in self.included_att_in_trans if set(itemset_att).issubset(set(tr.keys())) ]

        itemset_trans = related_trans
        for item in itemset:
            val, _, att = get_attribute_name(item, self.var_dic)
            itemset_trans = [tr for tr in itemset_trans if att in tr and tr[att] == val]

        ant_trans = related_trans
        for item in list(rule['antecedent']):
            val, _, att = get_attribute_name(item, self.var_dic)
            ant_trans = [tr for tr in ant_trans if att in tr and tr[att] == val]


        con_trans = related_trans
        for item in list(rule['consequent']):
            val, _, att = get_attribute_name(item, self.var_dic)
            con_trans = [tr for tr in con_trans if att in tr and tr[att] == val]

        return (len(itemset_trans) * len(related_trans)) / (len(ant_trans) * len(con_trans)) if (len(ant_trans) * len(con_trans)) > 0 else 1




    def __get_cross_support(self,itemset):
        min_sup = 1
        max_sup = 0
        if self.included_att_in_trans == 0:
            return 0
        for item in itemset:
            val, _, att = get_attribute_name(item, self.var_dic)
            item_trans = [tr for tr in self.included_att_in_trans if att in tr and tr[att] == val]
            supp =  len(item_trans)*1.0 / len(self.included_att_in_trans)
            if supp > max_sup:
                max_sup =  supp
            if supp < min_sup:
                min_sup = supp

        return min_sup / max_sup



    def __get_rar_support(self,rule):
        included_trans = self.included_att_in_trans
        itemset = list(rule['antecedent']) + list(rule['consequent'])
        for item in itemset:
            _, _, att = get_attribute_name(item, self.var_dic)
            included_trans = [tr for tr in included_trans if att in tr]

        if included_trans == 0:
            return 0

        return rule['support'] * len(self.included_att_in_trans) / len(included_trans)



    def __select_rules_by_real_support(self,rules):
        frequent_rules = []
        for rule in rules:
            if support_approach == 'RAR':
                real_supp = self.__get_rar_support(rule)
                real_lift = self.__get_rar_lift(rule)
            elif support_approach == 'CROSS':
                real_supp = self.__get_cross_support(list(rule['antecedent']) + list(rule['consequent']))
                real_lift = rule['lift']
            else:
                real_supp = rule['support']
                real_lift = rule['lift']

            if real_supp >= self.used_parameters['real_support_threshold']:
                new_rule = {}
                new_rule['antecedent'] = rule['antecedent']
                new_rule['consequent'] = rule['consequent']
                new_rule['confidence'] = rule['confidence']
                new_rule['support'] = rule['support']
                new_rule['lift'] = rule['lift']
                new_rule['real_support'] = real_supp
                new_rule['real_lift'] = real_lift
                frequent_rules.append(new_rule)

        print_with_timestamp(str(len(frequent_rules)) + ' rules are extracted by support (based on non-null records) threshold of '+str(self.used_parameters['real_support_threshold']))
        return frequent_rules


    # check if a rule is filtered
    def __filtered_rule(self,rule):
        filtered = False

        if self.association_antecedent == {}:
            all_antecedents = True
        else:
            all_antecedents = False

        if not all_antecedents:
            filtered = True
            for ant in rule['antecedent']:
                _, _, att = get_attribute_name(ant, self.var_dic)
                if att in self.association_antecedent.keys():
                    filtered = False
                    break

        if not filtered and self.only_one_to_one_association:
            if len(rule['antecedent']) != 1:
                filtered = True

        return filtered


    # Compare two rules by efficiency
    def __compare_two_rules(self,rule, compared_rule):
        if set(rule['consequent']) != set(compared_rule['consequent']):
            return 'Distinct'

        set_ant = set(rule['antecedent'])
        set_compared_ant = set(compared_rule['antecedent'])

        if set_ant.issuperset(set_compared_ant):
            if compared_rule['lift'] >= rule['lift']:
                return 'ComparedEfficient'
            else:
                return 'Distinct'

        if set_compared_ant.issuperset(set_ant):
            if rule['lift'] >= compared_rule['lift']:
                return 'RuleEfficient'
            else:
                return 'Distinct'

        return 'Distinct'


    # Select most efficient rules
    def __prune_rules(self,selected_rules):
        efficient_rules = []
        for rule in selected_rules:
            Included = True
            comparing_rules = efficient_rules
            for compared_rule in comparing_rules:
                compare_rules = self.__compare_two_rules(rule, compared_rule)
                if compare_rules == 'ComparedEfficient':
                    Included = False
                    break
                elif compare_rules == 'RuleEfficient':
                    efficient_rules.remove(compared_rule)
            if Included:
                efficient_rules.append(rule)

        print_with_timestamp(str(len(efficient_rules)) + ' rules are remained after pruning' )
        return efficient_rules



    # Record a rule-set selected as relevant
    def record_rules(self):
        relevant_rules =  self.__select_relevant_rules()
        collected_relevant_rules = relevant_rules.collect()
        self.recorded_rules = {}
        print_with_timestamp(str(relevant_rules.count()) + ' Relevant rules are collected')

        if (self.association_antecedent != {}) or self.only_one_to_one_association:
            all_selected_rules = [rule for rule in collected_relevant_rules if not self.__filtered_rule(rule)]
            print_with_timestamp(str(len(all_selected_rules)) + ' rules after applying filters')
        else:
            all_selected_rules = collected_relevant_rules

        if self.apply_real_support:
            all_selected_rules= self.__select_rules_by_real_support(all_selected_rules)

        if self.pruning_rules:
            efficient_rules = self.__prune_rules(all_selected_rules)
        else:
            efficient_rules = all_selected_rules

        mining_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        rule_cnt = 0

        for rule in efficient_rules:
            rule_cnt = rule_cnt + 1
            ruleid = mining_date + '_' + str(rule_cnt)
            self.recorded_rules[ruleid] = {'antecedent': rule['antecedent'],
                                           'consequent': rule['consequent'],
                                           'support': rule['support'],
                                           'confidence': rule['confidence'],
                                           'lift': rule['lift'],
                                           'cluster':'',
                                           'real_support': rule['real_support'] if self.apply_real_support else 0,
                                           'real_lift': rule['real_lift'] if self.apply_real_support else 0
                                           }

        if cluster_rules:
            self.__cluster_rules(self.recorded_rules)

        if self.combine_rules:
            self.recorded_rules_in_DB = self.__combine_all_rules()
            if cluster_rules:
                self.__cluster_rules(self.recorded_rules_in_DB)
        else:
            self.recorded_rules_in_DB = self.recorded_rules

        if self.clean_previous_rules:
            self.__clean_previous_rules()

        for ruleid in self.recorded_rules_in_DB:
            self.__record_rule(self.recorded_rules_in_DB[ruleid], ruleid)

        self.data_base.commit_transactions()
        relevant_rules.unpersist(blocking=False)
        print_with_timestamp(str(len(self.recorded_rules_in_DB)) + ' rules are recorded in DB')
        return self.recorded_rules


    def __update_rule(self, rule_id, fld, val):
        self.data_base.execute_sql("""
        update AssociationRule
        set %s = '%s'
        where ruleid = '%s'  
        """%(fld,val,rule_id))
        self.data_base.commit_transactions()
        pass


    # Select relevant rules by applying lift threshold and filtering rules doesn't contain selected consequent
    def __select_relevant_rules(self):
        con_list = list(self.association_consequent.keys())

        if con_list == ['*'] or con_list ==[]:
            self.selected_rules = self.allrules_df.filter(
                (self.allrules_df.lift > self.used_parameters['lift_threshold']) ) \
                .sort(col("lift").desc(), col("support").desc(), col("confidence").desc()) \
                .limit(rules_limit)
        else:
            filter_cond = ''
            for con in con_list:
                filter_cond += """ (array_contains(col("consequent"), '""" + con + """')) """ if filter_cond == '' else """ | (array_contains(col("consequent"), '""" + con + """')) """
            command_str = """self.selected_rules =  self.allrules_df.filter( (""" + filter_cond + ') & ' + \
                          """(self.allrules_df.lift > self.used_parameters['lift_threshold']) )""" + \
                          """.sort(col("lift").desc(), col("support").desc(), col("confidence").desc())""" + \
                          """.limit(rules_limit) """
            exec(command_str)

        self.selected_rules.cache()
        return self.selected_rules


    #Convert rules to numeric transactions
    def rules_to_numeric_transactoin(self):
        transactions = []
        for rule_id in self.recorded_rules:
            rule = self.recorded_rules[rule_id]
            rule_features = {}
            for ant in rule['antecedent']:
                val, _, att = get_attribute_name(ant, self.var_dic)
                rule_features[att] = get_numeric_value(att,val,var_mapping)
            rule_dic = {'features':rule_features,'label':list_to_str(rule['consequent'],exclude_brackets=True)}
            sup = int(rule['support']*100)
            for itr in range(sup):
                transactions.append(rule_dic)

        return transactions


    # Get state of Association Rules' Measures
    def __get_indicators_state(self,rule):
        supp_state = lambda x, thr: 1 if x < 1.5*thr else 2 if x < 2*thr else 3
        conf_state = lambda x, thr: 1 if x < thr + ((1-thr)/4) else 2 if x < thr + ((1-thr)*3/4) else 3
        lift_state = lambda x, thr: 1 if x < 1.5*thr else 2 if x < 2*thr else 3
        return [supp_state(rule['support'], self.used_parameters['support_threshold']),
                conf_state(rule['confidence'], self.used_parameters['confidence_threshold']),
                lift_state(rule['lift'] , self.used_parameters['lift_threshold'])]


    # Get Name of a Cluster of rules
    def __get_cluster_name(self,cluster_rules):
        supp_state = lambda x, thr: 'Low' if x < 1.5*thr else 'Med' if x < 2*thr else 'High'
        conf_state = lambda x, thr: 'Low' if x < thr + ((1-thr)/4) else 'Med' if x < thr + ((1-thr)*3/4) else 'High'
        lift_state = lambda x, thr: 'Low' if x < 1.5*thr else 'Med' if x < 2*thr else 'High'
        supp_cond = supp_state (statistics.mean([rule['support'] for rule in cluster_rules]), self.used_parameters['support_threshold'])
        conf_cond = conf_state (statistics.mean([rule['confidence'] for rule in cluster_rules]), self.used_parameters['confidence_threshold'])
        lift_cond = lift_state (statistics.mean([rule['lift'] for rule in cluster_rules]), self.used_parameters['lift_threshold'])
        return  supp_cond + 'Support_'+conf_cond + 'Confidence_'+lift_cond+'Lift'


    # Cluster Association Rules based on interestingness measures
    def __cluster_rules(self,rules):
        recognized_clusters = {}
        if len(rules) <9:
            print('Due to small amount of rules, clustering is ignored')
            return

        feat = numpy.array([self.__get_indicators_state(rules[rid]) for rid in rules])
        kmeans = KMeans(random_state=0).fit(feat)
        ind = 0
        rule_cluster = {}
        for rl_id in rules:
            rule_cluster[rl_id] = kmeans.labels_[ind]
            ind +=1

        for rl_id in rules:
            cluster = rule_cluster[rl_id]
            if cluster not in recognized_clusters:
                cluster_name = self.__get_cluster_name([rules[rid] for rid in rules if rule_cluster[rid]==cluster ])
                recognized_clusters[cluster] = cluster_name
            else:
                cluster_name = recognized_clusters[cluster]
            rules[rl_id]['cluster'] = cluster_name

        print_with_timestamp('Rules are clustered into ' + str(len(recognized_clusters)) + ' clusters')
        pass


    # Clear previous rules in database
    def __clean_previous_rules(self):
        self.data_base.execute_sql("""
        delete AssociationRule
        """)
        self.data_base.execute_sql("""
        delete Attribute
        """)
        pass


