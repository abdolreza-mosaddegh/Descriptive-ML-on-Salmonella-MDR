##########################################
#Managing transactions
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Email: am2685@cornell.edu
#Create Date: 2022_10_18
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_04_11
##########################################


import pandas
from Common_functions import *
from sklearn.model_selection import train_test_split

##########################################
##Parameters
##########################################
transaction_limit =1000000             #Maximum transactions for association mining (Default is 1000000)
min_transaction = 20                   #Minimum transactions required for association mining (Default is 20)
empty_cell_threshold = 0.999           #Default Empty Cell threshold for testing association rules (Default is 0.999)
consider_complete_records = False      #Only transactions include all datasets should be considered (Default is False)
meta_data = [                          #Meta data variables should be excluded from transactions used for association mining
]


##########################################
##Classes
##########################################

##Class for managing transactions as an input for FP-Growth algorithm
class transaction_set:

    # Constructor of transaction_set class
    def __init__(self, included_cohorts, data_sources, correlation_target,focused_patterns,trans_id,data_base,
                 excluded_vars=[],sampling_condition='',test_data_condition='',test_data_percentage=0,
                 Dtree_model=False, complementary_data = {}):
        self.included_cohorts = included_cohorts
        self.data_items = data_sources
        self.correlation_target = correlation_target
        self.focused_patterns = focused_patterns
        self.data_base = data_base
        self.excluded_vars = excluded_vars
        self.sampling_condition = sampling_condition
        self.test_data_condition = test_data_condition
        self.test_data_percentage = test_data_percentage
        self.Dtree_model = Dtree_model
        self.complementary_data = complementary_data
        self.transaction_list = []
        self.max_trans_len = 0
        self.var_dict = {}
        self.meta_data_fields = list(trans_id.keys())
        if meta_data != []:
            self.meta_data_fields.extend(meta_data)
        self.id_field = list(trans_id.keys())[0]
        self.id_entity = trans_id[self.id_field]
        pass


    # Split Test and Train data based on user parameters
    def __split_test_train(self):
        where_condition = ' where ' + self.test_data_condition if self.test_data_condition.strip() != '' else ''
        attributes_df = self.data_base.get_records("""
            select distinct %s from %s 
              %s
            """ % (self.id_field,self.id_entity,where_condition))

        cl_num = list(attributes_df[self.id_field])

        if self.test_data_condition.strip() != '':
            self.test_cl_num = cl_num
        elif self.test_data_percentage ==0:
            self.test_cl_num = []
        else:
            _, self.test_cl_num = train_test_split(cl_num, test_size=self.test_data_percentage, random_state=4)

        pass



    # Fetch data items of an Id from datasourc
    def __fetch_items_by_id(self,datasource_name, id):
        select_items = ''
        attributes = self.data_items[datasource_name]

        if attributes == ['*']:
            select_items = '*'
        else:
            for atr in attributes:
                select_items = select_items + (' ' if select_items == '' else ', ') + str(atr)

        attributes_df = self.data_base.get_records("""
           select distinct %s from %s
           where %s = '%s'
            """ % (select_items, datasource_name,self.id_field, id))

        return attributes_df



    # Get a dict of varnames and values of an Id from datasourc
    def __get_var_values(self, datasource_name, id, foc_pattern = []):
        var_value = {'target':{},'other':{}}
        vars = {}
        attributes_df = self.__fetch_items_by_id(datasource_name, id)

        if len(attributes_df.index) <= 0:
            return vars, var_value

        for rw in attributes_df.index:
            for cl in attributes_df.columns:
                varname = str(cl)
                value = attributes_df.loc[rw, cl]
                if varname == self.correlation_target:
                    varname = str(value)
                    value = str(attributes_df.loc[rw, 'Status'])
                    if not is_empty(value) and (foc_pattern==[] or value in foc_pattern) :
                        var_value['target'].update({varname:value})
                        vars.update({varname: datasource_name})
                elif not is_empty(value) and varname not in self.meta_data_fields and varname not in self.excluded_vars and varname not in ['Status', 'Class','Drug'] :
                    if varname not in var_value['other']:
                        var_value['other'].update({varname:value})
                        vars.update({varname: datasource_name})
                    elif var_value['other'][varname] != value:
                        var_value['other'] = update_dict(var_value['other'],{varname:value})

        return vars, var_value



    # Get data items of test data of an Id
    def __get_test_data_by_id(self,id):
        tr = []
        features={}
        labels = {}
        vars = {}
        var_value = {'target':{},'other':{}}

        for datasource_name in self.data_items:
            ds_vars, ds_var_value = self.__get_var_values(datasource_name, id)
            vars.update(ds_vars)
            var_value['target'].update(ds_var_value['target'])
            var_value['other'] = update_dict(var_value['other'], ds_var_value['other'])

        if len(var_value['target']) <= 0:
            return vars, tr

        for varname in var_value['other']:
            st = get_attribute_numeric_status(varname, var_value['other'][varname],var_mapping)
            if st != {}:
                features.update(st)

        for varname in var_value['target']:
            value = var_value['target'][varname]
            features.update({varname: get_numeric_value('Status',value,var_mapping)})
            labels.update({varname :value})

        for trg in labels:
            feat = {ft:features[ft] for ft in features if ft != trg}
            tr.append({'features':feat, 'label':trg+':'+labels[trg]})

        return vars, tr



    # Get data items of tree data of an Id
    def __get_tree_data_by_id(self, id):
        tr = []
        features={}
        labels = {}
        vars = {}
        var_value = {'target':{},'other':{}}

        for datasource_name in self.data_items:
            ds_vars, ds_var_value = self.__get_var_values(datasource_name, id)
            vars.update(ds_vars)
            var_value['target'].update(ds_var_value['target'])
            var_value['other'] = update_dict(var_value['other'], ds_var_value['other'])


        if len(var_value['target']) <= 0:
            return vars, tr

        for varname in var_value['other']:
            st = get_attribute_numeric_status(varname, var_value['other'][varname],var_mapping)
            if st != {}:
                features.update(st)

        for varname in var_value['target']:
            value = var_value['target'][varname]
            st = get_attribute_numeric_status(varname, value, var_mapping,'Status')
            features.update(st)
            if value in self.focused_patterns:
                labels.update({varname :value})

        for trg in labels:
            feat = {ft:features[ft] for ft in features if ft != trg}
            tr.append({'features':feat, 'label':trg+':'+labels[trg]})

        return vars, tr




    # Get data items of tree data of an Id
    def __get_all_data_by_id(self, id):
        vars = {}
        var_value = {'target':{},'other':{}}

        for datasource_name in self.data_items:
            ds_vars, ds_var_value = self.__get_var_values(datasource_name, id)
            vars.update(ds_vars)
            var_value['target'].update(ds_var_value['target'])
            var_value['other'] = update_dict(var_value['other'], ds_var_value['other'])


        return vars, var_value



    # Get data items of an Id
    def __get_data_by_id(self, id):
        tr = []
        vars = {}
        var_value = {'target':{},'other':{}}

        for datasource_name in self.data_items:
            ds_vars, ds_var_value = self.__get_var_values(datasource_name, id,self.focused_patterns)
            vars.update(ds_vars)
            var_value['target'].update(ds_var_value['target'])
            var_value['other'] = update_dict(var_value['other'], ds_var_value['other'])


        if len(var_value['target']) <= 0 and len(var_value['other']) <= 0:
            return vars, tr

        for varname in var_value['target']:
            value = var_value['target'][varname]
            tr.append(varname + ':' + value)
            if varname in var_mapping:
                var_mapping[varname].update({value: var_mapping['Status'][value]})
            else:
                var_mapping[varname] = {value: var_mapping['Status'][value]}


        for varname in var_value['other']:
            if isinstance(var_value['other'][varname],list):
                values = var_value['other'][varname]
            else:
                values = [var_value['other'][varname]]
            for value in values:
                st = get_attribute_status( varname, value)
                if st != '':
                    tr.append(st)

        if tr== '':
            tr.append('Without Any resistance')

        return vars, tr



    # Fetch transactions for testing the model
    def __fetch_test_transactions(self):
        where_clause = ''
        selected_cohorts= ''

        for cohort in self.included_cohorts:
            selected_cohorts = selected_cohorts + (' ' if selected_cohorts == '' else ', ') + ' \'' + cohort + '\''

        if self.sampling_condition.strip() != '':
            where_clause = ' where (' + self.sampling_condition + ')'

        if self.included_cohorts != []  and self.included_cohorts != ['All']:
            where_clause = where_clause + (' where' if where_clause == '' else ' and') + (' cohort in (%s) ' % (selected_cohorts,))

        all_ids_df = self.data_base.get_records("""    
            select distinct %s from %s  
             %s
             order by %s
            """ % (self.id_field,self.id_entity, where_clause,self.id_field))

        selected_ids_df = all_ids_df.head(transaction_limit)

        self.test_transaction_list = []

        if len(selected_ids_df.index) == 0:
            return

        for rw in selected_ids_df.index:
            id = selected_ids_df.loc[rw, self.id_field]
            if id  in self.test_cl_num:
                var_dic, datasource_trans = self.__get_test_data_by_id(str(id))
                if datasource_trans != []:
                    self.var_dict.update(var_dic)
                    for tr in datasource_trans:
                        self.test_transaction_list.append(tr)
        pass



    # Fetch numeric transactions for tree model
    def __fetch_tree_transactions(self):
        where_clause = ''
        selected_cohorts= ''
        self.tree_transaction_list = []

        if not self.Dtree_model:
            return

        if self.sampling_condition.strip() != '':
            where_clause = ' where (' + self.sampling_condition + ')'

        for cohort in self.included_cohorts:
            selected_cohorts = selected_cohorts + (' ' if selected_cohorts == '' else ', ') + ' \'' + cohort + '\''

        if self.included_cohorts != []  and self.included_cohorts != ['All']:
            where_clause = where_clause + (' where' if where_clause == '' else ' and') + (' cohort in (%s) ' % (selected_cohorts,))

        all_ids_df = self.data_base.get_records("""    
            select distinct %s from %s  
             %s
             order by %s
            """ % (self.id_field,self.id_entity, where_clause,self.id_field))

        selected_ids_df = all_ids_df.head(transaction_limit)

        if len(selected_ids_df.index) == 0:
            return

        for rw in selected_ids_df.index:
            id = str(selected_ids_df.loc[rw, self.id_field])
            var_dic, datasource_trans = self.__get_tree_data_by_id(id)
            if datasource_trans != []:
                for tr in datasource_trans:
                    self.tree_transaction_list.append(tr)

        pass


    def __fetch_transactions_with_complementary_data(self):
        where_clause = ''
        selected_cohorts= ''
        self.complete_transaction_with_complementary_data_list = []

        if self.complementary_data == {}:
            return

        if self.sampling_condition.strip() != '':
            where_clause = ' where (' + self.sampling_condition + ')'

        for cohort in self.included_cohorts:
            selected_cohorts = selected_cohorts + (' ' if selected_cohorts == '' else ', ') + ' \'' + cohort + '\''

        if self.included_cohorts != [] and self.included_cohorts != ['All']:
            where_clause = where_clause + (' where' if where_clause == '' else ' and') + (' cohort in (%s) ' % (selected_cohorts,))

        all_ids_df = self.data_base.get_records("""    
            select distinct %s from %s  
             %s
             order by %s
            """ % (self.id_field,self.id_entity, where_clause,self.id_field))

        selected_ids_df = all_ids_df.head(transaction_limit)



        if len(selected_ids_df.index) == 0:
            return

        old_data_items = self.data_items
        self.data_items = self.data_items | self.complementary_data

        for rw in selected_ids_df.index:
            id = selected_ids_df.loc[rw, self.id_field]
            if id not in self.test_cl_num:
                var_dic, transaction = self.__get_all_data_by_id(str(id))
                if transaction['target'] != {} :
                    self.complete_transaction_with_complementary_data_list.append(transaction)

        self.data_items = old_data_items
        pass


    def __fetch_compelete_transactions(self):
        where_clause = ''
        selected_cohorts= ''

        if self.sampling_condition.strip() != '':
            where_clause = ' where (' + self.sampling_condition + ')'

        for cohort in self.included_cohorts:
            selected_cohorts = selected_cohorts + (' ' if selected_cohorts == '' else ', ') + ' \'' + cohort + '\''

        if self.included_cohorts != [] and self.included_cohorts != ['All']:
            where_clause = where_clause + (' where' if where_clause == '' else ' and') + (' cohort in (%s) ' % (selected_cohorts,))

        all_ids_df = self.data_base.get_records("""    
            select distinct %s from %s  
             %s
             order by %s
            """ % (self.id_field,self.id_entity, where_clause,self.id_field))

        selected_ids_df = all_ids_df.head(transaction_limit)

        self.complete_transaction_list = []

        if len(selected_ids_df.index) == 0:
            return

        for rw in selected_ids_df.index:
            id = selected_ids_df.loc[rw, self.id_field]
            if id not in self.test_cl_num:
                var_dic, transaction = self.__get_all_data_by_id(str(id))
                if transaction['target'] != {} :
                    self.complete_transaction_list.append(transaction)

        print_with_timestamp(str(len(self.complete_transaction_list)) + ' transactions are retrieved')
        pass



    # Fetch transactions for training the model
    def __fetch_transactions(self):
        where_clause = ''
        selected_cohorts= ''

        if self.sampling_condition.strip() != '':
            where_clause = ' where (' + self.sampling_condition + ')'

        for cohort in self.included_cohorts:
            selected_cohorts = selected_cohorts + (' ' if selected_cohorts == '' else ', ') + ' \'' + cohort + '\''

        if self.included_cohorts != []  and self.included_cohorts != ['All']:
            where_clause = where_clause + (' where' if where_clause == '' else ' and') + (' cohort in (%s) ' % (selected_cohorts,))

        all_ids_df = self.data_base.get_records("""    
            select distinct %s from %s  
             %s
             order by %s
            """ % (self.id_field,self.id_entity, where_clause,self.id_field))

        selected_ids_df = all_ids_df.head(transaction_limit)

        self.transaction_list = []
        self.max_trans_len = 1

        if len(selected_ids_df.index) == 0:
            return

        for rw in selected_ids_df.index:
            transaction = []
            id = selected_ids_df.loc[rw, self.id_field]
            if id not in self.test_cl_num:
                var_dic, datasource_trans = self.__get_data_by_id(str(id))
                if datasource_trans != [] :
                    transaction.extend(datasource_trans)
                    self.var_dict.update(var_dic)
                    self.transaction_list.append(transaction)
                    if len(transaction) > self.max_trans_len:
                        self.max_trans_len = len(transaction)
                    if len(self.transaction_list) > transaction_limit:
                        break

        pass



    # Fetch all transactions and apply empty cell threshold and transaction limit parameters
    def fetch_data(self):
        self.__split_test_train()
        self.__fetch_transactions()
        self.__fetch_test_transactions()
        self.__fetch_tree_transactions()
        self.__fetch_compelete_transactions()
        self.__fetch_transactions_with_complementary_data()
        self.complete_transaction_with_complementary_data_list = self.complete_transaction_list if self.complete_transaction_with_complementary_data_list == [] else self.complete_transaction_with_complementary_data_list
        included_classes = self.__get_classes()
        selected_trans = []
        for tran in self.transaction_list:
            included = True
            completed_celss = len(tran)
            completed_cell_ratio = completed_celss / self.max_trans_len
            if (completed_cell_ratio + empty_cell_threshold) < 1:
                included = False

            if included:
                selected_trans.append(tran)

        if len(selected_trans) > transaction_limit:
            return selected_trans[0:transaction_limit],self.test_transaction_list,self.tree_transaction_list,self.complete_transaction_list,self.complete_transaction_with_complementary_data_list,included_classes,self.var_dict
        else:
            return selected_trans,self.test_transaction_list,self.tree_transaction_list,self.complete_transaction_list,self.complete_transaction_with_complementary_data_list,included_classes,self.var_dict


    def __get_classes(self):
        attributes_df = self.data_base.get_records("""
           select  distinct Drug,Class 
           from %s 
            where  class is not null
            order by Class
            """%(self.id_entity,))

        if len(attributes_df.index) <= 0:
            return []

        return { itm['Drug']:itm['Class'] for itm in attributes_df[['Drug', 'Class']].to_dict('records')}




##########################################
##Functions
##########################################


#Returns a dictionary of a variable and it's numeric value (fld is used when the name of field is different from name of variable)
def get_attribute_numeric_status(var, val, varmap,fld=None):
    if is_empty(val):
        return {}
    if isinstance(val,list):
        return categorial_to_binary_variables(var,val)
    if fld:
        numeric_val = get_numeric_value(fld, val, varmap)
    else:
        numeric_val = get_numeric_value(var, val, varmap)
    if numeric_val == None:
        return ({var +'('+str(val)+')': 1})
    return {var: numeric_val}


#Returns a string of a variable and it's value
def get_attribute_status(var, val):
    if is_empty(val):
        return ''
    return var + ':' + str(val)


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
def handel_null_values(transaction_list,label_list,feature_names,test):
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
        if null_ratio <= tree_null_thr or test:
            final_feature_names.append(ft)
            imputed_by_feature[ft] = get_imputed_by_label(transaction_list,label_list,ft)

    for ind in range(len(transaction_list)):
        tr = transaction_list[ind]
        lb = label_list[ind]
        new_tr = {ft:tr[ft] if ft in tr and tr[ft]!=None else imputed_by_feature[ft][lb] for ft in final_feature_names }
        final_transaction_list.append(new_tr)

    return final_transaction_list,final_feature_names


#Change a name based on a dictionary of names and substitues
def change_name(name,dict):
    for ky in dict:
        name = name.replace(ky,dict[ky])
    return name



#Convert categorial variable to several binary variables
def categorial_to_binary_variables(varname,list_of_values,list_of_all_posible_values=None):
    new_var_value = {}
    for val in list_of_values:
        if val not in new_var_value:
            new_var_value[varname +'('+str(val)+')'] = 1

    if list_of_all_posible_values!=None:
        for val in list_of_all_posible_values:
            if val not in new_var_value:
                new_var_value[varname +'('+str(val)+')'] = 0

    return new_var_value














