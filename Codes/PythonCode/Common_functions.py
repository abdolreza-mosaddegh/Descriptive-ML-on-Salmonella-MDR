##########################################
#Common Functions for Association Rule Mining
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Email: am2685@cornell.edu
#Create Date: 2022_10_18
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_04_11
##########################################

import re
import pandas
import datetime

##########################################
##Global Variables
##########################################

#Mapping of ordinal variables to numbers
var_mapping = {
 'Gender':{'Female':1,'Male':2}
,'AgeGroup':{'Child':1,'Adult':18,'Senior':65}
,'Status':{'susceptible':1,'intermediate':2,'SSD':2,'resistant':3}
}

##########################################
##Functions
##########################################

#Print input message with timestamp of current time
def print_with_timestamp(message_str):
    print('')
    print(message_str + ': ' + datetime.datetime.now().strftime("%Y_%m_%d (%H:%M:%S)"))


#Convert an input list to a string
def list_to_str(list_of_values,exclude_brackets=False, exclude_cot=True):
    list_str = str(list(list_of_values))
    if exclude_cot:
        list_str =  list_str.replace('\'', '')
    if exclude_brackets:
        list_str =  list_str.replace('[','').replace(']','')
    return list_str


#Convert an input dictionary to a string
def dict_to_str(a_dict,exclude_brackets=True):
    dct = dict(a_dict)
    if exclude_brackets:
        return str(dct).replace('\'', '').replace('{','').replace('}','')
    else:
        return str(dct).replace('\'','')

#Returns name of a variable in an input string using a var-entity dictionary
def get_attribute_name(statement,var_map):
    att_name = re.sub(":.*", "", statement)
    correlation = statement.replace(att_name, '').replace(': ', '').replace(':', '').replace('\'', '').replace('\"', '')
    datasource_name = (var_map.get(att_name) or '')
    return correlation, datasource_name, att_name


#Change a numeric string to a number
def str_to_number(val):
    if not is_numeric(val):
        return None
    if int(float(val))==float(val):
        return int(float(val))
    return float(val)


#Check if the value is numeric
def is_numeric(val):
    if str(val).replace('.', '').replace('-', '').isnumeric():
        return True
    return False


#Test an input against Blank and Null values
def is_empty(val):
    if val == None:
        return True
    if isinstance(val, list):
        if val == []:
            return True
        return False
    if isinstance(val, dict):
        if val == {}:
            return True
        return False
    if pandas.isna(val):
        return True
    if str(val) == '' or str(val) == 'na' or str(val) == 'NA':
        return True
    return False


#Update a dictionary by adding a new dictionary
def update_dict(main_dic, new_dic):
    if new_dic == {}:
        return main_dic
    for varname in new_dic:
        if varname not in main_dic:
            main_dic[varname] = new_dic[varname]
        else:
            if not isinstance(main_dic[varname], list):
                main_dic[varname] = [main_dic[varname]]
            if not isinstance(new_dic[varname], list):
                new_dic[varname] = [new_dic[varname]]
            main_dic[varname] = list(set(main_dic[varname]).union( set(new_dic[varname])))
            if len(main_dic[varname]) == 1:
                main_dic[varname] = main_dic[varname][0]
    return main_dic


#Get value of a parameter in a configuration dictionary
def get_param_value(config,param, dafault_value = None, mandatory = False):
    for sub_config in config:
        if param in config[sub_config]:
            return config[sub_config][param]
    if dafault_value != None:
        return dafault_value
    if mandatory:
        raise Exception(param + ' Parameter is not Existed in Configuration')
    return

#Returns numeric value of a categorial attribute
def get_numeric_value(var,val,value_map):
    if is_numeric(val):
        return str_to_number(val)
    if var in value_map:
        values = value_map[var]
        if val in values:
            return str_to_number(values[val])
    return None
