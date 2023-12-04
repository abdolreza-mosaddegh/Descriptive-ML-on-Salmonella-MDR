##########################################
#Handeling Configuration of Association Mining
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Email: am2685@cornell.edu
#Create Date: 2022_10_18
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_04_11
##########################################

import yaml
import os.path as path
from Common_functions import *

##########################################
##Parameters
##########################################
default_config_file_type  = 'YAML'
default_config_file_title = 'default_config'
default_mandatory_tags    = ['thresholds','transactions','rules','models','database']

##########################################
##Class for Setting Configurations
##########################################

class rule_mining_config:
    # Constructor
    def __init__(self, default_file_title=default_config_file_title,
                       config_file_type=default_config_file_type,
                       mandatory_tags=default_mandatory_tags):
        self.config_file_type = config_file_type
        self.default_config_file = default_file_title  + '.' + config_file_type
        self.mandatory_config_tags = mandatory_tags
        pass


    # Retrieve Configurations from YAML file
    def __get_configurations_from_file(self):
        config_file = get_file_name(
                      input_message='Enter Name of Configuration File or Press Enter for Default Configuration File (%s):' %(
                                     self.default_config_file),
                      file_type=self.config_file_type ,
                      default_file=self.default_config_file)

        if config_file:
            with open(config_file, 'r') as file:
                self.configs = yaml.safe_load(file)
                file.close()


    # validate config
    def __valid_config(self,config):
        for tag in self.mandatory_config_tags:
            if tag not in self.configs[config]:
                return False
        return True


    # validate all configs
    def valid_configs(self):
        for config in self.configs:
            if not self.__valid_config(config):
                return False
        return True


    # get all configs
    def get_configurations(self):
        self.__get_configurations_from_file()
        if not self.valid_configs():
            raise Exception('Invalid format of the config file')
        return self.configs


##########################################
##Functions
##########################################

# Get an input filename
def get_file_name(input_message, file_type, default_file = None):
    input_file = input(input_message).strip()
    input_file = default_file if input_file == None or input_file == '' else input_file
    file_extention = '.' + file_type
    if input_file.replace(file_extention, '') == input_file:
        input_file = input_file + file_extention
    if not path.isfile(input_file):
        input_file = path.dirname(__file__) + '\\' + input_file
        if not path.isfile(input_file):
            raise Exception(file_type + ' file does not exists in the path containing Python code')
    return input_file


# print specifications of a config
def print_config_spec(config, publish_tags=default_mandatory_tags):
    print('')
    for sub_config in config:
        if sub_config in publish_tags:
            print(sub_config + ' settings:')
            print(dict_to_str(config[sub_config]))
            print('')
