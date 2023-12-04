##########################################
#Association Mining of Microbe Resistance
#using Paralell FP_Growth on SPARK Platform
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Email: am2685@cornell.edu
#Create Date: 2022_10_18
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_04_11
##########################################

##########################################
##Python Libraries
##########################################
import pandas
import datetime
import pyspark.sql as spl
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import *
from sklearn.metrics import accuracy_score

##########################################
##Project Libraries
##########################################
from Common_functions import *
from rule_mining_config import *
from transaction_set import *
from rule_graph import *
from rule_set import *
from data_base import *
from dtree import *
from agnes import *
from network_analysis import *

##########################################
##Spark Parameters
##########################################
driver_memory= '54g'        # Memory dedicated to Spark driver nodes (For large datasets or numerous features, most available memory should be dedicated to the drivers)
# executor_memory = '8g'    # Memory dedicated to Spark executor nodes (If not set the system default is used)
# number_of_nodes = 30      # Number of cores dedicated to Spark (If not set the maximum available cores are used)

##########################################
##Classes
##########################################

##Class of association rules model
class _association_rules:

    # Constructor of association_rules class
    def __init__(self, title, config, spark):
        self.spark = spark
        self.title = title
        self.config = config
        db_connection_string = get_param_value(config, 'db_connection_string', mandatory=True)
        self.data_base = data_base(db_connection_string)


    # Destructor
    def __del__(self):
        self.data_base.close_connection()


    #Create PFP-Growth Moddel
    def __create_pfp_model(self,selected_trans):
        self.train_pfp = spark_df_from_list(selected_trans, self.spark)
        self.pfp_model = FPGrowth(itemsCol="items",
                                  predictionCol="prediction",
                                  minSupport=get_param_value(self.config,'support_threshold', mandatory = True),
                                  minConfidence=get_param_value(self.config,'confidence_threshold', mandatory = True)
                                  ).fit(self.train_pfp)
        self.associationRules = self.pfp_model.associationRules
        print_with_timestamp('Data is fitted to the model')


    # Presents accuracy of an ML model on test and train data
    def __test_model(self, model, train_ft,train_lb, test_ft,test_lb, model_name= ''):
        acuracy_train = test_model(model,train_ft,train_lb)
        acuracy_test = test_model(model,test_ft,test_lb)
        print('Model %s Accuracy on Train Data is %s and on Test Data is %s'%(
               model_name,acuracy_train,acuracy_test))


    #Mining associations of data items
    def find_associations(self,included_cohorts):
        optional_param = lambda param_name,def_val: get_param_value(self.config, param_name, dafault_value=def_val)
        mandatory_param = lambda param_name: get_param_value(self.config, param_name, mandatory=True)

        cohort_title = list_to_str(included_cohorts, exclude_brackets=True)
        mining_title = (self.title + ' ' + cohort_title).strip()
        data_source_list = list(mandatory_param('data_sources').keys())
        print_with_timestamp(list_to_str(data_source_list) + ' datasources group process is started ' + cohort_title)

        ##Fetch Transactions from DB
        selected_trans, test_trans, tree_trans, complete_trans, complementary_trans, classes, var_dic = transaction_set(
                                                 included_cohorts,
                                                 mandatory_param('data_sources'),
                                                 mandatory_param('correlation_target'),
                                                 mandatory_param('focused_patterns'),
                                                 mandatory_param('trans_id'),
                                                 self.data_base,
                                                 optional_param('excluded_vars', []),
                                                 optional_param('sampling_condition', ''),
                                                 optional_param('test_data_condition',''),
                                                 optional_param('test_data_percentage',0),
                                                 optional_param('Dtree_model', False),
                                                 optional_param('complementary_data', {})
                                                 ).fetch_data()

        transaction_number = len(selected_trans)
        print_with_timestamp(str(transaction_number) + ' transactions are selected')

        if transaction_number < min_transaction:
            print('Rule mining ignored due to low transaction number')
        else:
            ##Finding the frequent patterns having interestingness measures above the thresholds
            self.__create_pfp_model(selected_trans)

            ##Record extracted rules
            recorded_rules = rule_set(included_cohorts,
                                      self.associationRules,
                                      self.config,
                                      transaction_number,
                                      complete_trans,
                                      var_dic,
                                      self.data_base,
                                      optional_param('pruning_rules', False),
                                      optional_param('apply_complementary_support' , False),
                                      optional_param('only_one_to_one_association', False),
                                      optional_param('combine_rules', False),
                                      optional_param('association_antecedent', {}),
                                      optional_param('association_consequent', {}),
                                      optional_param('clean_previous_rules', False),
                                      optional_param('record_in_database', True)
                                     ).record_rules()

            ##Visualize rules using graphs
            if len(recorded_rules)>0:
                rule_graph(mining_title,
                           recorded_rules,
                           var_dic,
                           classes,
                           optional_param('graph_types', ['3dScatter','Chord']),
                           optional_param('graph_parameters',{}),
                           optional_param('rules_to_visualize_cond', {}),
                           optional_param('hidden_vars_in_graph', []),
                           optional_param('grouping_conditions', {}),
                           optional_param('rule_weight_factor','lift'),
                           optional_param('chart_folder','charts')
                           ).visualize_rules()

            ##Decision Tree model
            if optional_param('Dtree_model', False):
                dtree(tree_trans,
                      list(var_dic.keys()),
                      optional_param('chart_folder', 'charts'),
                      'Dtree_' + mining_title,
                      test_trans
                      ).draw_dtree()

            ##Hierarchical Clustering model
            if optional_param('Agnes_model', False):
                agnes(complementary_trans,
                      var_dic,
                      self.data_base,
                      data_source_list,
                      optional_param('sampling_condition', ''),
                      optional_param('agnes_target_attributes', []),
                      optional_param('agnes_analytical_attribute', None),
                      optional_param('chart_folder', 'charts'),
                      'Agnes_' + mining_title,
                ).clustering()

        ##Network Analysis model
        if optional_param('Network_model', False):
            network(complete_trans,
                    self.data_base,
                    mandatory_param('data_sources'),
                    classes,
                    optional_param('sampling_condition', ''),
                    optional_param('clean_previous_net_analysis', False),
                    optional_param('chart_folder', 'charts'),
                    'Network_' + mining_title,
                    ).analyze_network()

        print_with_timestamp(list_to_str(data_source_list) + ' datasources group is processed')




##Main Class for Parallel FP-Growth using Spark
class spark_fp_growth:

    # Constructor
    def __init__(self):
        ##Get configurations
        self.configs = rule_mining_config().get_configurations()
        print_with_timestamp('Configurations are retrieved')

        ##Initialize Spark sesssion
        self.spark = spl.SparkSession.builder.appName('FPGrowth').config('spark.driver.memory', driver_memory).getOrCreate()
        print_with_timestamp('Spark started')


    # Destructor
    def __del__(self):
        self.spark.catalog.clearCache()
        self.spark.stop()


    #Find association rules of a config
    def __association_rules_by_config(self, config_name):
        config = self.configs[config_name]
        self.rule_by_cohort = get_param_value(config,'rule_by_cohort', False)
        included_cohorts = get_param_value(config, 'cohort_group', dafault_value=['All'])
        association_rules_mining = _association_rules(config_name, config, self.spark)
        if self.rule_by_cohort:
            excluded_vars.append('Cohort')
            for cohort in included_cohorts:
                print('Cohort: ' + cohort)
                association_rules_mining.find_associations([cohort])
        else:  # for all cohorts
            association_rules_mining.find_associations(included_cohorts)


    #Find association rules of all configs
    def find_association_rules(self):
        for config_name in self.configs:
            print_with_timestamp('Start of Association Mining using following configuration [' + config_name + ' config] ')
            print_config_spec(self.configs[config_name])
            self.__association_rules_by_config(config_name)
            print_with_timestamp('End of Association Mining of ' + config_name + ' config')


##########################################
##Functions
##########################################

#Convert an input item list to a Spark Dataframe
def spark_df_from_list(input_list,spark_inst):
    pdd = pandas.DataFrame(columns=['id', 'items'], index=range(len(input_list)))
    for rw in range(len(input_list)):
        tran = list(dict.fromkeys(input_list[rw]))
        pdd.iloc[rw, 1] = tran
        pdd.iloc[rw, 0] = rw
    spd = spark_inst.createDataFrame(pdd)
    spd.cache()
    return spd


# Get accuracy of an ML model
def test_model(model,features,labels):
    predicted_labels = model.predict(features)
    acuracy = accuracy_score(labels, predicted_labels)
    return acuracy


##########################################
##Main
##########################################
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print_with_timestamp('Start of rule mining process')
    warnings.filterwarnings("ignore")

    ##PFP_Growth association mining using spark
    spark_fp_growth().find_association_rules()

    print_with_timestamp('End of process')
    end_time = datetime.datetime.now()
    print('Proccsseing Time: ' + str(end_time - start_time))
