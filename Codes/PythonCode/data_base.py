##########################################
#Handeling Database interactions
#Owner: Cazer Lab - Cornell University
#Creator: Abdolreza Mosaddegh (ORCID: 0000-0001-5840-3628)
#Email: am2685@cornell.edu
#Create Date: 2022_10_18
#Modified by: Abdolreza Mosaddegh
#Last Update Date: 2023_04_11
##########################################

import pandas
import pyodbc

##########################################
##Class for Database interactions
##########################################

class data_base:

    # Constructor: set a connection and cursor to DB using input connection string
    def __init__(self, con_str):
        try:
            self.db_connection = pyodbc.connect(con_str)
            self.db_cursor = self.db_connection.cursor()
            print('Connected to DB')
        except Exception as e:
            raise Exception('Error in Connecting to DB: ' + str(e))
        pass

    # get a dataframe by executing the input sql instruction
    def get_records(self, sql_str):
        return pandas.read_sql(sql_str, self.db_connection)

    # execute the input sql instruction without commit
    def execute_sql(self, sql_str):
        self.db_cursor.execute(sql_str)
        pass

    # commit all DB transactions
    def commit_transactions(self):
        self.db_connection.commit()
        pass

    # close connection to DB
    def close_connection(self):
        self.db_connection.close()
        pass



