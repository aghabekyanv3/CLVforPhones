import sqlite3
import logging
import pandas as pd
import numpy as np
import csv
logging.basicConfig(level=logging.DEBUG)

class etl:

    def __init__(self):
        pass
        # self.cnxn = sqlite3.cnxnect('clv.db')
        # self.cursor = self.cnxn.cursor()

    def create_sql_table(self, table_name:str,columns:list):
        """
        Create an SQLite table with the specified name and columns if it does not already exist.

        Parameters:
        -----------
        table_name : str
            The name of the table to be created.

        columns : list
            A list of column names and their data types in the format "column_name data_type".

        Returns:
        --------
        None
        """
        query=f"""  
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
            {', '.join(columns)}
                                );
                """
        logging.info(query)
        cnxn = sqlite3.connect('clv.db')
        cursor = cnxn.cursor()
        cursor.execute(query)

        logging.info(f'the {table_name} is created')
        cursor.close()
        cnxn.close()
    
    def row_insert(self, row):
        """
        Insert a single row of data into an SQLite table named 'persons'.

        Parameters:
        -----------
        row : str
            The data for the row to be inserted, formatted as a string.

        Returns:
        --------
        None
        """
        cnxn = sqlite3.cnxnect('clv.db')
        cursor = cnxn.cursor()
        query = f""""
        INSERT INTO persons VALUES                   
        ({row}])
        """
        logging.info(query)
        cursor.execute(query)
        cnxn.commit()

        logging.info(f'the row {row} has been added')
        cursor.close()
        cnxn.close()

    def bulk_insert(self, df: pd.DataFrame, table_name: str, chunk_size: int = 1000) -> str:
        """
        Bulk insert data from a Pandas DataFrame into a SQLite table.

        Parameters:
        ----------
        df : pandas.DataFrame
            The DataFrame containing the data to be inserted.

        table_name : str
            The name of the table in the SQLite database where data will be inserted.

        chunk_size : int, optional
            The size of each data chunk for batch insertion. Default is 1000.

        Returns:
        -------
        str
            A message indicating the completion of the data insertion process.
        """
        cnxn = sqlite3.connect('clv.db')
        cursor = cnxn.cursor()
        
        df = df.replace(np.nan, None)  # for handling NULLs
        df.rename(columns=lambda x: x.lower(), inplace=True)
        
        columns = list(df.columns)
        sql_column_names = [i.lower() for i in self.get_table_columns(table_name=table_name)]
        columns = list(set(columns) & set(sql_column_names))
        ncolumns = list(len(columns) * '?')
        data_to_insert = df.loc[:, columns]
        
        if len(columns) > 1:
            cols, params = ', '.join(columns), ', '.join(ncolumns)
        else:
            cols, params = columns[0], ncolumns[0]
        
        values = [tuple(i) for i in data_to_insert.values]
        
        logging.info(f'the shape of the table which is going to be imported {df.shape}')
        logging.info(f'insert structure: colnames: {cols} params: {params}')
        
        logging.warning('Starting data insertion in chunks...')
        total_rows = len(values)
        inserted_rows = 0
        
        while inserted_rows < total_rows:
            chunk = values[inserted_rows:inserted_rows + chunk_size]
            query = f"""INSERT INTO {table_name} ({cols}) VALUES ({params});"""
            cursor.executemany(query, chunk)
            cnxn.commit()
            inserted_rows += len(chunk)
        
        cursor.close()
        cnxn.close()
        logging.warning('Data insertion completed')

    
    def trunctate_table(self, table_name):
        """
        Truncate (delete all rows) an SQLite table.

        Parameters:
        -----------
        table_name : str
            The name of the table to be truncated.

        Returns:
        --------
        None
        """
        cnxn = sqlite3.connect('clv.db')
        cursor = cnxn.cursor()

        truncate_query = f'DELETE FROM {table_name};'
        cursor.execute(truncate_query)

        cnxn.commit()

        logging.info(truncate_query)
        cursor.execute(truncate_query)
        cnxn.commit()

        logging.info(f'the table has been truncated')
        cursor.close()
        cnxn.close()

    def get_table_columns(self,table_name:str):
        """
        Retrieve column names of an SQLite table.

        Parameters:
        -----------
        table_name : str
            The name of the table for which to retrieve column names.

        Returns:
        --------
        list
            A list of column names in the specified table.
        """
        cnxn = sqlite3.connect('clv.db')
        cursor = cnxn.cursor()
        
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        column_names = [col[1] for col in columns]
        
        cursor.close()
        cnxn.close()
        
        return column_names


    def load(self, id_number, table_name):
        """
        Load data from an SQLite table based on the provided ID number.

        Parameters:
        -----------
        id_number : int
            The ID number to use for loading data.

        table_name : str
            The name of the table from which to load data.

        Returns:
        --------
        None
        """
        cnxn = sqlite3.cnxnect('clv.db')
        cursor = cnxn.cursor()
        load_query = (f"""
        SELECT * FROM {table_name} 
        WHERE id_number = ?""")
        
        results = self.cursor.fetchone()

        cursor.execute(load_query)

        cnxn.commit()

        logging.info(load_query)
        cursor.execute(load_query)
        cnxn.commit()

        logging.info(f'loading row {results}') 
        cnxn.close()

        self.id_number = id_number

    def from_sql_to_local(chunksize:int, id_value:str,dbname='clv.db',tablename='transactions') -> object:
        """

        Parameters
        ----------
        chunksize : int
            the chunksize for the extract
        id_value : char
            the values by which it should be sorted, 'MSISDN, DATE_ID'
        dbname : char
            (Default value = 'CVM')
        tablename : char
            (Default value = 'DIM_ARMENIA_MSISDNS')
        chunksize:int :
            
     
        Returns
        -------

        """
        
        offset=0
        dfs=[]
        cnxn = sqlite3.connect('clv.db')
        cursor = cnxn.cursor()
        
        while True:
            query=f"""
            SELECT * FROM [{dbname}].[{tablename}]
                ORDER BY {id_value}
                OFFSET  {offset}  ROWS
                FETCH NEXT {chunksize} ROWS ONLY  
            """
            data = pd.read_sql_query(query,cnxn, parse_dates=['RELEASE_DATE','START_DATE', 'Min_END_DATE', 'Max_END_DATE']) #dtype={'MSISDN':str})
            logging.info(f'the shape of the chunk: {data.shape}')
            dfs.append(data)
            offset += chunksize
            if len(dfs[-1]) < chunksize:
                logging.warning('loading the data from SQL is finished')
                cursor.close()
                cnxn.close()
                logging.debug('connection is closed')
                break
        df = pd.concat(dfs)

        return df
         

    def delete_table(self,table_name):
        """
        Truncate (delete all rows) an SQLite table while preserving its structure.

        Parameters:
        -----------
        table_name : str
            The name of the table to be truncated.

        Returns:
        --------
        None
        """
        cnxn = sqlite3.connect('clv.db')

        cursor = cnxn.cursor()

        delete_query = f"DROP TABLE IF EXISTS {table_name};"
        logging.info(delete_query)

        cursor.execute(delete_query)

        cnxn.commit()

        logging.info(f"table '{table_name}' deleted.")
        cnxn.close()
    
    def insert_data_from_csv(self, table_name, csv_file):
        """
        Insert data from a CSV file into an SQLite table.

        Parameters:
        -----------
        table_name : str
            The name of the table where data will be inserted.

        csv_file : str
            The path to the CSV file containing the data to be inserted.

        Returns:
        --------
        None
        """
        cnxn = sqlite3.connect('clv.db')
        cursor = cnxn.cursor()

        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)

            columns_str = ", ".join(next(csv_reader))
            placeholders = ", ".join(['?'] * len(columns_str.split(',')))
            insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            logging.info(insert_query)

            data_to_insert = [tuple(row) for row in csv_reader]
            cursor.executemany(insert_query, data_to_insert)

        cnxn.commit()
        logging.info(f"csv values from {csv_file} inserted to '{table_name}'.")
        cnxn.close()

    def from_sql_to_local(chunksize:int, id_value:str,dbname='clv.db',tablename='transactions') -> object:
        """

        Parameters
        ----------
        chunksize : int
            the chunksize for the extract
        id_value : char
            the values by which it should be sorted, 'MSISDN, DATE_ID'
        dbname : char
            (Default value = 'CVM')
        tablename : char
            (Default value = 'DIM_ARMENIA_MSISDNS')
        chunksize:int :
            
     
        Returns
        -------

        """
        
        offset=0
        dfs=[]
        cnxn = sqlite3.connect('clv.db')

        cursor = cnxn.cursor()

        
        while True:
            query=f"""
            SELECT * FROM {tablename}
                ORDER BY {id_value}
                OFFSET  {offset}  ROWS
                FETCH NEXT {chunksize} ROWS ONLY  
            """
            data = pd.read_sql_query(query,cnxn, parse_dates=['RELEASE_DATE','START_DATE', 'Min_END_DATE', 'Max_END_DATE']) #dtype={'MSISDN':str})
            logging.info(f'the shape of the chunk: {data.shape}')
            dfs.append(data)
            offset += chunksize
            if len(dfs[-1]) < chunksize:
                logging.warning('loading the data from SQL is finished')
                cursor.close()
                cnxn.close()
                logging.debug('connection is closed')
                break
        df = pd.concat(dfs)

        return df

if __name__ == "__main__":
    pass

