from .database import Database

from typing import List
from typing import Tuple

class BreastCancerDB:
    def __init__(self, host:str, port:int, user:str, password:str, database:str):
        self.__db = Database(host=host, port=port, user=user, password=password, database=database)

    def selectAll(self) -> List[Tuple]:
        '''
            Executes a SELECT query on the 'breast-cancer' table and retrieves all rows.

            Returns:
                    A list of tuples containing the fetched rows.
        '''
        query = 'SELECT * FROM `breast-cancer`'

        isConnected = self.__db.connect()

        if isConnected:
            try:
                self.__db._cursor.execute(query)
                results = self.__db._cursor.fetchall()
                
                if len(results) > 0:
                    print('Select query executed successfully. Results found.')
                else:
                    print('Select query executed successfully. No results found.')
                
                return results
            except Exception as error:
                print(f'Error executing select query: {str(error)}')
            finally:
                self.__db.disconnect()
        else:
            print('Unable to connect to the database.')
            
        return []
    
    def selectColumnNames(self) -> List[Tuple]:
        '''
            Executes a SELECT query on the 'breast-cancer' table and retrieves all rows.

            Returns:
                    A list of tuples containing the fetched rows.
        '''
        query = 'SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = \'breast-cancer\' ORDER BY ORDINAL_POSITION'

        isConnected = self.__db.connect()

        if isConnected:
            try:
                self.__db._cursor.execute(query)
                results = self.__db._cursor.fetchall()
                
                if len(results) > 0:
                    print('Select query executed successfully. Results found.')
                else:
                    print('Select query executed successfully. No results found.')
                
                return results
            except Exception as error:
                print(f'Error executing select query: {str(error)}')
            finally:
                self.__db.disconnect()
        else:
            print('Unable to connect to the database.')
            
        return []



                


