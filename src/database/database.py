import mysql.connector

class Database:
    '''
        Class for connecting to and disconnecting from a MySQL database.
    '''
    def __init__(self, host:str, port:int, user:str, password:str, database:str):
        '''
            Initializes a Database instance with the specified connection details.

            Args:
                host (str): The hostname or IP address of the MySQL server.
                port (int): The port number for the MySQL server.
                user (str): The username for authenticating the connection.
                password (str): The password for authenticating the connection.
                database (str): The name of the MySQL database.
        '''
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database

    def connect(self) -> bool:
        '''
            Establishes a connection to the MySQL database.
        '''
        try:
            self._db = mysql.connector.connect(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password
            )
            self._cursor = self._db.cursor()
            print(f'Successfully connected to {self._database} database.')
            
            return True

        except mysql.connector.Error as error:
            print(f'Error while connecting to {self._database} database: {error}')

        return False

    def disconnect(self):
        '''
            Disconnects from the MySQL database.
        '''
        if self._db:
            self._cursor.close()
            self._db.close()
            print(f'Successfully disconnected from {self._database} database.')