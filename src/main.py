from database.bc_database import BreastCancerDB

def main():
    print('Succesfully started!')
    
    host = '127.0.0.1'
    port = 3306
    user = 'root'
    password = 'root'
    database = 'ag002'

    bcDatabase = BreastCancerDB(host=host, port=port, user=user, password=password, database=database)

    results = bcDatabase.selectAll()
    print(results)

    names = bcDatabase.selectColumnNames()
    print(names)

if __name__ == '__main__':
    main()