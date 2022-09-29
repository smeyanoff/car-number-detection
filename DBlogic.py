import time

def make_commit_to_db(db_connection, values_list):

    """
    Fuction makes commit to DB.
        db_connection (sqlite3.connect)
        values_list (list of tuples) - list of values to commit 
    """
    c = db_connection.cursor()

    for values in values_list:
        c.execute(f"""
            insert into car_enters_check 
            values (
                ?,
                '{
                    time.strftime('%Y-%m-%d %H:%M:%S',
                    time.localtime())
                    }',
                ?,
                ?
                )""", 
                values)
        
    db_connection.commit()


