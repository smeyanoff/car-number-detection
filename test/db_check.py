import sqlite3

import settings

conn = sqlite3.connect(settings.DATABASE)
c = conn.cursor()

c.execute(
    """
    select *
    from car_enters_check
    """
)

for row in c:
    print(row)

c.close()
