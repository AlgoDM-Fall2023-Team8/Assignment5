import snowflake.connector
import json

# Read the JSON data from the file
with open('data.json', 'r') as file:
    data = json.load(file)

# Snowflake connection parameters
account = 'mcxbspr-vub48940'
user = 'thechainsmokers'
password = 'Thechainsmokers123'
warehouse = 'COMPUTE_WH'
database = 'IMAGES'
schema = 'Public'

# Establish connection to Snowflake
conn = snowflake.connector.connect(
    user=user,
    password=password,
    account=account,
    warehouse=warehouse,
    database=database,
    schema=schema
)

# Create a Snowflake cursor
cur = conn.cursor()

# SQL statement to create the table
create_table_sql = '''
CREATE OR REPLACE TABLE annotations (
    id VARCHAR,
    "labels" VARCHAR
)
'''

# Execute the SQL statement to create the table
cur.execute(create_table_sql)

# Iterate through the JSON data and insert rows into the Snowflake table
for item in data:
    id_val = item['id']
    print(id_val)
    name_of_image = item['annotations']
    print(name_of_image)
    
    # SQL statement to insert data into the table
    insert_sql = '''
    INSERT INTO annotations ("ID", "labels")
    VALUES (%s, %s)
    '''
    
    # Execute the SQL statement to insert data
    cur.execute(insert_sql, (id_val, name_of_image))

# Commit the changes
conn.commit()

# Close the cursor and connection
cur.close()
conn.close()
