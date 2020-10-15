import os
import pyodbc
import pandas as pd


'''
    Connection for the SQL database
'''
conn = pyodbc.connect(
    "Driver={SQL Server Native Client 11.0};"
    "Server=SB-WKS032W10\SQLEXPRESS1;"
    "Database=ILS_rec_13.9;"
    "Trusted_Connection=yes"
    )


print('Importing the dataset')
# SQL to load the dataset for Model Building
sql_dataset = """
select p.PERSON_ID as personId,p.LANGUAGE_ID as languageId,p.CLIENT_ID as clientId,
p.COMPANY_ID as companyId,p.COUNTRY_ID as countryId,p.AUTHENTIFICATIONSTATUS_ID as authentificationId,
p.SALUTATION_ID as salutationId,ec.OBJECT_ID as objectId
from person p, portfolio pf, e_component ec
where pf.person_id = p.person_id
and pf.component_id = ec.component_id
and pf.course_id = 0 
and p.authentificationstatus_id = 1 
and pf.status in (8,9,10,11,12)
order by p.PERSON_ID
"""

print('Importing the ranking')
sql_ranking = """
select ec.OBJECT_ID, count(ec.OBJECT_ID)
from person p, portfolio pf, e_component ec
where pf.person_id = p.person_id
and pf.component_id = ec.component_id
and pf.course_id = 0 
and p.authentificationstatus_id = 1 
and pf.status in (8,9,10,11,12)
GROUP by ec.OBJECT_ID ORDER by 2 DESC
"""

dataDataframe = pd.io.sql.read_sql(sql_dataset, conn)
ranking_df = dataDataframe.objectId.value_counts()
ranking_df = pd.DataFrame(ranking_df)
ranking_df = ranking_df.reset_index()