"""
__author__: josep ferrandiz

"""
import os
import snowflake.connector
from config.logger import logger


def exec_q(qry, conn):
    cur = conn.cursor()
    cur.execute(qry)
    df_out = cur.fetch_pandas_all() if 'select' in qry.lower() else list()
    cur.close()
    return df_out


def get_data(f_name, start_date_str):
    cwd = os.getcwd()
    qpath = cwd + f_name
    with open(qpath, 'r') as f:
        qry_ = f.read()
    # qry_list = qry_.split(';')

    q0 = 'set (date_start)=(\'' + start_date_str + '\');\n'
    qry_list = [q0, qry_]

    with snowflake.connector.connect(
            user=os.environ['ROADSTER_SNOWFLAKE_USER'],
            password=os.environ['ROADSTER_SNOWFLAKE_PASS'],
            account='bja64687',
            warehouse='MEDIUM_WH',
            database='LANDING',
            schema='ROADSTER'
        ) as conn_:
        for qry in qry_list:
            logger.info('starting sql')
            db_df = exec_q(qry, conn_)
            logger.info('sql raw completed with ' + str(len(db_df)))
    logger.info('saving sql data to ' + cwd + '/data/sql_data.par')
    db_df.to_parquet(cwd + '/data/sql_data.par')
    return db_df       # return only the last query result
