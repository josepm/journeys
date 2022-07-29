"""
__author__: josep ferrandiz

Input data format:
- DF with columns: customer_id, timestamp, state (URL/activity), other flags?
- Processing
  - sort by time stamp
  - flags are derived from state visits and change future states
  - Group by customer_id, shift and count transitions
  - Number the states from 0 to D with,
    - D - 2: C state
    - D - 1: L state
  - Output: DF with customer_id, init_state, i_state, j_state, count.
            - 'i_state' is the state we are in and 'j_state' the state we transition to
            - For every customer in the data, the output DF will have:
              - constant value for init_state
"""

# TODO move clustering gap to utilities
# TODO refactor em functions
# done drop em functions not used
# TODO refactor init amtx and amtx
# TODO check what is happening with to_drop
# TODO drop journey, segments terminology to have a generic Markov chain mixture
# Done move pars to cfg
# Done if setting k_opt through AIC print summary of results. Loop support for k_opt and save multiple models
# TODO amtx state checks same as data?
# TODO notebook with state map access, mdl_smry, cluster settings, ...
# TODO drop adds
# TODO check init conditions
# TODO segment identification


import os
import em
import sql
import pandas as pd
import numpy as np
import pre_processing as pp
import config.config as cfg
from config.logger import logger
from joblib import Parallel, delayed
import time

N_JOBS = -1


if __name__ == "__main__":
    start = time.time()
    sql_data = sql.get_data('/sql/' + cfg.qry_file, cfg.start_date)
    logger.info('sql get_data: ' + str(len(sql_data)) + ' users: ' + str(sql_data['USER_DISTINCT_ID'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)))
    # sql_data = pd.read_parquet(os.getcwd() + '/data/sql_data.par')

    # data clean up
    n_data, s_map = pp.set_data(sql_data, cfg.j_len_min, cfg.j_dur_min, cfg.start_days_cut, cfg.end_days_cut)                # journey_id, init_state, i_state, j_state, count
    n_data['journey_id'] = n_data['journey_id'].astype(pd.StringDtype(storage='pyarrow'))
    for c in ['init_state', 'i_state', 'j_state', 'count']:
        n_data[c] = n_data[c].astype(np.int32)  # UInts do not work

    # check transient and absorbing states. Drop journeys with invalid transitions
    n_abs = len(cfg.abs_states)
    n_states = max(n_data['i_state'].nunique(), n_data['j_state'].nunique())   # unique states
    n_data = n_data[n_data['count'] > 0].copy()
    n_data_in = n_data.copy()

    # check absorbing
    start = time.time()
    n_data = pp.check_absorbing(n_data.copy(), n_states, n_abs)
    logger.info('check_absorbing: ' + str(len(n_data)) + ' users: ' + str(n_data['journey_id'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)))

    # check transient
    start = time.time()
    n_data = pp.check_transient(n_data.copy(), n_states, n_abs)
    logger.info('check_transient: ' + str(len(n_data)) + ' users: ' + str(n_data['journey_id'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)))

    # check init state: only need to check that init is not absorbing
    # start = time.time()
    z = n_data[n_data['init_state'] >= n_states - n_abs]
    if len(z) > 0:  # init states would be absorbing!
        n_data = n_data[~n_data['journey_id'].isin(z['journey_id'].unique())].copy()
    # logger.info('check_init: ' + str(len(n_data)) + ' users: ' + str(n_data['journey_id'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)))

    # check cycles: subset of states from which the chain never gets absorbed. These states should be dropped
    # NOT IMPLEMENTED

    # save the cleaned data
    logger.info('save cleaned-mapped data to ' + os.getcwd() + '/data/journeys.par')
    n_data.to_parquet(os.getcwd() + '/data/journeys.par')
    j_obj = em.process_segment(n_states, n_abs, n_data,
                               weighted=cfg.weighted, min_var=cfg.min_var, stochastic=cfg.stochastic,
                               state_maps=s_map, k_opt=cfg.k_opt, cluster=cfg.cluster)


