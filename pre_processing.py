"""
__author__: josep ferrandiz
"""

import sys
import os
import pandas as pd
import numpy as np
import config.config as cfg
from config.logger import logger
from joblib import Parallel, delayed
import time
from dask.distributed import Client
from dask import dataframe as dd

N_JOBS = -1


def set_context_flags(f, f_dict):
    for flag, v_states in f_dict.items():
        t_flag = f[f['state'].isin(v_states)]['timestamp'].min()
        if pd.notna(t_flag):
            b = f['timestamp'] >= t_flag
            g = f[b].copy()
            g[flag] = True
            f = pd.concat([f[~b], g], axis=0)
    return f


def set_state_context(r):
    if r['contacted'] is True:
        r['ctx'] = '+contacted'
    else:
        if r['showroom'] is True:
            r['ctx'] = '+showroom'
        else:
            if r['identified'] is True:
                r['ctx'] = '+identified'
            else:
                pass
    return r


def state_context_final(f):
    for c in ['i_state_name', 'j_state_name']:
        f[c] = f[c].astype(str) + f['ctx']
        f[c] = f[c].astype(pd.StringDtype(storage='pyarrow'))
    f.drop(list(cfg.ctx_flags.keys()) + ['ctx'], axis=1, inplace=True)
    for lbl in ['+contacted', '+identified', '+showroom']:
        for a in ['sale_completed', 'sale_lost']:
            f['j_state_name'] = f['j_state_name'].str.replace(pat=a + lbl, repl=a, regex=False)
    return f


def drop_bad_journeys(f, j_len_min, j_dur_min, end_days_cut, start_days_cut):
    f.sort_values(by=['timestamp'], inplace=True)
    end_d = pd.to_datetime(f[f['state'] != 'car_sold']['timestamp'].max().date() - pd.to_timedelta(end_days_cut, unit='D'))      # car sales dates are off
    start_d = pd.to_datetime(f[f['state'] != 'car_sold']['timestamp'].min().date() + pd.to_timedelta(start_days_cut, unit='D'))  # car sales dates are off
    f = f[(f['timestamp'] < end_d) & (f['timestamp'] >= start_d)].copy()
    j_sz = pd.DataFrame(f.groupby('user_distinct_id').size(), columns=['j_len']).reset_index()                                    # journey length (pages)
    j_hr = pd.DataFrame(f.groupby('user_distinct_id').agg({'timestamp': ['min', 'max']})).reset_index()                           # journey duration (hrs)
    j_hr.columns = ['user_distinct_id', 'min', 'max']
    j_hr['h_dur'] = (j_hr['max'] - j_hr['min']).dt.seconds / 3600.0
    j_sel = j_hr.merge(j_sz, on='user_distinct_id', how='left')
    j_sel = j_sel[(j_sel['j_len'] >= j_len_min) & (j_sel['h_dur'] >= j_dur_min)].copy()                                            # cut short journeys (pages and hrs)
    ids = j_sel['user_distinct_id'].values                                                                                         # valid distinct ids
    f = f[f['user_distinct_id'].isin(ids)].copy()
    return f


def set_states(f):  # init_state and j_states
    # start = time.time()
    f.sort_values(by=['timestamp'], inplace=True)
    groups = f.groupby('user_distinct_id')
    # logger.info('init vals: ' + str(np.round(time.time() - start)) + 'secs')

    # j_state
    # start = time.time()
    f['j_state_name'] = groups[['i_state_name']].shift(-1)
    # logger.info('j_state: ' + str(np.round(time.time() - start)) + 'secs')

    # fill j_state NAs
    # start = time.time()
    f['bool_'] = groups[['i_state_name']].transform(lambda x: any(x.isin(cfg.abs_states['sale_completed'])))  # true for journeys with  car sale
    # logger.info('z-merge: ' + str(np.round(time.time() - start)) + 'secs')

    # start = time.time()
    z1 = f[f['bool_'] == True].copy()
    z1['j_state_name'].fillna('sale_completed', inplace=True)
    z2 = f[f['bool_'] == False].copy()
    z2['j_state_name'].fillna('sale_lost', inplace=True)
    fx = pd.concat([z1, z2], axis=0)
    fx.drop('bool_', axis=1, inplace=True)
    return fx


def journey_context(f, f_dict):
    start = time.time()
    f.sort_values(by=['timestamp'], inplace=True)
    f['ctx'] = ''

    # set context flags
    for flag, v_states in cfg.ctx_flags.items():
        f['bool_' + flag] = f['state'].isin(v_states)

    for flag, v_states in f_dict.items():
        z = f[f['bool_' + flag] == True].copy()
        groups = z.groupby('user_distinct_id')
        zt = pd.DataFrame(groups['timestamp'].min()).reset_index()
        zt.columns = ['user_distinct_id', 'tmin']
        f = f.merge(zt, on='user_distinct_id', how='left')
        f[flag] = f['timestamp'] >= f['tmin']
        f.drop(['bool_' + flag, 'tmin'], axis=1, inplace=True)
    logger.info('set_data:: set_context_flags: ' + str(len(f)) + ' users: ' + str(f['user_distinct_id'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')
    f.to_parquet('~/data/f2.par')

    start = time.time()
    f.sort_values(by=['timestamp'], inplace=True)
    f = f.apply(set_state_context, axis=1).reset_index(drop=True)
    logger.info('set_data:: set_state_context: ' + str(len(f)) + ' users: ' + str(f['user_distinct_id'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')

    start = time.time()
    f = state_context_final(f.copy())
    logger.info('set_data:: state_context_final: ' + str(len(f)) + ' users: ' + str(f['user_distinct_id'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')
    return f


def set_data(df, j_len_min, j_dur_min, start_days_cut, end_days_cut):
    df.columns = [c.lower() for c in df.columns]
    logger.info('initial data:: len df: ' + str(len(df)) + ' users: ' + str(df['user_distinct_id'].nunique()))

    # cfg data
    state_maps = cfg.state_maps
    ctx_flags = cfg.ctx_flags

    # drop agents
    agents_df = pd.read_csv(os.getcwd() + '/data/agents.tsv', sep='\t')
    agents = agents_df['AGENT_ID'].to_list()
    df = df[~df['user_distinct_id'].isin(agents)].copy()

    # drop old cols
    for c in ['showroom_status', 'first_sales_match_date']:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

    # map state
    df['i_state_name'] = df['state'].replace(state_maps)

    # set dtypes
    df['timestamp'] = pd.to_datetime(df['timestamp'].values)
    for c in ['user_distinct_id', 'dpid', 'state', 'i_state_name']:
        df[c] = df[c].astype(pd.StringDtype(storage='pyarrow'))
        df[c].fillna(pd.NA, inplace=True)

    # context flags
    for c in ctx_flags.keys():
        df[c] = False

    # to drop:
    # - short journeys (in pages (state) and duration)
    # - journeys that end activities less than <end_days_cut> from last time (car_sold may not be updated yet)
    # - journeys that start activities before than <start_days_cut> from the first time (otherwise, it may be a partial session)
    start = time.time()
    df = drop_bad_journeys(df.copy(), j_len_min, j_dur_min, end_days_cut, start_days_cut)
    logger.info('usable data:: len df: ' + str(len(df)) + ' users: ' + str(df['user_distinct_id'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')
    df.to_parquet('~/data/df0.par')

    # set j_state_name and init_state
    start = time.time()
    df = set_states(df.copy())
    logger.info('set_data:: set_states: ' + str(len(df)) + ' users: ' + str(df['user_distinct_id'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')
    df.to_parquet('~/data/df2.par')

    # context
    start = time.time()
    df = journey_context(df.copy(), cfg.ctx_flags)
    logger.info('set_data:: journey_context: ' + str(len(df)) + ' users: ' + str(df['user_distinct_id'].nunique()) + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')
    df.to_parquet('~/data/df3.par')

    # init state
    # start = time.time()
    groups = df.groupby('user_distinct_id')
    z_first = groups[['i_state_name']].first().reset_index()
    z_first.columns = ['user_distinct_id', 'initial_state_name']
    df = df.merge(z_first, on='user_distinct_id', how='left')
    df.to_parquet('~/data/df4.par')
    # logger.info('init state: ' + str(np.round(time.time() - start)))

    # counts
    df['count'] = 1
    start = time.time()
    fv = df.groupby(['user_distinct_id', 'initial_state_name', 'i_state_name', 'j_state_name'])['count'].sum().reset_index()
    logger.info('set_data:: counts: ' + str(np.round(time.time() - start, 2)) + 'secs')
    fv.to_parquet('~/data/fv.par')

    # number states
    states = list(set((set(fv['i_state_name'].unique()).union(set(fv['j_state_name'].unique())))))
    for a in cfg.abs_states.keys():  # put absorbing states at the end of the list.
        states.remove(a)
        states.append(a)

    # renaming and move to state nbrs
    s = pd.DataFrame({'state_name': states})
    s.reset_index(inplace=True)
    s.rename(columns={'state_name': 'j_state_name', 'index': 'j_state'}, inplace=True)
    ff = fv.merge(s, on='j_state_name', how='left')
    s.rename(columns={'j_state_name': 'i_state_name', 'j_state': 'i_state'}, inplace=True)
    ff = ff.merge(s, on='i_state_name', how='left')
    s.rename(columns={'i_state_name': 'initial_state_name', 'i_state': 'init_state'}, inplace=True)
    ff = ff.merge(s, on='initial_state_name', how='left')
    s.rename(columns={'initial_state_name': 'state_name', 'init_state': 'id'}, inplace=True)
    ff.rename(columns={'user_distinct_id': 'journey_id'}, inplace=True)
    ff['journey_id'] = ff['journey_id'].astype('string')

    # map checks
    b1 = (ff['i_state_name'] == ff['j_state_name']) & (ff['i_state'] != ff['j_state'])
    b2 = (ff['i_state_name'] != ff['j_state_name']) & (ff['i_state'] == ff['j_state'])
    if b1.sum() > 0 or b2.sum() > 0:
        fpath = '~/data/fmaps.par'
        logger.error('state matching error. Saving maps to ' + fpath)
        ff.to_parquet(fpath)
        sys.exit(-1)

    # decoder ring
    si = ff[['i_state_name', 'i_state']].drop_duplicates()
    si.columns = ['state_name', 'state_id']
    sj = ff[['j_state_name', 'j_state']].drop_duplicates()
    sj.columns = ['state_name', 'state_id']
    fmap = pd.concat([si, sj], axis=0).drop_duplicates()
    fmap.reset_index(inplace=True, drop=True)

    fout = ff[['journey_id', 'init_state', 'i_state', 'j_state', 'count']].copy()
    fout.drop_duplicates(inplace=True)
    for c in ['init_state', 'i_state', 'j_state', 'count']:
        fout[c] = fout[c].astype(np.int32)
    fpath = '~/data/fdata.par'
    logger.info('saving data to ' + fpath)
    fout.to_parquet(fpath)
    return fout, fmap


def check_absorbing(f, n_states, n_abs, verbose=True):
    # check that the absorbing states are absorbing indeed
    # no need to groupby journey
    # assumes only > 0 counts
    drop_ids = list()
    for s in range(n_states - n_abs, n_states):            # s is absorbing
        b_out = (f['i_state'] == s) & (f['i_state'] != s)  # error if outflow
        drop_ids += list(f[b_out == True]['journey_id'].unique())
    if len(drop_ids) > 0:
        f = f[~f['journey_id'].isin(drop_ids)].copy()
        if verbose is True:
            logger.info('check_absorbing:dropped ' + str(drop_ids))
    return f


def check_transient(f, n_states, n_abs, verbose=True):
    # only positive counts in f
    # check not absorbing: jumps out of every transient state in each journey
    b = (f['i_state'] == f['j_state']) & (f['i_state'] < n_states - n_abs)
    f1 = f[b].copy()
    f1['count'] = 0     # drop return jumps
    f2 = f[~b].copy()   # only flow out
    f0 = pd.concat([f1, f2], axis=0)
    g = f0.groupby(['journey_id', 'i_state'])['count'].sum().reset_index()  # sum(count) is the total flow out of each state i in each journey
    drop_ids = g[g['count'] == 0]['journey_id'].unique()                    # journeys with absorbing states that should be transient
    if len(drop_ids) > 0:
        f = f[~f['journey_id'].isin(drop_ids)].copy()
        if verbose is True:
            logger.info('check_transient:dropped ' + str(drop_ids))
    return f
