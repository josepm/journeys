"""
__author__: josep ferrandiz

"""
import os
import numpy as np
import pandas as pd
import sql
import geohash as gh
import gc


def set_hash(a_df, hash_cols, keep=None, out_col='cookie'):
    a_df[out_col] = pd.util.hash_pandas_object(a_df[hash_cols], index=False)
    drop_cols = hash_cols if keep is None else [x for x in hash_cols if x not in keep]
    h_map = a_df[[out_col] + drop_cols].drop_duplicates()
    h_map.reset_index(inplace=True, drop=True)  # out_col to hash_cols map
    a_df.drop(drop_cols, axis=1, inplace=True)  # output data DF
    return h_map, a_df.reset_index(drop=True)


def f_match(f, max_dpid):
    if len(f) == 1:
        return None
    elif len(f) <= 9000:
        if f['dpid'].nunique() > max_dpid:                         # wrong GPS probably
            return None

        fx = f[['user_distinct_id', 'dpid']].drop_duplicates()     # drop dups in same cell
        fy = fx.copy()
        c = fx.merge(fy, how='cross')
        fout = c[(c['user_distinct_id_x'] < c['user_distinct_id_y']) & (c['dpid_x'] != c['dpid_y'])].copy()        # lwr diagonal
        return fout
    else:
        print('\ttoo large::: cookie: ' + str(f.cookie.unique()[0]) +
              # ' g_user: ' + str(f['g_user'].unique()[0]) +
              ' f: ' + str(len(f)))
        return None


def u_match(f, d_all_, gcol):
    state = f[gcol].unique()[0]
    dx = u_match_(f)
    if state not in d_all_.keys():
        d_all_[state] = dict()

    for k, v in dx.items():
        if k in d_all_[state].keys():
            d_all_[state][k] += dx[k]
            d_all_[state][k] = list(set(d_all_[state][k]))
        else:  # new key
            d_all_[state][k] = list(set(v))
    return None


def u_match_(gd_):
    # join all matches with overlaps in values
    fxy = gd_[['user_distinct_id_x', 'user_distinct_id_y']].drop_duplicates()
    fxy.sort_values(by='user_distinct_id_x', inplace=True)
    fx = fxy.groupby('user_distinct_id_x').apply(lambda x: [x['user_distinct_id_x'].unique()[0]] + x['user_distinct_id_y'].to_list())
    dx_ = fx.to_dict()   # {..., u1: [u1, u2, ...], ...}
    return process_dict(dx_)


def process_dict(dx):
    t_dict = tuple(dx.items())
    for ix in range(len(t_dict)):
        ki, vi = t_dict[ix]
        for jx in range(ix + 1, len(t_dict)):
            kj, _ = t_dict[jx]
            if kj in vi:
                dx[kj] += dx.pop(ki)
            break
    return dx


def fill_dpid(f):
    b = (f['dpid_lat'].isnull()) | (f['dpid_lon'].isnull() > 0)
    if b.sum() == 0:
        return f
    else:
        f1 = f[b].copy()
        f1['dpid_lat'] = f1['user_lat'].mean()
        f1['dpid_lon'] = f1['user_lon'].mean()
        f2 = f[~b].copy()
        return pd.concat([f1, f2], axis=0)


def email_cnt(r, uf):
    g = uf[uf['user_distinct_id'].isin(r['values'])]
    r['values'] = g['user_distinct_id'].to_list()        #list(set(r['values']).intersection(set(uf['distinct_id'].to_list())))
    r['n_distinct_id'] = len(r['values'])           # nbr of distinct IDs
    r['u_emails'] = g['email'].nunique()            # nbr unique emails
    r['r_users'] = len(g)                           # nbr of registered users (i.e with email)
    return r


if __name__ == "__main__":
    # US Only
    lat_max = 49.382808
    lat_min = 24.521208
    lon_max = -66.945392
    lon_min = -124.736342

    # max_dpids: max unique dpids with same cookie and same geo_cell (more means bad GPS data)
    max_dpid_ = 5

    # days
    days = 1   # days > 1 not implemented

    # vectorize distance function
    v_gh_encode = np.vectorize(gh.encode)  # lat, lon, prec

    df = sql.get_data('/sql/u_match.sql')
    df.columns = [c.lower() for c in df.columns]
    print('initial length: ' + str(len(df)) + ' unique users: ' + str(df.user_distinct_id.nunique()))

    # only registered users that are not agents
    agents_df = pd.read_csv(os.getcwd() + '/data/agents.tsv', sep='\t')
    agents = agents_df['AGENT_ID'].to_list()
    uf = pd.read_parquet('~/data/users.par')
    uf.rename(columns={'distinct_id': 'user_distinct_id'}, inplace=True)
    my_uf = uf[~uf['user_distinct_id'].isin(agents)]
    df = my_uf.merge(df, on=['user_distinct_id', 'dpid'], how='left')
    df.dropna(inplace=True)
    print('registered no-agent length: ' + str(len(df)) + ' unique users: ' + str(df.user_distinct_id.nunique()))

    # print('======= nulls ===========')
    # print(df.isnull().sum())
    # df = df.groupby('dpid').apply(fill_dpid)
    df.dropna(inplace=True)
    print('post nulls length: ' + str(len(df)) + ' unique users: ' + str(df.user_distinct_id.nunique()))
    for c in df.columns:
        if 'itude' not in c:
            df[c] = df[c].astype(pd.StringDtype(storage='pyarrow'))
    df['date'] = pd.to_datetime(df['date'].values)
    df['user_lon'] = df['user_lon'].astype(float)
    df['user_lat'] = df['user_lat'].astype(float)
    df['dpid_lon'] = df['dpid_lon'].astype(float)
    df['dpid_lat'] = df['dpid_lat'].astype(float)
    print('======= dtypes ===========')
    print(df.dtypes)

    # US only
    df.reset_index(inplace=True, drop=True)
    df = df[(df['user_lon'] >= lon_min) & (df['user_lon'] <= lon_max) & (df['user_lat'] >= lat_min) & (df['user_lat'] <= lat_max)].copy()
    lat = df['user_lat'].values
    lon = df['user_lon'].values
    df['g_user'] = v_gh_encode(lat, lon, precision=6)  # 5=5km, 4=39.1km (worst case), 6=0.6km

    lat = df['dpid_lat'].values
    lon = df['dpid_lon'].values
    df['g_dpid'] = v_gh_encode(lat, lon, precision=3)

    # df['is_close'] = df.apply(lambda x: int(x['g_dpid'] in x['g_user']), axis=1)

    # columns to hash together
    h_cols = ['browser', 'browser_version', 'browser_size', 'os', 'os_version', 'is_mobile', 'device_language',
              'device_colors', 'device_resolution',
              'device_category', 'user_agent',
              # 'is_close',
              # 'g_user',
              # 'dpid_state'
              ]
    df.dropna(subset=h_cols, inplace=True)
    print('final rows: ' + str(len(df)))
    print('final users: ' + str(df.user_distinct_id.nunique()))
    fmap, f_data = set_hash(df.copy(), h_cols, )
    f_data.to_parquet(os.getcwd() + '/data/f_data.par')
    f_data.drop_duplicates(subset=['user_distinct_id', 'dpid', 'cookie'], inplace=True)
    gc.collect()

    # group by cookie and window days
    d_all = dict()
    for start_d in sorted(list(f_data['date'].unique())):
        end_d = start_d + pd.Timedelta(days, unit='D')
        date_str = pd.to_datetime(start_d).date()
        fd = f_data[(f_data['date'] >= start_d) & (f_data['date'] < end_d)]
        print(str(start_d) + ' end: ' + str(end_d) + ' rows: ' + str(len(fd)) +
              ' cookies: ' + str(fd.cookie.nunique()) +
              # ' g_user: ' + str(fd.g_user.nunique()) +
              ' users: ' + str(fd.user_distinct_id.nunique()))
        gcols = ['cookie', 'g_user', 'dpid_state']  #'g_dpid']  #, 'dpid_state']
        gd = fd.groupby(gcols).apply(f_match, max_dpid=max_dpid_).reset_index()  # Note: no y_user can appear as x_user because of sorting
        gd.drop('level_' + str(len(gcols)), axis=1, inplace=True)
        gd.to_parquet(os.getcwd() + '/data/gd_' + str(date_str) + '.par')
        print('\tgd len: ' + str(len(gd)) + ' x_users: ' + str(gd['user_distinct_id_x'].nunique()) + ' y_users: ' + str(gd['user_distinct_id_y'].nunique()))
        gc.collect()
        _ = gd.groupby(gcols[-1]).apply(u_match, d_all_=d_all, gcol=gcols[-1])
        for s in d_all.keys():  # states
            print(str(start_d) + ' state: ' + str(s) + ' keys: ' + str(len(d_all[s])))
        gc.collect()
    flist = [pd.DataFrame({'state': [s] * len(d_all[s]), 'key': d_all[s].keys(), 'values': d_all[s].values()}) for s in d_all.keys()]  # states
    f_all_ = pd.concat(flist, axis=0)
    f_all_.to_parquet(os.getcwd() + '/data/f_all_.par')

    # uf = pd.read_parquet('~/data/users.par')
    # my_users = [x for arr in f_all_['values'].values for x in arr]
    # my_users = list(set(my_users) - set(agents))
    # my_uf = uf[uf['distinct_id'].isin(my_users)]

    xf = f_all_.apply(email_cnt, uf=my_uf, axis=1)
    xf = xf[xf['n_distinct_id'] > 1].copy()

    # we always must have: n_distinct_id >= r_users >= u_emails >= 0
    # correct condition: n_distinct_id == r_users & u_emails == 1
    # hopefully correct: n_distinct_id > r_users & u_emails == 1, especially if r_users > 1.
    # incorrect condition: u_emails > 1
    # unknown: r_users = 0
    for n in range(2, 20):
        nf = xf[xf['n_distinct_id'] == n]
        cf = nf[(nf['n_distinct_id'] == nf['r_users']) & (nf['u_emails'] == 1)]  # correct
        hf = nf[(nf['n_distinct_id'] > nf['r_users']) & (nf['u_emails'] == 1)]   # hopefully correct
        ef = nf[(nf['u_emails'] > 1)]                                            # matching error
        uf = nf[nf['r_users'] == 0]                                              # unknown
        kn = len(cf) + len(hf) + len(ef)                                         # kn + len(uf) = len(nf)
        if kn > 0:
            print('n_distinct_id: ' + str(n) +
                  ' observable count: ' + str(kn) +
                  ' correct: ' + str(np.round(100 * len(cf) / kn, 2)) +
                  ' hopefully correct: ' + str(np.round(100 * len(hf) / kn, 2)) +
                  '% not observable: ' + str(np.round(100 * (1.0 - kn / len(nf)), 2)) + '%')

    oooooooooooooooooooooooooooooooo
    d_out = process_dict(d_all)
    f_all = pd.DataFrame({'key': d_out.keys(), 'values': d_out.values()})
    f_all['len'] = f_all.apply(lambda x: len(x['values']), axis=1)
    print(f_all['len'].describe())
    f_all.to_parquet(os.getcwd() + '/data/u_matches.par')
    f_all = f_all[f_all['len'] <= f_all['len'].quantile(0.9)]
    f_all.to_parquet(os.getcwd() + '/data/u_matches_q.par')












