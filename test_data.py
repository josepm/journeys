"""
__author__: josep ferrandiz

"""
import pandas as pd
import numpy as np
import itertools


def complete_df(f, idx_mtx, n_states_):
    # fill missing counts with 0
    if f['init_state'].nunique() != 1:
        print('init state error: ' + str(f['init_state'].unique()))
        return None
    g = idx_mtx.merge(f, on=list(idx_mtx.columns), how='left')
    g['journey_id'] = f['journey_id'].unique()[0]
    g['init_state'] = f['init_state'].unique()[0]
    g['count'].fillna(0, inplace=True)
    g['journey_id'] = f['journey_id'].unique()[0]

    # journey gets absorbed into 1 state only with one jump only
    g.sort_values(by='j_state', inplace=True)

    # impose absorption
    b = (g['i_state'] == n_states_ - 2) | (g['i_state'] == n_states_ - 1)
    cnts = g['count'].values
    cnts[b] = 0
    g['count'] = cnts

    # set counts into absorption 1 or 0
    # note some journeys can be absorbed in either state
    b = (g['j_state'] == n_states_ - 2) | (g['j_state'] == n_states_ - 1)
    cnts = g['count'].values
    cnts[b] = np.where(cnts[b] > 0, 1, 0)  # set jump count to absorbing to 1 or 0
    g['count'] = cnts
    if g[b]['count'].sum() == 0:  # this journey has no absorption. Add one
        i = np.random.randint(0, n_states_ - 2)
        j = np.random.randint(n_states_ - 2, n_states_)
        c = (g['j_state'] == j) & (g['i_state'] == i)
        cnts = g['count'].values
        cnts[c] = 1
        g['count'] = cnts
    return g


def test_data(n_states, n_journeys):
    # generate fake data
    # n_states = 10
    # n_journeys = 5
    max_cnt = 25  # max transitions between 2 states
    min_len = int(n_states / 2)
    i_states__ = [list(np.random.choice(np.arange(n_states - 2), replace=False, size=np.random.randint(low=min_len, high=n_states - 2, size=1)))
                  for _ in range(n_journeys)]
    i_states_ = list()
    j_states_ = list()
    counts_ = list()
    init_state_ = list()
    journeys_ = list()
    journey = 1
    for li in i_states__:
        if len(li) >= min_len:
            i_states_.append(li)
            j_states_ += [list(np.random.choice(np.arange(n_states), replace=True, size=len(li)))]
            counts_ += [list(np.random.choice(np.arange(max_cnt), replace=True, size=len(li)))]
            init_state_ += [[np.random.randint(n_states - 2, size=1)[0]] * len(li)]
            journeys_ += [[journey] * len(li)]
            journey += 1

    i_states = [x for y in i_states_ for x in y]
    j_states = [x for y in j_states_ for x in y]
    counts = [x for y in counts_ for x in y]
    init_state = [x for y in init_state_ for x in y]
    journeys = [x for y in journeys_ for x in y]

    data = pd.DataFrame({
        'journey_id': journeys,
        'init_state': init_state,
        'i_state': i_states,
        'j_state': j_states,
        'count': counts
    })
    return data


def sim_journeys(fj, n_max):
    fj_list = list()
    n = np.random.randint(n_max)
    if n > 0:
        _ = fj.apply(add_journeys, n=n, fj_list=fj_list, axis=1)
        sf = pd.concat(fj_list, axis=0)
    fj['journey_id'] = fj['journey_id'].astype(str) + '.0'
    return pd.concat([fj, sf], axis=0) if n > 0 else fj


def add_journeys(x, n, fj_list):
    # replicate paths with different counts
    jny = pd.DataFrame(np.random.randint(low=int(x['count'] / 2), high=1 + int(1.5 * x['count']), size=n), columns=['count'])
    jid = [str(x['journey_id']) + '.' + str(ix) for ix in range(1, n + 1)]
    jny['i_state'] = x['i_state']
    jny['j_state'] = x['j_state']
    jny['init_state'] = x['init_state']
    jny['journey_id'] = jid
    fj_list.append(jny)
    return None


def main(n_states, n_journeys, n_max=10):
    data = test_data(n_states, n_journeys)
    s_data = data.groupby('journey_id').apply(sim_journeys, n_max=n_max).reset_index(drop=True)  # generate similar journeys
    transitions = list(itertools.product(range(n_states), range(n_states)))
    smtx = pd.DataFrame(transitions, columns=['i_state', 'j_state'])
    n_data = s_data.groupby('journey_id').apply(complete_df, idx_mtx=smtx, n_states_=n_states).reset_index(drop=True)
    return n_data

