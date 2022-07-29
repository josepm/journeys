"""
__author__: josep ferrandiz

"""
# TODO Q and R matrices
# TODO Visits matrix
# TODO absorption probs and completion odds

import sys
import os
import pandas as pd
import numpy as np
import pickle
import time
import itertools
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score as ch_score
from sklearn.metrics import davies_bouldin_score as db_score
from sklearn.metrics import silhouette_score as sh_score
from config.logger import logger

N_JOBS = -1
min_val = 1.0e-12


def load_mdl(fname='journeys.pkl'):
    fpath = os.path.expanduser('~/my_projects/journeys/journeys/data/') + fname
    with open(fpath, 'rb') as fpk:
        mdl = pickle.load(fpk)
        # logger.info('model loaded from ' + fpath)
    return mdl


def process_segment(n_states, n_abs, n_data, weighted, min_var, stochastic,
                    max_mix=8, state_maps=None, k_opt=None, cluster=True, fname=None):
    # fname: without extension
    j_obj = Journey(n_states, n_abs, weighted, min_var, stochastic, max_mix=max_mix, state_maps=state_maps)
    cwd = os.getcwd()
    res = dict({'segments': [], 'aic': [], 'll': [], 'pars': [], 'fname': []})
    if k_opt is not None:
        if k_opt <= 1:
            logger.error('invalid k_opt. Resetting to 3')
            k_opt = 3
        if fname is None:
            fname = 'model'
        logger.info('checking segmentation from 1 to ' + str(k_opt - 1) + ' segments included')
        for k in range(1, k_opt):
            j_obj.em(n_data, k_opt=k, cluster=cluster)
            res['aic'].append(j_obj.aic())
            res['ll'].append(j_obj.best_ll)
            res['pars'].append(j_obj.pars)
            res['segments'].append(k)
            fpath = cwd + '/data/' + fname + '_' + str(k) + '.pkl'
            res['fname'].append(fpath)
            j_obj.save(fpath)
    else:
        j_obj.em(n_data, k_opt=None, cluster=cluster)
        res['aic'].append(j_obj.aic())
        res['ll'].append(j_obj.best_ll)
        res['pars'].append(j_obj.pars)
        res['segments'].append(j_obj.n_mix)
        fpath = cwd + '/data/' + fname + '_' + str(j_obj.n_mix) + '.pkl'
        res['fname'].append(fpath)
        j_obj.save(fpath)
    df_out = pd.DataFrame(res)
    logger.info('saving DF to ' + cwd + '/data/segment_results.csv')
    df_out.to_csv(cwd + '/data/segment_results.csv', index=False)
    return df_out


def trims(f, col, min_val=1.0e-12):
    # drop very small values in a series and normalize to 1
    f[col] = np.where(f[col].values < min_val, 0.0, f[col].values)
    if f[col].sum() > 0.0:
        f[col] /= f[col].sum()
    return f


# def inertia(X, cls):
#     cls_vals = list(set(cls.labels_))
#     sse = 0.0
#     for lbl in cls_vals:
#         Xc = X[cls.labels_ == lbl]
#         Cc = Xc.mean(axis=0)
#         sse += np.sum((Xc - Cc) ** 2)
#     return sse


def set_nmix(data, k_max, kmin=2, B=20, s_ctr=0):
    # gap, CH, silhouette, DB optimal clusters
    # https://hastie.su.domains/Papers/gap.pdf

    def gen_ref_data(X):
        return np.random.uniform(low=np.min(X, axis=0), high=np.max(X, axis=0), size=np.shape(X))  # must be uniform between the min and max in each col

    def gen_ref_kmean(k, X):
        # return MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300).fit(gen_ref_data(X)).inertia_
        return KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300).fit(gen_ref_data(X)).inertia_

    def gen_kmean(k, X):
        # cls = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, batch_size=256).fit(X)
        cls = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300).fit(X)
        inertia_ = cls.inertia_
        labels_ = cls.labels_
        ch = ch_score(X, labels_)
        db = db_score(X, labels_)
        sh = sh_score(X, labels_)
        return inertia_, ch, db, sh

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    start = time.time()
    logger.info('set_nmix: start gen_kmean')
    data_res = Parallel(n_jobs=N_JOBS)(delayed(gen_kmean)(k, data) for k in range(2, k_max + 1))
    # data_res = [gen_kmean(k, data) for k in range(2, k_max + 1)]
    data_log_inertia = np.log(np.array([x[0] for x in data_res]))
    ch_res = np.array([x[1] for x in data_res])
    db_res = np.array([x[2] for x in data_res])
    sh_res = np.array([x[3] for x in data_res])
    k_ch = kmin + np.argmax(ch_res)
    k_db = kmin + np.argmin(db_res)
    k_sh = kmin + np.argmax(sh_res)
    logger.info('set_nmix: gen_kmean done: ' + str(np.round(time.time() - start, 2)) + 'secs')

    # reference data
    logger.info('set_nmix: start gap')
    start = time.time()
    ref_log_inertia, ref_std = [], []
    for k in range(kmin, k_max + 1):
        u_cls = Parallel(n_jobs=N_JOBS)(delayed(gen_ref_kmean)(k, data) for _ in range(B))
        # u_cls = [gen_ref_kmean(k, data) for _ in range(B)]
        k_log_inertia = np.log(np.array(u_cls))
        ref_log_inertia.append(np.mean(k_log_inertia))
        ref_std.append(np.std(k_log_inertia))
    ref_log_inertia = np.array(ref_log_inertia)
    ref_std = np.sqrt(1.0 + 1.0 / B) * np.array(ref_std)
    logger.info('set_nmix: gap done: ' + str(np.round(time.time() - start, 2)) + 'secs')

    df = pd.DataFrame({
        'ref': np.log(ref_log_inertia),
        'data': np.log(data_log_inertia),
        'k': range(kmin, k_max + 1),
        'std': ref_std,
        'gap': ref_log_inertia - data_log_inertia
    })
    df['gap_std_shift'] = (df['gap'] - df['std']).shift(-1)
    df['diff'] = df['gap'] - df['gap_std_shift']  # gap(k) - (gap(k+1) - std(k+1))
    df.to_csv('~/data/gap.csv', index=False)      # save gap info
    df.set_index('k', inplace=True)               # index = cluster size
    thres, k_gap, ctr = 0.0, 2, 0
    while thres > df['diff'].min() and ctr < 1000:  # find the smallest k with diff > thres. Ideally thres = 0
        z = df[df['diff'] >= thres]
        if len(z) > 0:
            k_gap = z.index.min()
            break
        thres -= 0.001
        ctr += 1

    logger.info('optimal clusters: ch:: ' + str(k_ch) + ' db: ' + str(k_db) + ' sh: ' + str(k_sh) + ' gap::' + str(k_gap) + ' with threshold: ' + str(thres))

    # we prefer more than kmin segments if possible but not too many
    cls_ = pd.Series([k_gap, k_ch, k_db, k_sh]).sort_values(ascending=True)
    n_mix = cls_[cls_ > kmin].mode().values[0] if cls_.max() > kmin else kmin
    if n_mix == k_max:
        logger.warning('set_nmix::there may be more clusters than k_max = ' + str(k_max) + '. ctr = ' + str(s_ctr))
        return set_nmix(data, k_max + kmin, B=B, s_ctr=s_ctr + 1) if s_ctr < 5 else n_mix
    else:
        return n_mix


def check_stochastic(tmtx, n_states, n_abs):
    # check unused states (no counts in or out: prob is NA)
    if tmtx['prob'].isnull().sum() > 0:
        logger.warning('null probs in stochastic matrix. Setting to 0')
        logger.warning('\n' + str(tmtx[tmtx['prob'].isnull()]))
        tmtx['prob'].fillna(0.0, inplace=True)

    # stochastic check
    tmtx, to_drop = row_sums(tmtx.copy(), n_states, n_abs)
    return tmtx, to_drop


def row_sums(tmtx, n_states, n_abs):     # stochastic check
    t_sum = tmtx.groupby('i_state')['prob'].sum().reset_index()
    r_sum = pd.DataFrame([0] * len(range(n_states))).reset_index()
    r_sum.columns = ['i_state', 'prob']
    r_sum = r_sum.merge(t_sum, on='i_state', how='left')
    r_sum['r_sum'] = r_sum[['prob_x', 'prob_y']].max(axis=1)
    r_sum.drop(['prob_x', 'prob_y'], axis=1, inplace=True)
    to_drop = list(r_sum[r_sum['r_sum'] == 0.0]['i_state'].unique())  # absorbing states
    _ = [to_drop.remove(s) for s in range(n_states - n_abs, n_states) if s in to_drop]

    # make sure the actual absorbing states are not dropped!
    for s in range(n_states - n_abs, n_states):
        if s in to_drop:
            to_drop.remove(s)
            tmtx.set_index(['i_state', 'j_state'], inplace=True)
            tmtx.loc[(s, s), 'prob'] = 1.0
            tmtx.reset_index(inplace=True)

    # if len(to_drop) > 0:
    #     logger.warning('invalid absorbing states:  ' + str(to_drop) + ' Dropping them')

    # normalize
    tmtx = tmtx[(~tmtx['i_state'].isin(to_drop)) & (~tmtx['j_state'].isin(to_drop))].copy()
    tmtx = tmtx.merge(r_sum, on='i_state', how='left')
    tmtx['prob'] /= tmtx['r_sum']
    tmtx.drop('r_sum', axis=1, inplace=True)
    return tmtx, to_drop


def to_stochastic(df):
    den_df = df[['journey_id', 'i_state', 'j_state', 'count']].groupby(['journey_id', 'i_state'])['count'].sum().reset_index()
    den_df.columns = ['journey_id', 'i_state', 'den']
    pf = df[['journey_id', 'init_state', 'i_state', 'j_state', 'count']].merge(den_df, on=['journey_id', 'i_state'], how='left')
    b0 = pf['den'] == 0.0
    if b0.sum() == 0:
        pf['count'] = pf['count'] / pf['den']
    else:
        pf0 = pf[b0].copy()
        pf1 = pf[~b0].copy()
        pf1['count'] = pf1['count'] / pf1['den']
        pf0['count'] = pf0.apply(lambda x: float(x['i_state'] == x['j_state']), axis=1)
        pf = pd.concat([pf0, pf1], axis=0)
    pf.drop('den', axis=1, inplace=True)
    return pf[pf['count'] > 0.0]


class Journey:
    def __init__(self, n_states, n_abs, weighted=True, min_var=0.25, stochastic=True, max_mix=8, state_maps=None):
        """
        Assumptions:
        - states: 0,..., n_states - n_abs - 1 are transient
        - states: n_state - n_abs, ..., n_states - 1 are absorbing

        :param n_states: total number of states
        :param n_abs: number of absorbing states
        :param weighted: True: weight by journey counts, False, regular unweighted average by journey
        :param min_var: min variance to explain by PCA for initial cluster counts
        :param stochastic: scale data by converting counts to row-stochastic or not at all. Do not scale by features (columns) I think
        :param max_mix: max number of mixtures
        """
        if max_mix < 2:
            logger.error('should have at 2 mixtures: ' + str(max_mix))
            sys.exit(-1)
        if n_states <= n_abs:
            logger.error('should have at least ' + str(n_abs + 1) + ' states: ' + str(n_states))
            sys.exit(-1)

        self.max_mix = max_mix
        self.n_states = n_states
        self.n_abs = n_abs
        self.max_iter, self.min_iter = 50, 10
        self.rel_err = 1.0e-03      # min improvement across iterations
        self.best_ll = None
        self.last_ll = None
        self.state_maps = state_maps  # map state id to state name
        self.segments = None
        self.pi_df = None
        self.amtx = None
        self.mu = None
        self.best_mu = None
        self.xi_df = None
        self.best_pi = None
        self.best_amtx = None

        # hyper pars
        self.weighted = weighted        # weighted journey avgs (True) vs, regular avg
        self.min_var = min_var          # min variance to capture for clustering PCA
        self.stochastic = stochastic    # data scaling method: either i-row sum to 1. Do not scale by feature???

    def save(self, fpath):
        with open(fpath, 'wb') as fpk:
            pickle.dump(self, fpk, pickle.HIGHEST_PROTOCOL)
            logger.info('model saved to ' + fpath)

    def initialize(self, data, k_opt=None, cluster=True):
        # assumption: data has been checked and is of the form journey_id, init_state, i_state, j_state, count
        if self.stochastic is True:
            data = to_stochastic(data[data['count'] > 0.0].copy())

        # initial segmentation (based on clustering)
        start = time.time()
        n_data, cl_labels, self.n_mix = self.clustering(data, k_opt=k_opt, cluster=cluster)
        logger.info('initialize: clustering completed:: optimal mixtures:  ' + str(self.n_mix) + ' duration: ' + str(np.round(time.time() - start)) + 'secs')

        # initialize segments (pi, amtx). Each mixture is called a segment
        start = time.time()
        self.segments = {segment: Segment(self.n_states, self.n_abs, segment, n_data[n_data['init_segment'] == segment], self.weighted, self.stochastic)
                         for segment in np.unique(cl_labels)}
        self.pi_df = pd.concat([s_obj.pi_df for s_obj in self.segments.values()]).reset_index(drop=True)
        self.amtx = pd.concat([s_obj.amtx for s_obj in self.segments.values()]).reset_index(drop=True)
        logger.info('initialize: segments completed' + ' duration: ' + str(np.round(time.time() - start)) + 'secs')

        # initialize the mixture distribution
        start = time.time()
        u_dict = pd.Series(cl_labels).value_counts(normalize=True).to_dict()
        self.mu = np.zeros(self.n_mix)
        for k in range(np.shape(self.mu)[0]):
            self.mu[k] = u_dict.get(k, 0.0)
        self.best_mu = self.mu
        logger.info('initialize: mixture distribution completed' + ' duration: ' + str(np.round(time.time() - start)) + 'secs')

        # initialize posterior
        start = time.time()
        pf = n_data[['journey_id', 'init_state', 'init_segment']].drop_duplicates()
        cf = pf[['journey_id']].merge(pd.DataFrame({'segment': range(self.n_mix)}), how='cross')
        sf = cf.merge(pf, on='journey_id', how='left')
        sf['xi'] = sf.apply(lambda x: float(x['init_segment'] == x['segment']), axis=1)
        sf.drop('init_segment', axis=1, inplace=True)
        self.xi_df = sf[['journey_id', 'segment', 'xi', 'init_state']].drop_duplicates()
        self.xi_df = self.xi_df[self.xi_df['xi'] > 0].copy()
        logger.info('initialize: posterior completed' + ' duration: ' + str(np.round(time.time() - start)) + 'secs')

        # initial ll
        start = time.time()
        self.best_ll = self.log_likelihood(n_data)
        self.init_ll = self.best_ll
        self.last_ll = self.best_ll
        self.best_mu = self.mu
        self.best_pi = self.pi_df
        self.best_amtx = self.amtx
        logger.info('initialize: log-likelihood completed: ' + str(self.n_mix) +
                    ' segments. Initial LL: ' + str(self.best_ll) +
                    ' duration: ' + str(np.round(time.time() - start)) + 'secs')
        n_data.drop('init_segment', axis=1, inplace=True)
        return n_data      # n_data is either counts (stochastic == False) or stochastic (stochastic == True)

    def clustering(self, data, k_opt=None, cluster=True):
        # assumption: data has been checked
        data['i->j'] = data['i_state'].astype(str) + '->' + data['j_state'].astype(str)
        p_data = pd.pivot_table(data, index='journey_id', values='count', columns='i->j')       # one row per journey with counts i->j
        p_data.fillna(0, inplace=True)
        for c in p_data.columns:
            if p_data[c].nunique() <= 1:
                p_data.drop(c, inplace=True, axis=1)
                logger.info('dropping column: ' + str(c))

        c = None
        for c in range(1, np.shape(p_data)[1]):
            pca = PCA(n_components=c).fit(p_data)
            if np.sum(pca.explained_variance_ratio_) > self.min_var:
                break
        logger.info('PCA components: ' + str(c))
        pca_data = PCA(n_components=c).fit_transform(p_data)
        if k_opt is None:
            k_opt = set_nmix(pca_data, k_max=self.max_mix)
        start = time.time()
        if cluster is True:
            logger.info('cluster based initial segments')
            kmeans = KMeans(n_clusters=k_opt, init='k-means++', n_init=10, max_iter=300).fit(pca_data)
            labels = kmeans.labels_
            # f_map = pd.DataFrame({'journey_id': list(p_data.index), 'init_segment': labels})
            # n_data = data.merge(f_map, on='journey_id', how='left')  # add initial segment to data
            # n_data.drop('i->j', axis=1, inplace=True)
            # logger.info('clustering:: initial clusters:: ' + str(np.round(time.time() - start, 2)) + 'secs')
        else:
            logger.info('random initial segments')
            labels = np.random.randint(0, k_opt, size=data['journey_id'].nunique())
        f_map = pd.DataFrame({'journey_id': list(p_data.index), 'init_segment': labels})
        n_data = data.merge(f_map, on='journey_id', how='left')  # add initial segment to data
        n_data.drop('i->j', axis=1, inplace=True)
        # logger.info('clustering:: initial clusters done:: ' + str(np.round(time.time() - start, 2)) + 'secs')
        return n_data, labels, k_opt

    def em(self, data, k_opt=None, cluster=True, verbose=True):
        """
        assumption: data has been checked
        # k_opt: nbr of segments. If None it is computed
        EM computation
        :param data: data frame with journey_id, init_state, i_state, j_state, count with journey level checks for absorption and transient completed
        :param k_opt: number of segments (if None use KMeans to find opt number)
        :param cluster: if True use labels from clustering, else randomly initialize assigment of journeys to cluster
        :param verbose:
        :return:
        """
        self.n_journeys = data['journey_id'].nunique()
        start = time.time()
        s_data = self.initialize(data[data['count'] > 0.0].copy(), k_opt=k_opt, cluster=cluster)
        logger.info('EM initialize completed: ' + str(self.n_mix) + ' segments. Duration: ' + str(np.round(time.time() - start)) + 'secs')
        start = time.time()
        n_iter, ll_err = 0, np.inf
        logger.info('EM starts for ' + str(self.n_mix) + ' segments')
        while n_iter <= self.max_iter:
            ll_err = self.em_(s_data)
            if verbose is True:
                logger.info('n_mix: ' + str(self.n_mix) +
                            ' iter: ' + str(n_iter) +
                            ' init ll: ' + str(np.round(self.init_ll, 2)) + ' best ll: ' + str(np.round(self.best_ll, 2)) +
                            ' current ll: ' + str(np.round(self.ll, 2)) +
                            ' ll improvement: ' + str(np.round(100 * ll_err, 3)) + '%')
            if ll_err <= self.rel_err and n_iter >= self.min_iter:
                break
            n_iter += 1
        if ll_err > self.rel_err and n_iter > self.max_iter:
            logger.warning('Warning: could not converge for ' + str(self.n_mix) + ' segments after ' + str(n_iter) + ' iterations.')
        logger.info('EM completed for ' + str(self.n_mix) + ' segments. Duration: ' + str(np.round(time.time() - start, 2)) + 'secs. AIC: ' + str(np.round(self.aic())))

    def em_(self, data):
        start = time.time()
        self.em_e_step(data)    # e_step
        logger.info('\tem_: e-step complete' + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')

        start = time.time()
        self.em_m_step()
        logger.info('\tem_: m-step complete' + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')

        # log-likelihood
        start = time.time()
        self.ll = self.log_likelihood(data)

        # checks
        if self.ll < self.best_ll:  # this should not happen!
            logger.error('LL from EM should always increase::last: ' + str(np.round(self.last_ll, 2)) + ' current: ' + str(np.round(self.ll, 2)))
            ll_err = np.inf
        else:
            self.best_ll = self.ll
            self.best_mu = self.mu
            self.best_pi = self.pi_df
            self.best_amtx = self.amtx
            ll_err = -(self.ll - self.last_ll) / self.last_ll   # step improvement
        self.last_ll = self.ll
        logger.info('\tem_: LL complete' + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')

        return ll_err

    def em_e_step(self, data):
        # e_step
        start = time.time()
        logger.info('\tem_e_step: start')
        # df_list = [s_obj.e_step_k(self.mu[k], data) for k, s_obj in self.segments.items()]
        df_list = Parallel(n_jobs=N_JOBS)(delayed(s_obj.e_step_k)(self.mu[k], data) for k, s_obj in self.segments.items())
        ef = pd.concat(df_list, axis=0)
        logger.info('\t\tem_e_step: ef::' + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')
        init_f = ef[['journey_id', 'init_state']].drop_duplicates()

        # compute xi_n(k) = 1/sum_h(N_n(h)/N_n(k))
        start = time.time()
        pf = pd.pivot_table(ef, index='journey_id', columns='segment', values='l_N_n')
        pf.columns = list(pf.columns.values)
        segments = [k for k in pf.columns]
        for cols in itertools.product(pf.columns, repeat=2):
            pf[str(cols)] = np.exp(pf[cols[1]] - pf[cols[0]])   # exp(log(N_h) - log(N_k)) = N_h/N_k
            pf[str(cols)].fillna(0, inplace=True)

        k_df_list = list()
        for k in segments:
            k_df = pd.DataFrame(index=pf.index)
            cols = [str((k, c)) for c in segments]
            k_df['xi'] = 1.0 / pf[cols].sum(axis=1)
            k_df['segment'] = k
            k_df_list.append(k_df)
        self.xi_df = pd.concat(k_df_list)
        self.xi_df.reset_index(inplace=True)
        self.xi_df = self.xi_df.merge(init_f, on='journey_id', how='left')         # journey_id, segment, init_state, xi
        self.xi_df.drop_duplicates(inplace=True)
        logger.info('\t\tem_e_step: xi::' + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')

    def em_m_step(self):
        # ######## segment mixture
        logger.info('\tem_m_step: start')
        start = time.time()
        num = self.xi_df[['segment', 'xi']].groupby('segment')['xi'].sum()
        self.mu = num.values / num.sum()
        logger.info('\t\tem_m_step:  mixture complete' + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')

        # ######## segment init state
        # self.pi_df columns: segment, init_state, prob
        start = time.time()
        qi = pd.Series([0] * (self.n_states - self.n_abs))
        self.pi_df = pd.concat([s_obj.pi_k(self.xi_df[self.xi_df['segment'] == k][['journey_id', 'init_state', 'xi']], qi) for k, s_obj in self.segments.items()], axis=0)
        logger.info('\t\tem_m_step: init_state complete' + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')

        # ######## transition probs
        # columns: segment, i_state, j_state, prob
        start = time.time()
        self.amtx = pd.concat([s_obj.amtx_k(self.xi_df[self.xi_df['segment'] == k][['journey_id', 'xi']]) for k, s_obj in self.segments.items()], axis=0)
        logger.info('\t\tem_m_step:  transition mtx complete' + ' duration: ' + str(np.round(time.time() - start, 2)) + 'secs')

    def log_likelihood(self, data):
        # model log-likelihood
        df_list = [s_obj.log_likelihood_k(self.mu[k], data) for k, s_obj in self.segments.items()]
        lf = pd.concat(df_list, axis=0)  # one row per journey: segment, 'mu_k * e^l_kn'
        gf = lf[['journey_id', 'mu_k * e^l_kn']].groupby('journey_id').sum().reset_index()  # segment, sum over segments
        gf.rename(columns={'mu_k * e^l_kn': 'likelihood'}, inplace=True)
        return np.log(gf['likelihood']).sum()

    def set_pars(self):
        self.pars = self.n_mix - 1 + sum([s_obj.n_pars() for k, s_obj in self.segments.items()])

    def aic(self):
        # the smaller, the better
        self.set_pars()
        return 2.0 * self.pars - 2.0 * self.best_ll + 2.0 * self.pars * (self.pars + 1.0) / (self.n_journeys - self.pars - 1.0)

    def bic(self):
        # the smaller, the better
        self.set_pars()
        return self.pars * np.log(self.n_journeys) - 2.0 * self.best_ll

    def sale_smry(self):
        _ = [s_obj.set_mdl_k() for k, s_obj in self.segments.items()]
        sale_smry_ = pd.concat([s_obj.sale_smry_k() for k, s_obj in self.segments.items()])
        if self.state_maps is not None:
            sale_smry_ = sale_smry_.merge(self.state_maps, left_on='absorption_state', right_on='state_id', how='left')
            sale_smry_.drop('absorption_state', axis=1, inplace=True)
        else:
            sale_smry_.rename(columns={'absorption_state': 'state_id'}, inplace=True)
        return sale_smry_.sort_values(by='state_id')

    def segment_identification(self, journey):
        # journey is a journey DF, with cols journey_id, init_state, state_i, state_j, count
        if self.stochastic is True:
            journeys = to_stochastic(journeys[journeys['count'] > 0.0].copy())
        self.em_e_step(journeys)  # journey_id, segment, xi (normalized), init_state
        return self.xi_df.groupby('journey_id').apply(lambda x: x[x['xi'] == x['xi'].max()]['segment'])


class Segment:
    def __init__(self, n_states, n_abs, segment, init_data, weighted, stochastic):
        """
        Assumptions:
        - states: 0,..., n_states - n_abs - 1 are transient
        - states: n_state - n_abs, ..., n_states - 1 are absorbing
        :param n_states: total number of states
        :param n_abs: number of absorbing states
        :param weighted: True: weight by journey counts, False, regular unweighted average by journey
        :param segment: segment id number
        :param stochastic: normalize journey counts
        :param init_data: initial data to initialize DTMC model
        """
        logger.info('initializing segment ' + str(segment))
        self.n_states = n_states
        self.n_abs = n_abs
        self.segment = int(segment)
        self.weighted = weighted
        self.mdl_k = False
        self.to_drop = list()
        self.stochastic = stochastic

        # absorbing matrix (identity)
        self.bmtx = pd.DataFrame(list(itertools.product(range(self.n_states - self.n_abs, self.n_states), range(self.n_states))), columns=['i_state', 'j_state'])
        self.bmtx['prob'] = self.bmtx.apply(lambda x: float(x['i_state'] == x['j_state']), axis=1)

        self.init_amtx_k(init_data)
        self.init_pi_k(init_data)
        self.init_s_df(init_data)
        self.state_adj()

    def init_pi_k(self, init_data):
        # initialize pi: init_state distribution
        i_pi = init_data[(~init_data['i_state'].isin(self.to_drop))][['journey_id', 'init_state']].drop_duplicates()
        p_dict = i_pi['init_state'].value_counts(normalize=True).to_dict()
        pi = np.zeros(self.n_states)
        init_state = np.zeros(self.n_states)
        for k in range(self.n_states):
            pi[k] = p_dict.get(k, 0.0)
            init_state[k] = k
        self.pi_df = pd.DataFrame({'prob': pi, 'init_state': init_state})
        self.pi_df['segment'] = int(self.segment)

    def init_amtx_k(self, init_data):
        amtx = self.init_amtx_k_(init_data)
        if len(self.to_drop) > 0:
            amtx = self.init_amtx_k_(init_data)

        # all together
        self.amtx = pd.concat([amtx, self.bmtx], axis=0).reset_index(drop=True)    # segment transition matrix
        self.amtx = self.amtx[self.amtx['prob'] > 0.0].copy()                      # this will drop states that have become absorbing and should not be absorbing???
        self.amtx, self.to_drop = check_stochastic(self.amtx.copy(), self.n_states, self.n_abs)
        if len(self.to_drop) > 0:
            logger.warning('segment ' + str(self.segment) + ' has invalid absorbing states:  ' + str(self.to_drop) + ' Dropping them')
        self.amtx['segment'] = int(self.segment)
        self.amtx['prob'] = self.amtx['prob'].astype(float)

    def init_amtx_k_(self, init_data):
        # initialize transition matrix: i_state, j_state, prob
        # transient mtx
        in_data = init_data[(init_data['i_state'] < self.n_states - self.n_abs) &
                            (~init_data['i_state'].isin(self.to_drop)) &
                            (~init_data['j_state'].isin(self.to_drop))].copy()
        if self.stochastic is True:
            in_data = to_stochastic(in_data[in_data['count'] > 0.0].copy())

        if self.weighted:
            zn = pd.DataFrame(in_data[['i_state', 'j_state', 'count']].groupby(['i_state', 'j_state'])['count'].sum(), columns=['count']).reset_index()
            zd = pd.DataFrame(in_data[['i_state', 'count']].groupby('i_state')['count'].sum()).reset_index()
            zd.columns = ['i_state', 'den']
            z = zn.merge(zd, on='i_state', how='left')
            z['prob'] = z['count'] / z['den']
            amtx = z[['i_state', 'j_state', 'prob']].copy()
        else:
            zn = pd.DataFrame(in_data[['journey_id', 'i_state', 'j_state', 'count']].groupby(['journey_id', 'i_state', 'j_state'])['count'].sum(), columns=['count']).reset_index()
            zd = pd.DataFrame(in_data[['journey_id', 'i_state', 'count']].groupby(['journey_id', 'i_state'])['count'].sum()).reset_index()
            zd.columns = ['journey_id', 'i_state', 'den']
            zlen = pd.DataFrame({'N': zd.groupby('i_state')['journey_id'].nunique()}).reset_index()
            zd = zd.merge(zlen, on='i_state', how='left')
            zj = zn.merge(zd, on=['journey_id', 'i_state'], how='left')
            zj['prob'] = zj['count'] / zj['den']
            zj['prob'] /= zj['N']
            amtx = zj.groupby(['i_state', 'j_state'])['prob'].sum().reset_index()
        return amtx

    def init_s_df(self, init_data):
        # s_df (posterior)
        f = init_data[(~init_data['i_state'].isin(self.to_drop)) & (~init_data['j_state'].isin(self.to_drop))].copy()
        self.s_df = f[f['init_segment'] == self.segment][['journey_id', 'init_state', 'i_state', 'j_state', 'count']].drop_duplicates()

    def e_step_k(self, mu_k, data):
        # data columns: journey_id, init_state, i_state, j_state, count
        # self.amtx.to_parquet('~/data/amtx_' + str(self.segment) + '.par')
        amtx = self.amtx[self.amtx['prob'] > 0]
        data_ = data[(~data['i_state'].isin(self.to_drop)) & (~data['j_state'].isin(self.to_drop))].copy() if len(self.to_drop) > 0 else data.copy()
        if self.stochastic is True:
            data_ = to_stochastic(data_[data_['count'] > 0.0].copy())
        # data_.to_parquet('~/data/data_' + str(self.segment) + '.par')
        s_df = data_.merge(amtx, on=['i_state', 'j_state'], how='left')
        s_df.dropna(subset=['prob'], inplace=True)  # some amtx transitions may be missing
        s_df['init_state'] = s_df['init_state'].astype(int)
        self.pi_df['init_state'] = self.pi_df['init_state'].astype(int)
        s_df = s_df.merge(self.pi_df, on='init_state', how='left')
        s_df['segment'] = int(self.segment)

        np.seterr(divide='ignore')
        s_df['l_pi_i'] = np.log(s_df['prob_y'].values)
        s_df['l_mu_k'] = np.log(mu_k) if mu_k > 0.0 else -np.inf
        s_df['l_xi_ij'] = s_df['count'] * np.log(s_df['prob_x'])
        np.seterr(divide='warn')

        z = pd.DataFrame(s_df[['journey_id', 'l_xi_ij']].groupby('journey_id').sum()).reset_index()
        s_df.drop('l_xi_ij', axis=1, inplace=True)
        s_df = s_df.merge(z, on='journey_id', how='left')
        s_df['l_N_n'] = s_df['l_mu_k'] + s_df['l_pi_i'] + s_df['l_xi_ij']

        s_df.drop(['prob_x', 'prob_y', 'segment_x', 'segment_y', 'l_pi_i', 'l_mu_k', 'l_xi_ij'], axis=1, inplace=True)
        self.s_df = s_df.copy()             # journey_id, segment, init_state, i_state, j_state, count l_N_n
        return s_df[['journey_id', 'segment', 'init_state', 'l_N_n']].drop_duplicates()  # journey_id, segment, init_state, l_N_n

    def pi_k(self, fk, qi):
        pi = fk.groupby('init_state', as_index=True)['xi'].sum()
        pi /= pi.sum()
        z = pd.concat([pi, qi], axis=1)
        z.fillna(0, inplace=True)
        pi = z.max(axis=1)
        for ix in range(self.n_states - self.n_abs, self.n_states):
            pi[ix] = 0.0
        fout = pd.DataFrame({'init_state': list(pi.index), 'prob': pi.values})
        fout = fout[~fout['init_state'].isin(self.to_drop)].copy()
        fout['prob'] = np.where(fout['prob'].values < min_val, 0.0, fout['prob'].values)
        fout['prob'] /= fout['prob'].sum()
        fout.fillna(0, inplace=True)
        self.pi_df = fout.copy()      # init_state, prob
        self.pi_df['segment'] = int(self.segment)
        return self.pi_df                   # segment, init_state, prob

    def amtx_k(self, xi_k):
        fs = self.s_df[['journey_id', 'i_state', 'j_state', 'count']].drop_duplicates()
        f = fs.merge(xi_k, on='journey_id', how='left')  # 'journey_id', 'i_state', 'j_state', 'count', xi
        f['xi*count'] = f['xi'] * f['count']

        if self.weighted:
            cf = pd.DataFrame(f.groupby(['i_state', 'j_state'])['xi*count'].sum()).reset_index()
            rf = cf.groupby('i_state')['xi*count'].sum().reset_index()  # sum over columns
            cf.rename(columns={'xi*count': 'prob'}, inplace=True)
            rf.rename(columns={'xi*count': 'prob_sum'}, inplace=True)
            amtx = cf.merge(rf, on='i_state', how='left')
            amtx = amtx[amtx['i_state'] < self.n_states - self.n_abs].copy()
            amtx['prob'] /= amtx['prob_sum']
            amtx.drop('prob_sum', axis=1, inplace=True)
            amtx['prob'].fillna(0.0, inplace=True)
        else:
            zn = pd.DataFrame(f.groupby(['journey_id', 'i_state', 'j_state'])['xi*count'].sum()).reset_index()
            zd = f.groupby(['journey_id', 'i_state'])['xi*count'].sum().reset_index()  # sum over columns
            zn.rename(columns={'xi*count': 'prob'}, inplace=True)
            zd.rename(columns={'xi*count': 'prob_sum'}, inplace=True)
            zlen = pd.DataFrame({'N': zd.groupby('i_state')['journey_id'].nunique()}).reset_index()
            zd = zd.merge(zlen, on='i_state', how='left')
            zj = zn.merge(zd, on=['journey_id', 'i_state'], how='left')
            zj['prob'] /= (zj['prob_sum'] * zj['N'])
            amtx = zj.groupby(['i_state', 'j_state'])['prob'].sum().reset_index()

        amtx = amtx.groupby('i_state').apply(trims, col='prob', min_val=1.0e-12)
        self.amtx = pd.concat([amtx, self.bmtx], axis=0).reset_index(drop=True)      # transition matrix for this segment
        self.amtx = self.amtx[self.amtx['prob'] > 0.0].copy()
        self.amtx, _ = check_stochastic(self.amtx.copy(), self.n_states, self.n_abs)
        self.amtx['segment'] = int(self.segment)
        self.amtx['prob'] = self.amtx['prob'].astype(float)
        self.state_adj()
        return self.amtx  # cols: segment, i_state, j_state, prob

    def state_adj(self):   # make sure pi has the same i-states and amtx
        if self.amtx['i_state'].nunique() != self.pi_df['init_state'].nunique():
            pi_df = self.amtx[['i_state']].drop_duplicates().merge(self.pi_df, left_on='i_state', right_on='init_state', how='left')  # this drops states in pi_df not in amtx
            pi_df['prob'].fillna(0.0, inplace=True)                                                                                   # this adds missing states in pi_df
            self.pi_df = pi_df[['i_state', 'prob', 'segment']].copy()
            self.pi_df.rename(columns={'i_state': 'init_state'}, inplace=True)
            self.pi_df['prob'] /= self.pi_df['prob'].sum()
            self.pi_df.sort_values(by='init_state', inplace=True)
            self.pi_df.reset_index(inplace=True, drop=True)

    def log_likelihood_k(self, mu_k, data):    # log_likelihood_k(k, n): segment likelihood by journey
        self.amtx['prob'] = self.amtx['prob'].astype(float)
        amtx = self.amtx[self.amtx['prob'] > 0.0]
        fa = data[['journey_id', 'i_state', 'j_state', 'count']].merge(amtx[['i_state', 'j_state', 'prob']], on=['i_state', 'j_state'], how='inner')
        fa['count * l_A'] = fa['count'] * np.log(fa['prob'])
        fa_sum = fa[['journey_id', 'count * l_A']].groupby('journey_id').sum().reset_index()
        fa_sum.rename(columns={'count * l_A': 'l_amtx'}, inplace=True)   # journey_id, s(i, j) * log(A_ij)

        pi_df = self.pi_df[self.pi_df['prob'] > 0.0]
        fp = data[['journey_id', 'init_state']].merge(pi_df[['init_state', 'prob']], on=['init_state'], how='inner')
        fp['l_pi'] = np.log(fp['prob'])
        fp.rename(columns={'init_state': 'i_state'}, inplace=True)
        fp.drop_duplicates(inplace=True)

        lf = fp.merge(fa_sum, on=['journey_id'], how='inner')
        lf['l_kn'] = lf['l_pi'] + lf['l_amtx']
        lf['mu_k * e^l_kn'] = mu_k * np.exp(lf['l_kn'])
        lf['segment'] = int(self.segment)
        return lf

    def set_mdl_k(self):
        mq = self.amtx[(self.amtx['i_state'] < self.n_states - self.n_abs) & (self.amtx['j_state'] < self.n_states - self.n_abs)]
        mr = self.amtx[(self.amtx['i_state'] < self.n_states - self.n_abs) & (self.amtx['j_state'] >= self.n_states - self.n_abs)]
        z = mq[['j_state']].drop_duplicates().merge(mr, left_on='j_state', right_on='i_state', how='left')
        mr = z[['j_state_x', 'j_state_y', 'prob']].copy()
        mr.rename(columns={'j_state_x': 'i_state', 'j_state_y': 'j_state'}, inplace=True)
        mr['prob'].fillna(0.0, inplace=True)
        mr['j_state'].fillna(self.n_states - 1, inplace=True)  # absorbing state (does not matter which one as it is never reached from i_state

        self.qmtx = pd.pivot_table(mq, index='i_state', columns='j_state', values='prob').fillna(0)
        self.rmtx = pd.pivot_table(mr, index='i_state', columns='j_state', values='prob').fillna(0)
        try:
            vmtx = np.linalg.inv(np.eye(len(self.qmtx)) - self.qmtx.values)
            self.vmtx = pd.DataFrame(vmtx, index=self.qmtx.index, columns=self.qmtx.columns)
            if self.vmtx.min().min() < 0.0:
                logger.error('invalid vmtx for segment ' + str(self.segment))
            self.mdl_k = True
            return 0
        except np.linalg.LinAlgError:
            print(self.qmtx)
            logger.error('segment:: ' + str(self.segment) + ': singular matrix')
            f = pd.DataFrame(np.eye(len(self.qmtx)) - self.qmtx.values)
            f.columns = [str(c) for c in f.columns]
            f.to_parquet('~/data/v_err.par')
            return -1

    def sale_smry_k(self):
        if self.mdl_k is False:
            ret = self.set_mdl_k()
            if ret != 0:
                return None
        vr = np.matmul(self.vmtx.values, self.rmtx.values)   # n_states x n_abs: vr[i, abs]: prob of abs from state i. we can find the states that best/worse convert
        abs_probs = np.matmul(self.pi_df[self.pi_df['init_state'] < self.n_states - self.n_abs]['prob'].values, vr)                   # 1 x n_abs
        # abs_odds = np.array([abs_probs[n] / (1.0 - abs_probs[n]) for n in range(self.n_abs)])
        sale_smry = pd.DataFrame(abs_probs, columns=['absorption_prob'])
        # sale_smry['absorption_odds'] = abs_odds
        sale_smry['absorption_state'] = range(self.n_states - self.n_abs, self.n_states)
        sale_smry['pre-absorption steps'] = np.matmul(self.pi_df[self.pi_df['init_state'] < self.n_states - self.n_abs]['prob'].values, self.vmtx.values).sum()
        sale_smry['segment'] = int(self.segment)
        return sale_smry

    def n_pars(self):
        rows = self.amtx['i_state'].nunique() - self.n_abs      # drop abs (last 2 rows are fixed)
        cols = self.amtx['j_state'].nunique() - self.n_abs - 1  # rows sum to 1
        pis = rows
        return pis + rows * cols
