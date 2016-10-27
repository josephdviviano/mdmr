#!/usr/bin/env python
"""
Finds all of the connectivity data in the directory supplied (from datman),
keeps track of the subjects, and runs a fingerprint analysis on them.

References:

Finn et al. 2015. Functional connectome fingerprinting: identifying individuals
using patterns of brain connectivity. Nature Neuroscience 18(11).

Zapala & Schork, 2006. Multivariate regression analysis of distance matrices for
testing association between gene expression patterns related variables. PNAS
103(51).
"""
import os, sys, glob, copy
import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd

# data munging
def get_database_name(subject):
    if 'P00' in subject:
        return subject
    else:
        return '_'.join(subject.split('_')[0:3])

def get_diagnosis(string):
    if '1' in string:
        return 'C'
    elif '2' in string:
        return 'S'
    else:
        return ''

def scrub_data(x):
    """
    Removes NaNs from a vector, and raises an exception if the data is empty.
    """
    x[np.isnan(x)] = 0
    if np.sum(x) == 0:
        raise Exception('vector contains no information')

    return x

def parse_data(data, variables):
    """Parses SPINS data from datman and redcap."""

    subjects, diagnosis, X, Y = [], [], [], []

    for i, subject in enumerate(data):

        # skip subject if either x or y are empty
        try:
            x = scrub_data(data[subject]['X'])
            y = scrub_data(data[subject]['Y'])
        except:
            continue

        # stack rows
        s = get_database_name(subject)
        if len(X) == 0:
            X = x
            Y = r_to_z(y)
        else:
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))

        subjects.append(subject)
        diagnosis.append(get_diagnosis(variables.loc[s]['redcap_event_name']))

    diagnosis = np.array(diagnosis)

    return X, Y, subjects, diagnosis

# statistics
def r_to_z(R):
    """Fischer's r-to-z transform on a matrix (elementwise)."""
    return(0.5 * np.log((1+R)/(1-R)))

def r_to_d(R):
    """Converts a correlation matrix R to a distance matrix D."""
    return(np.sqrt(2*(1-R)))

def assert_square(X):
    if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
        raise Exception("Input matrix must be square")

def full_rank(X):
    """Ensures input matrix X is not rank deficient."""
    k = X.shape[1]
    rank = np.linalg.matrix_rank(X)
    if rank < k:
        return False
    return True

def sig_cutoffs(null, two_sided=True):
    """Returns the significance cutoffs of the submitted null distribution."""
    return(np.array([np.percentile(F_null, 5), np.percentile(F_null, 95)]))

def reorder(X, idx, symm=False):
    """
    Reorders the rows of a matrix. If symm is True, this simultaneously reorders
    the columns and rows of a matrix by the given index.
    """
    if symm:
        assert_square(X)
        X = X[:, idx]
    X = X[idx, :]

    return(X)

def gowers_matrix(D):
    """Calculates Gower's centered matrix from a distance matrix."""

    assert_square(D)

    n = float(D.shape[0])
    I = np.identity(n) - (1/n)*np.ones((n, 1))*(np.ones((n, 1)).T)
    A = -0.5*np.square(D)
    G = I*A*I

    return(G)

def hat_matrix(X):
    """
    Caluclates distance-based hat matrix for an NxM matrix of M predictors from
    N variables. Adds the intercept term for you.
    """
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # add intercept
    Q1, R1 = np.linalg.qr(X)
    H = Q1.dot(Q1.T)

    return(H)

def calc_F(H, G, m=None):

    n = H.shape[0]
    I = np.identity(n)
    IG = I-G

    if m:
        F = (np.trace(H.dot(G).dot(H)) / (m-1)) / (np.trace(IG.dot(G).dot(IG)) / (n-m))
    else:
        F = (np.trace(H.dot(G).dot(H))) / np.trace(IG.dot(G).dot(IG))

    return F


def permute(H, G, n=10000):
    """
    Calculates a null F distribution from a symmetrically-permuted G (Gower's
    matrix), from the between subject connectivity distance matrix D, and a the
    H (hat matrix), from the original behavioural measure matrix X.

    The permutation test is accomplished by simultaneously permuting the rows
    and columns of G and recalculating F. We do not need to account for degrees
    of freedom when calculating F.
    """
    F_null = np.zeros(n)
    idx = np.arange(G.shape[0]) # generate our starting indicies

    for i in range(n):
        idx = np.random.permutation(idx)
        G_perm = reorder(G, idx, symm=True)
        F_null[i] = calc_F(H, G_perm)

    F_null.sort()

    return F_null

def mdmr(X, Y):
    """
    Multvariate regression analysis of distance matricies: regresses variables
    of interest X (behavioural) onto a matrix representing the similarity of
    connectivity profiles Y.
    """

    if not full_rank(X):
        raise Exception('X is not full rank:\ndimensions = {}'.format(X.shape))

    n = X.shape[0]       # number of subjects
    m = X.shape[1]       # number of predictors per subject
    R = np.corrcoef(Y)   # correlations of Z-scored correlations, as in Finn et al. 2015.
    D = r_to_d(R)        # distance matrix of correlation matrix
    G = gowers_matrix(D) # centered distance matrix (connectivity similarities)
    H = hat_matrix(X)    # hat matrix of regressors (cognitive variables)
    F = calc_F(H, G)     # F test of relationship between regressors and distance matrix
    F_null = permute(H, G)

    return F, F_null

def cluster():

    # hierarchical clustering
    fig = plt.figure()
    axd = fig.add_axes([0.09,0.1,0.2,0.8])
    axd.set_xticks([])
    axd.set_yticks([])
    link = sch.linkage(R, method='ward')
    clst = sch.fcluster(link, 3, criterion='maxclust')

    # plot
    dend = sch.dendrogram(link, orientation='right')
    idx = dend['leaves']
    R = reorder(R, idx, symm=True)
    subjects = np.asarray(subjects)[idx] # reorder subjects
    axm = fig.add_axes([0.3,0.1,0.6,0.8])
    im = axm.matshow(R, aspect='auto', origin='lower', cmap=plt.cm.Reds, vmin=0.3, vmax=0.6)
    axm.set_xticks(np.arange(len(subjects)))
    axm.set_yticks([])
    axm.set_xticklabels(subjects)
    axc = fig.add_axes([0.91,0.1,0.02,0.8])
    plt.colorbar(im, cax=axc)
    plt.show()

if __name__ == '__main__':
    """
    Collects data, runs MDMR analysis, runs cluster analysis.
    """
    columns = ['redcap_event_name',
               'scog_tasit_p1_total_positive',
               'scog_tasit_p1_total_negative',
               'scog_tasit_p2_total',
               'scog_tasit_p3_total',
               'scog_rmet_total',
               'scog_rad_total',
               'sans_total_sc',
               'wtar_std_score',
               'demo_age_study_entry',
               'demo_sex_birth']

    columns = ['redcap_event_name',
               'scog_tasit_p2_total',
               'scog_tasit_p3_total']
    # mri data
    nii_dir = '/archive/data/SPINS/pipelines/fmri/rest'
    candidates = glob.glob(nii_dir + '/*/*_roi-corrs.csv')

    # clinical data
    spreadsheet = '/projects/jviviano/data/spins/meeting/2yr/16-09-2016_jdv.csv'
    variables = pd.read_csv(spreadsheet, index_col=0)
    variables = variables[columns]

    # import
    data = {}
    for candidate in candidates:

        subject = os.path.basename(os.path.dirname(candidate))

        # X = clinical data
        s = get_database_name(subject)
        try:
            predictors = np.array(variables.loc[s][1:].astype(np.float))
        except:
            print('{} not found in {}'.format(s, spreadsheet))
            continue

        # Y = mri data -- upper triangle of each z-scored correlation matrix
        conn = np.genfromtxt(candidate, delimiter=',')
        conn = conn[np.triu_indices(conn.shape[0], 1)]
        data[subject] = {'Y': conn, 'X': predictors}

    X, Y, subjects, diagnosis = parse_data(data, variables)
    print('subjects: {} found, {} clean'.format(len(data.keys()), X.shape[0]))

    # mdmr analysis
    F, F_null = mdmr(X, Y)
    thresholds = sig_cutoffs(F_null)
    if F < thresholds[0]:
        print('F significant, F={} < {}'.format(F, thresholds[0]))
    elif F > thresholds[1]:
        print('F significant, F={} > {}'.format(F, thresholds[1]))

