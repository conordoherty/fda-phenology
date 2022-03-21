import numpy as np
from numba import njit
from scipy.linalg import eig

def make_class_means(x, y, class_labels):
    class_means = np.zeros((class_labels.size, x.shape[1]), dtype='float64')
    for j, label in enumerate(class_labels):
        class_means[j] = x[y==label, :].mean(axis=0)

    return class_means

@njit
def make_between_scatter(x, y, class_labels, mu, mu_j):
    s_b = np.zeros((mu.size, mu.size))
    for j, label in enumerate(class_labels):
        s_b += (y==label).sum()*np.outer(mu_j[j]-mu, mu_j[j]-mu)

    return s_b


@njit
def make_within_scatter(x, y, class_labels, mu_j):
    s_w = np.zeros((x.shape[1], x.shape[1]))
    for j, label in enumerate(class_labels):
        x_class = x[y==label, :]
        for i in range(x_class.shape[0]):
            s_w += np.outer(x_class[i]-mu_j[j], x_class[i]-mu_j[j])

    return s_w


def make_fda_vecs(x, y, alpha=0, num_vecs=1):
    mu = x.mean(axis=0)

    class_labels = np.unique(y)
    mu_j = make_class_means(x, y, class_labels)

    s_w = make_within_scatter(x, y, class_labels, mu_j)
    s_b = make_between_scatter(x, y, class_labels, mu, mu_j)
    s_b = s_b+np.diag(np.ones(s_b.shape[0])*alpha)

    e = eig(s_b, s_w)
    idx = e[0].argsort()[::-1]
    e_vecs = e[1][:, idx]
    e_vecs = np.real(e_vecs)[:, :num_vecs]

    return e_vecs
