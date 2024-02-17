# Author: Proloy Das <pdas6@mgh.harvard.edu>
# implements utils for the Data Driven Parcellation procedure from
# Citation: Chowdhury RA, Lina JM, Kobayashi E, Grova C (2013) MEG Source
# Localization of Spatially Extended Generators of Epileptic Activity: Comparing Entropic
# and Hierarchical Bayesian Approaches. PLoS ONE 8(2): e55969. 
# https://doi.org/10.1371/journal.pone.0055969
from copy import deepcopy
from distutils.log import warn
import itertools
from operator import le
import numpy as np
from purdonlabmeeg.fixes import (stable_cumsum, _safe_svd)
# from ..fixes import (_safe_svd, stable_cumsum)


def _apply_msp(inst, tmin, tmax, fwd, n_components=0.95, ):
    """multivariate source pre-localization method

    returns assignment of the sources to signal subspace of 
    `n_components` dimensions + noise dimension)"""
    from scipy.sparse import block_diag
    from mne.inverse_sparse.mxne_inverse import is_fixed_orient

    leadfield = fwd['sol']['data'].copy()
    sensor_ts = inst.get_data(tmin=tmin, tmax=tmax).copy()
    for x in (leadfield, sensor_ts):
        x /= np.sqrt(np.sum(x * x, axis=-1, keepdims=True))

    if sensor_ts.ndim == 2:
        sensor_ts = [sensor_ts]
    uncovs, ts = list(zip(*[(this_sensor_ts.dot(this_sensor_ts.T), this_sensor_ts.shape[-1])
                          for this_sensor_ts in sensor_ts]))
    cov = np.sum(np.stack(uncovs), axis=0) / np.sum(ts)

    U, S, V = _safe_svd(cov, full_matrices=False)

    # Get variance explained by singular values
    if not isinstance(n_components, int):
        explained_variance_ = S.copy()
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        ratio_cumsum = stable_cumsum(explained_variance_ratio_)
        n_components = np.searchsorted(ratio_cumsum, n_components)
    else:
        n_components = max(n_components-1, 1)
    # U *= S[None, :]
    xeta = leadfield.T.dot(U) ** 2
    # Collapse on the dipoles at each vertex 
    n_orient = 1 if is_fixed_orient(fwd) else 3
    n_vert = sum([len(s['vertno']) for s in fwd['src']])
    hemi_ = [len(s['vertno']) for s in fwd['src']] # lh -> [:hemi_[0]], rh -> [-hemi_[1]:]
    assert n_orient * n_vert == xeta.shape[0]
    M = block_diag([np.ones((1, n_orient), )] * n_vert)
    xeta = M.dot(xeta)

    assignments = np.argmax(xeta, axis=-1)
    B = np.concatenate((xeta[:, :n_components],
                        np.sum(xeta[:, n_components:], axis=-1, keepdims=True)),
                       axis=-1)
    assignments[assignments >= n_components] = n_components
    
    pre_clusters = []
    for ii in range(n_components+1):
        # split pre-clusters based on hemi
        pre_cluster, *rest = np.nonzero(assignments==ii)
        jj = np.searchsorted(pre_cluster, hemi_[0])
        pre_cluster.partition(jj-1)
        for pre_cluster_h in np.split(pre_cluster, [jj,]):
            if len(pre_cluster_h):
                elem = list(pre_cluster_h)
                elem.sort(reverse=True, key=lambda x: B[x, ii])
                pre_clusters.append(elem)
    # pre_clusters.append(list(set(range(n_vert)).difference(*pre_clusters)))        
    return pre_clusters, assignments


def _as_mne_labels(parcels, src):
    "src is surface sourcespace"
    nvert = [len(s['vertno']) for s in src] 
    vertno = np.concatenate([s['vertno'] for s in src], axis=-1)
    # for s in src:
    #     nvert.append(len(s['vertno']))
    #     vertno.append(s['vertno'])
    hemi_boundaries = stable_cumsum(nvert)
    labels = []
    for parcel in parcels:
        parcel = sorted(list(parcel))
        vmin, vmax = parcel[0], parcel[-1]
        if vmin < hemi_boundaries[0] and vmax >= hemi_boundaries[0]:
            raise ValueError(f'Inner hemi connections are not allowed! Check your vertno.')
        if vmin >= hemi_boundaries[0]:
            hemi = 'rh'
        elif vmax < hemi_boundaries[0]:
            hemi = 'lh'
        vertices = vertno[np.asanyarray(parcel)]
        label = mne.Label(vertices, hemi=hemi)
        # label = label.fill(src)
        labels.append(label)
    return labels


def find_seeds(pre_cluster, adjacency, order=3):
    s_seeds = []
    p_seeds = set(pre_cluster)
    for j in pre_cluster:
        if len(p_seeds) == 0:
            break
        if j not in p_seeds or len(p_seeds.intersection(adjacency.rows[j])) < 1:
            continue
        s_seeds.append(j)
        # Find k-neighbors
        kneighs = set([j])
        for s in range(order):
            kneighs = kneighs.union(*[adjacency.rows[i] for i in kneighs])
        # Remove all the k-neighbors from potential seeds
        p_seeds = p_seeds.difference(kneighs)
    return s_seeds


def region_grow1(s_seeds, pre_cluster, adjacency, verbose=1):
    "Just raw neighbor based region growth"
    regions = [[i] for i in s_seeds]
    pc = set(pre_cluster)
    pc = pc.difference(s_seeds)
    n_assigned = 0
    remaining = len(pc)
    while len(pc) > 0:
        # Loop through the regions
        for elem in regions:
	    # look for immediate neighbors
            neighs = set([])
            neighs = neighs.union(*[adjacency.rows[i] for i in elem])
	    # check which neighbors are not assigned yet
            neighs = pc.intersection(neighs)
	    # Add them to the region
            elem.extend(list(neighs))
            # Reomve assigned vertices
            pc = pc.difference(neighs)
        if remaining > len(pc):
            n_assigned += remaining - len(pc)
            if verbose == 1:
                print(f'{remaining - len(pc)} vertices were assiged,'
                    f'remaining {len(pc)} vertices')
            remaining = len(pc)
        else:
            if verbose == 1:
                print(f'{len(pc)} vertices remain unassigned.')
            break
    
    print(f'TOTAL: {n_assigned} vertices were assiged,'
            f'remaining {len(pc)} vertices')
    return regions


def region_grow2(s_seeds, pre_cluster, adjacency, verbose=1):
    "Voting based region growth."
    pc = set(pre_cluster)
    pc = pc.difference(s_seeds)
    regions = [set([i]) for i in s_seeds]
    n_assigned = 0
    while len(pc) > 0:
        assigned = []
        for v in pc:
            votes = [len(elem.intersection(adjacency.rows[v])) for elem in regions]
            w = max(range(len(votes)), key=lambda i:votes[i])
            if votes[w] > 0:
                regions[w] = regions[w].union([v])
                assigned.append(v)
            else:
                if verbose == 1:
                    print(f'{v} does not have any neighbors')
        if len(assigned) == 0:
            if verbose == 1:
                print(f'{len(pc)} vertices remain unassigned')
            break
        else:
            pc = pc.difference(assigned)
            n_assigned += len(assigned)
            if verbose == 1:
                print(f'{len(assigned)} vertices are assigned,'
                    f'remaining {len(pc)} vertices')
    print(f'TOTAL: {n_assigned} vertices were assiged,'
          f'remaining {len(pc)} vertices')
    return regions


def _grow_nonoverlapping_labels(seeds, src, graph, assigns, subject, subjects_dir):
    import mne
    "src is surface sourcespace"
    nvert = [len(s['vertno']) for s in src] 
    hemi_boundaries = stable_cumsum(nvert)
    vertices = np.concatenate([s['vertno'] for s in src], axis=-1)
    n_vertices = len(vertices)
    n_labels = len(seeds)
    assert n_vertices == hemi_boundaries[-1]

    # Prepare parcellation
    parc = np.empty(n_vertices, dtype='int32')
    parc[:] = -1

    sources = {} # vert -> (label, assignment)
    edge = []  # queue of vertices to process
    for label, seed in enumerate(seeds):
        if np.any(parc[seed] >= 0):
            raise ValueError("Overlapping seeds")
        parc[seed] = label
        for s in np.atleast_1d(seed):
            sources[s] = (label, assigns[s])
            edge.append(s)
    
    while edge:
        vert_from = edge.pop(0)
        label, from_assignment = sources[vert_from]
        for vert_to, dist in zip(graph.rows[vert_from], graph.data[vert_from]):
            # Prevent adding a point that has already been used
            # (prevents infinite loop) #TODO
            if dist == 0: continue
            vert_to_label = parc[vert_to]
            if vert_to_label >= 0:
                continue
            
            to_assignment = assigns[vert_to]
            if to_assignment != from_assignment:
                continue
                  
            # assign label value
            parc[vert_to] = label
            sources[vert_to] = (label, to_assignment)
            edge.append(vert_to)
    
    singular_labels = []
    while np.count_nonzero(parc < 0):
        # Remove the one element parcellations, and replace
        # them according to neighbors vote
        for label in range(n_labels):
            members = np.nonzero(parc == label)[0]
            if len(members) != 1:
                if len(members) > 1:
                    while label in singular_labels:
                        singular_labels.remove(label)
                continue
            s = int(np.where(parc == label)[0])
            neighbors = graph.rows[s].copy()
            neighbors.remove(s)
            neighbors_assign = parc[np.array(neighbors)]
            unique = np.unique(neighbors_assign)
            # new_label = max(unique, key=lambda x: np.count_nonzero(neighbors_assign==x))
            unique = sorted(unique, key=lambda x: np.count_nonzero(neighbors_assign==x), reverse=True)
            new_label = unique[0]
            # if label == 14: 
            #     import ipdb; ipdb.set_trace()
            if new_label < 0:   # if label < 0, fall back to the next one.
                try:
                    # if np.count_nonzero(neighbors_assign==unique[0]) == np.count_nonzero(neighbors_assign==unique[1]):
                    new_label = unique[1]
                except IndexError:
                    continue
            singular_labels.append(label)
            parc[s] = new_label
            sources[s] = (new_label, assigns[s])
            edge.append(s)

        # Now disregard the assignments for region growing    
        while edge:
            vert_from = edge.pop(0)
            label, from_assignment = sources[vert_from]
            for vert_to, dist in zip(graph.rows[vert_from], graph.data[vert_from]):
                # Prevent adding a point that has already been used
                # (prevents infinite loop) #TODO
                if dist == 0: continue
                vert_to_label = parc[vert_to]
                if vert_to_label >= 0:
                    continue
                # assign label value
                neighbors_assign = parc[np.array(graph.rows[vert_to])]
                unique = np.unique(neighbors_assign)
                label = max(unique, key=lambda x: np.count_nonzero(neighbors_assign==x))
                if label < 0:
                    continue
                parc[vert_to] = label
                sources[vert_to] = (label, to_assignment)
                edge.append(vert_to)

        # Now take care of the reamining still unassigned vertices, 
        # based on neighboring votes
        for s in np.nonzero(parc < 0)[0]:
            neighbors = graph.rows[s].copy()
            neighbors.remove(s)
            neighbors_assign = parc[np.array(neighbors)]
            unique = np.unique(neighbors_assign)
            unique = sorted(unique, key=lambda x: np.count_nonzero(neighbors_assign==x), reverse=True)
            label = unique[0]
            if label < 0:   # if label < 0, fall back to the next one.
                try:
                    # if np.count_nonzero(neighbors_assign==unique[0]) == np.count_nonzero(neighbors_assign==unique[1]):
                    label = unique[1]
                except IndexError:
                    continue
            parc[s] = label
            sources[s] = (label, assigns[s])
            edge.append(s)
    
    # removal of the singular labels from parc
    counts = [np.count_nonzero(parc==k) for k in range(n_labels)]
    new_parc = - np.ones_like(parc)
    new_k = 0
    for old_k in range(n_labels):
        if np.count_nonzero(parc == old_k) > 1: 
            new_parc[parc == old_k] = new_k
            new_k += 1
    assert new_k == n_labels - len(np.unique(singular_labels))
    n_labels = new_k
    parc = new_parc

    # new_parc = parc.copy()
    # for k, label in enumerate(sorted(singular_labels)):
    #     new_parc[parc >= label] -= 1        # (s-i) <- s tracks removal of earlier labels
    # assert n_labels == new_parc.max() + 1
    # old_parc = parc.copy()
    # n_labels -= len(singular_labels)
    # parc = new_parc
    print(np.count_nonzero(parc<0))
    # Convert prac to label:
    labels = [[], []]
    for i in range(n_labels):
        this_parc = np.nonzero(parc == i)[0]
        vmin, vmax = this_parc[0], this_parc[-1]
        if vmin < hemi_boundaries[0] and vmax >= hemi_boundaries[0]:
            raise ValueError(f'Inner hemi connections are not allowed! Check your vertno.')
        if vmin >= hemi_boundaries[0]:
            hemi = 'rh'
        elif vmax < hemi_boundaries[0]:
            hemi = 'lh'
        this_vertices = vertices[this_parc]
        label_ = mne.Label(this_vertices, hemi=hemi, name=f'{i}', subject=subject)
        labels[int(hemi == 'rh')].append(label_)
    for _labels in labels:
        _labels.sort(key=lambda label: label.center_of_mass(subjects_dir=subjects_dir))
    labels = list(itertools.chain(*labels))
    return labels, parc


def _grow_sequential_labels(pre_clusters, src, graph, extent, assigns, subject, subjects_dir):
    """
    pre_clusters: list of lists
        the individual lists are already (descendingly) sorted accroding to B
    assigns: array
        assignments to the pre-clusters 
    """
    import mne
    "src is surface sourcespace"
    nvert = [len(s['vertno']) for s in src] 
    hemi_boundaries = stable_cumsum(nvert)
    vertices = np.concatenate([s['vertno'] for s in src], axis=-1)
    n_vertices = len(vertices)
    assert n_vertices == hemi_boundaries[-1]
    parc = np.empty(n_vertices, dtype='int32')
    parc[:] = -1

    seeds = []
    # initialize active sources
    sources = {}  # vert -> (label, dist_from_seed)
    edge = []  # queue of vertices to process
    label = -1  # each time we pop a source pre_cluster, increase this counter
    labels_end = label + 1
    for _pre_cluster in pre_clusters:
        pre_cluster = deepcopy(_pre_cluster)
        labels_begin = labels_end 
        while pre_cluster:
            seed = pre_cluster.pop(0)
            if np.any(parc[seed] >= 0):
                raise ValueError("Overlapping seeds")
            label += 1
            parc[seed] = label
            seeds.append(seed)
            sources[seed] = (assigns[seed], 0.)
            edge.append(seed)
        
            # grow from current source
            while edge:
                vert_from = edge.pop(0)
                assign, old_dist = sources[vert_from]

                # add neighbors within allowable distance
                for vert_to, _ in zip(graph.rows[vert_from], graph.data[vert_from]):
                    # Prevent adding a point that has already been used
                    # (prevents infinite loop)
                    if (vert_to == seeds[label]).any():
                        continue
                    new_dist = old_dist + 1     # neighborhood distance, not actual distance 

                    # abort if outside of extent
                    if new_dist > extent:
                        continue

                    vert_to_assign = assigns[vert_to]
                    if vert_to_assign != assign:
                        continue

                    vert_to_label = parc[vert_to]
                    if vert_to_label >= 0:
                        # abort if the vertex is already occupied
                        continue
                        # _, vert_to_dist = sources[vert_to]
                        # # abort if the vertex is occupied by a closer seed
                        # if new_dist > vert_to_dist:
                        #     continue
                        # elif vert_to in edge:
                        #     edge.remove(vert_to) 

                    # assign label value
                    parc[vert_to] = label
                    sources[vert_to] = (assign, new_dist)
                    edge.append(vert_to)

                    # Remove the assigned dipoles from the pre_cluster
                    pre_cluster.remove(vert_to)
            
        # Pruning
        # Remove parcellations with < extent dipoles, and merge them 
        # to neighboring closed cluster
        labels_end = label + 1
        for ll in range(labels_begin, labels_end):
            members = np.nonzero(parc == ll)[0]
            if len(members) <= extent:
                neighbors = set([]).union(*[graph.rows[s] for s in members])
                neighbors = neighbors.difference(members)
                ss_assign, _ = sources[members[0]]
                filtered_nns = filter(lambda x: sources.get(x, (-1,))[0] == ss_assign, neighbors)
                new_labels = list(sorted(filtered_nns, key = lambda x: sources[x][1]))  # sorted distance
                if new_labels:
                    # parc[np.array(members)] = parc[new_labels[0]]
                    for s in members: parc[s] = parc[new_labels[0]]
                else:
                    warn(f'isolated sources')
    
    n_labels = label + 1   # last label index + 1    
    
    # Remove the parcellations with < extent dipoles, and replace
    # them according to neighbors vote
    for label in range(n_labels):
        members = np.nonzero(parc == label)[0]
        if len(members) < 1:
            continue
        elif len(members) > extent:
            continue
        # s = int(members)
        # neighbors = graph.rows[s].copy()
        # neighbors.remove(s)
        neighbors = set().union(*[graph.rows[s] for s in members])
        neighbors = neighbors.difference(members)
        neighbors_assign = parc[list(neighbors)]
        unique = np.unique(neighbors_assign)
        unique = sorted(unique, key=lambda x: np.count_nonzero(neighbors_assign==x), reverse=True)
        new_label = unique[0]
        if new_label < 0:   # if label < 0, fall back to the next one.
            try:
                # if np.count_nonzero(neighbors_assign==unique[0]) == np.count_nonzero(neighbors_assign==unique[1]):
                new_label = unique[1]
            except IndexError:
                continue
        for s in members: parc[s] = new_label

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.hist(parc, bins=np.arange(-1, parc.max(), 1))
    # fig.show()
        
    # Convert prac to label:
    new_parc = np.empty_like(parc)
    new_parc[:] = -1
    new_k = 0
    for old_k in range(n_labels):
        if np.count_nonzero(parc == old_k) > 0: 
            new_parc[parc == old_k] = new_k
            new_k += 1
    n_labels = new_k
    parc = new_parc
    print(np.count_nonzero(parc<0), n_labels, parc.max())

    labels = [[], []]
    for i in range(n_labels):
        this_parc = np.nonzero(parc == i)[0]
        vmin, vmax = this_parc[0], this_parc[-1]
        if vmin < hemi_boundaries[0] and vmax >= hemi_boundaries[0]:
            raise ValueError(f'Inner hemi connections are not allowed! Check your vertno.')
        if vmin >= hemi_boundaries[0]:
            hemi = 'rh'
        elif vmax < hemi_boundaries[0]:
            hemi = 'lh'
        this_vertices = vertices[this_parc]
        label_ = mne.Label(this_vertices, hemi=hemi, name=f'{i}', subject=subject)
        labels[int(hemi == 'rh')].append(label_)
    for _labels in labels:
        _labels.sort(key=lambda label: label.center_of_mass(subjects_dir=subjects_dir))
    labels = list(itertools.chain(*labels))
    return labels, parc


def data_driven_parcellation(epochs, fwd, tmin, tmax, orders, subjects_dir, subject):
    from mne import spatial_src_adjacency
    pre_clusters, assigns = _apply_msp(epochs, tmin, tmax, fwd, n_components=0.4, )
    adjacency = spatial_src_adjacency(fwd['src'])
    adjacency = adjacency.tolil()
    out = dict()
    for order in orders:
        # s_seeds = [find_seeds(pre_cluster, adjacency, order=order) for pre_cluster in pre_clusters]
        # s_seeds = list(itertools.chain.from_iterable(s_seeds))
        # labels, parc = _grow_nonoverlapping_labels(s_seeds, fwd['src'], adjacency, assigns,
        #                                            subject, subjects_dir) 
        labels, parc = _grow_sequential_labels(pre_clusters, fwd['src'], adjacency, 
                                               order, assigns, subject, subjects_dir)
        nparc = parc.max() + 1
        out[f'{order}'] = dict(labels=labels, parc=parc, nparc=nparc)
    return out


def format_component(i, parc, smoothness_weights):
    """Returns components to be used in hierarchial ReML

    Components, Q1k or Q2k are represented as `(data, (i, j))`,
    as in coo_array so that they can be readily converted to sparse
    matrices:
                `Qij = coo_array(Qij, shape=(m, m))`
    with (i, j) indices are indices wrt to y for Q1, X1 theta1 for Q2.
    """
    # Q2 = smoothness_weights.tolil().copy()
    # Q2[parc != i][:,  parc != i] = 0
    # Q2 = Q2.tocoo()
    # return (Q2.data, (Q2.row, Q2.col))
    import scipy.sparse as sparse
    rows = np.nonzero(parc == i)[0]
    data = smoothness_weights[rows][:, rows].toarray().ravel()
    data = smoothness_weights[np.ix_(rows, rows)].toarray().ravel()
    row_idx, col_idx = [np.array(ii, dtype=np.int64) for ii in zip(*list(itertools.product(rows, repeat=2)))]
    mat = sparse.csr_array((data, (row_idx, col_idx)))
    cov = mat.T.dot(mat)
    cov = cov.tocoo()
    data = cov.data
    row_idx = cov.row
    col_idx = cov.col
    return data, (row_idx, col_idx)


if __name__ == "__main__":
    import mne
    from mne.datasets import sample
    from numpy.random import default_rng
    import matplotlib.pyplot as plt
    
    data_path = sample.data_path()
    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
    event_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'
    trans = meg_path / 'sample_audvis_raw-trans.fif'

    subjects_dir = data_path / 'subjects'
    bem_sol_fname = subjects_dir / 'sample' / 'bem' / 'sample-5120-5120-5120-bem-sol.fif'
    src_fname = subjects_dir / 'sample' / 'bem' / 'sample-fsaverage-ico-5-src.fif'
    all_src_fname = subjects_dir / 'sample' / 'bem' / 'sample-all-src.fif'
    fwd_path = meg_path  / 'sample_audvis-meg-oct-6-fwd.fif'

    tmin = -0.2
    tmax = 0.3  # Use a lower tmax to reduce multiple comparisons

    #   Setup for reading the raw data
    raw = mne.io.read_raw_fif(raw_fname)
    events = mne.read_events(event_fname)


    raw.info['bads'] += ['MEG 2443']
    picks = mne.pick_types(raw.info, meg=True, eog=True, exclude='bads')
    event_id = [3, 4]  # L auditory
    reject = dict(grad=1000e-13, mag=4000e-15, eog=150e-6)
    epochs1 = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), reject=reject, preload=True)
    
    src = mne.setup_source_space('fsaverage', spacing='ico4', subjects_dir=subjects_dir,
                                 n_jobs=-1, add_dist=False)
    src = mne.morph_source_spaces(src, 'sample', subjects_dir=subjects_dir)
    # src = mne.read_source_spaces(src_fname)
    bem = mne.read_bem_solution(bem_sol_fname)
    fwd = mne.make_forward_solution(raw.info, trans, src=src, bem=bem,
                                    meg=True, eeg=False, n_jobs=-1)

    # fwd = mne.read_forward_solution(fwd_path)
    # src = fwd['src']
    # fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
    #                                         use_cps=True)
    # pre_clusters, assigns = _apply_msp(epochs1, 0.05, 0.15, fwd, n_components=0.4, )

    adjacency = mne.spatial_src_adjacency(fwd['src'])
    adjacency = adjacency.tolil()
    assert adjacency.shape[0] == fwd['sol']['data'].shape[-1] // 3

    # parcels6 = []
    # for pre_cluster in pre_clusters:
    #     s_seeds = find_seeds(pre_cluster, adjacency, order=10)
    #     parcels6.extend(region_grow2(s_seeds, pre_cluster, adjacency, verbose=0))
    # labels = _as_mne_labels(parcels6, fwd['src'])
    
    fig, axes = plt.subplots(ncols=3, figsize=(5.8*3, 7.2))
    for order, ax in zip([7, 5, 3], axes):
        # s_seeds = []
        # for pre_cluster in pre_clusters:
        #     s_seeds.extend(find_seeds(pre_cluster, adjacency, order=order))
        # labels, parc = _grow_nonoverlappq
        # g_labels(s_seeds, fwd['src'], adjacency, assigns,
        #                                         'sample', subjects_dir)
        pre_clusters, assigns = _apply_msp(epochs1, 0.05, 0.15, fwd, n_components=0.4, )
        labels, parc = _grow_sequential_labels(pre_clusters, fwd['src'], adjacency, order, assigns,
                                               'sample', subjects_dir)
            
        # Viz
        Brain = mne.viz.get_brain_class()
        # from surfer import Brain
        # brain = Brain('sample', 'rh', 'inflated', subjects_dir=subjects_dir,
        #             cortex='low_contrast', alpha=0.1, background='white', size=(800, 800))
        # brain.add_data(assigns[-fwd['src'][1]['nuse']:], vertices=fwd['src'][1]['vertno'],
        #                colormap='gist_ncar')
        # brain.close()

        brain = Brain('sample', 'both', 'white', subjects_dir=subjects_dir,
                    cortex='low_contrast', alpha=0.1, background='white', size=(800, 800),)
        import eelbrain
        colors = eelbrain.plot.colors_for_oneway(labels, cmap='prism', light_cycle=5,
                                                    always_cycle_hue=True)
        for label in labels:
            brain.add_label(label.fill(fwd['src']), color=colors[label], borders=False)
        brain.show_view(view='dorsal', distance=400)
        screenshot = brain.screenshot()

        # Strip the white portions
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

        ax.imshow(cropped_screenshot)
        ax.axis('off')
    fig.show()
    fig.savefig('Fig-DPP.png')
