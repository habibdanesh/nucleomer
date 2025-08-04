import os
import pickle
import csv
from tqdm import tqdm
import numpy as np
import networkx as nx
from functools import lru_cache

from .utils import read_fasta


def generate_dinuc_matrices(seqs, nuc_idx={'A': 0, 'C': 1, 'G': 2, 'T': 3},
                            verbose=False):
    """
    Generate dinucleotide matrices for a list of sequences.
    
    Parameters
    ----------
    seqs: list of str
        List of sequences for which to generate dinucleotide matrices.
    nuc_idx: dict, optional
        Dictionary mapping nucleotide characters to their indices in the matrix. 
        Defaults to {'A': 0, 'C': 1, 'G': 2, 'T': 3}.
    verbose: bool, optional
        If True, print progress information. Defaults to False.
    
    Returns
    -------
    dinuc_matrix: np.ndarray
        A 3D numpy array of shape (n_seqs, 4, 4) containing the dinucleotide matrices.
    """

    dinuc_matrix = np.zeros((len(seqs), 4, 4), dtype=np.uint8)
    for i, seq in enumerate(tqdm(seqs, desc="Generating dinucleotide matrices", unit="kmer") if verbose else seqs):
        for j in range(len(seq) - 1):
            dinuc_matrix[i, nuc_idx[seq[j]], nuc_idx[seq[j + 1]]] += 1
    return dinuc_matrix


def generate_dinuc_classes(dinuc_matrix, verbose=False):
    """
    Generate dinucleotide classes from a dinucleotide matrix.
    
    Parameters
    ----------
    dinuc_matrix: np.ndarray
        A 3D numpy array of shape (n_seqs, 4, 4) containing the dinucleotide matrices.
    verbose: bool, optional
        If True, print progress information. Defaults to False.
    
    Returns
    -------
    dinuc_classes: list[list[int]]
        A list of lists, where each inner list contains indices of dinucleotides that belong to the same class.
    """
        
    dinuc_classes = [[0]] # Add the first kmer to the first class
    for i in tqdm(range(1, len(dinuc_matrix)), desc="Generating dinucleotide classes", unit="kmer") if verbose \
            else range(1, len(dinuc_matrix)):
        i_mat = dinuc_matrix[i] # shape(4, 4)
        # Go through all the existing classes
        existing_class = False
        for c_idx, c_list in enumerate(dinuc_classes):
            j_mat = dinuc_matrix[c_list[0]] # shape(4, 4), take a representative dinuc matrix from the class
            if np.array_equal(i_mat, j_mat):
                dinuc_classes[c_idx].append(i) # Add the kmer to the existing class
                existing_class = True
                break
        if not existing_class:
            # If no existing class was found, create a new one and add the kmer to it
            dinuc_classes.append([i])
    if verbose:
        print(f"# dinucleotide classes: {len(dinuc_classes)}")
    return dinuc_classes


def fast_generate_dinuc_classes(dinuc_matrix, verbose=False):
    """
    Generate dinucleotide classes from a dinucleotide matrix.
    
    Parameters
    ----------
    dinuc_matrix: np.ndarray
        A 3D numpy array of shape (n_seqs, 4, 4) containing the dinucleotide matrices.
    verbose: bool, optional
        If True, print progress information. Defaults to False.
    
    Returns
    -------
    dinuc_classes: list[list[int]]
        A list of lists, where each inner list contains indices of dinucleotides that belong to the same class.
    """
        
    N = len(dinuc_matrix)

    # Flatten each 4×4 matrix to a 16-long row (no copy)
    flat = dinuc_matrix.reshape(N, 16)

    # unique rows + inverse map
    # keys: unique flattened matrices, shape (C, 16)
    # inv: length-N array; inv[i] = class-id of row i
    keys, inv = np.unique(flat, axis=0, return_inverse=True)

    # Build dinuc_classes list
    C = len(keys)
    dinuc_classes = [[] for _ in range(C)]

    iterator = range(N)
    if verbose:
        iterator = tqdm(iterator, desc="Generating dinucleotide classes", unit="kmer")

    for i in iterator:
        dinuc_classes[inv[i]].append(i)
    if verbose:
        print(f"# dinucleotide classes: {len(dinuc_classes)}")
    return dinuc_classes


def find_dinuc_class(kmer_idx, dinuc_classes):
    """
    Find the dinucleotide class index for a given k-mer index.
    
    Parameters
    ----------
    kmer_idx: int
        Index of the k-mer for which to find the dinucleotide class.
    dinuc_classes: list[list[int]]
        A list of lists, where each inner list contains indices of dinucleotides that belong to the same class.
    
    Returns
    -------
    c_idx: int
        The index of the dinucleotide class that contains the k-mer.
    """

    for c_idx, c_list in enumerate(dinuc_classes):
        if kmer_idx in c_list:
            return c_idx


def calculate_node_pvals(preds, dinuc_classes, rng, n_samples=1000):
    """
    Calculate p-values for each node based on the predictions and dinucleotide classes.
    
    Parameters
    ----------
    preds: np.ndarray
        A 2D numpy array of shape (n_kmers, n_backgrounds) containing the predictions for each k-mer.
    dinuc_classes: list[list[int]]
        A list of lists, where each inner list contains indices of dinucleotides that belong to the same class.
    rng: np.random.Generator
        A random number generator for reproducibility.
    n_samples: int, optional
        Number of samples to use for calculating p-values. Defaults to 1000.
    
    Returns
    -------
    pvals: np.ndarray
        A 1D numpy array of shape (n_kmers,) containing the p-values for each k-mer.
    single_dinucs: list[bool]
        A list of booleans indicating whether each k-mer is a single dinucleotide (True) or not (False).
    """

    pvals = np.full(len(preds), np.nan, dtype=np.float32) # shape(4^k,)
    n_kmers, n_bgs = preds.shape
    single_dinucs = []
    
    for i in tqdm(range(n_kmers), unit="kmer"):
        kmer_i_mean = preds[i].mean()
        
        # Get the subset of preds with kmers in the same dinucleotide class
        c_idx = find_dinuc_class(i, dinuc_classes)
        subset_indices = dinuc_classes[c_idx]
        len_subset = len(subset_indices)
        if len_subset == 1:
            subset_indices = range(n_kmers) # If there is only one kmer in the class, use all kmers
            len_subset = len(subset_indices)
            single_dinucs.append(True)
        else:
            single_dinucs.append(False)
        subset_preds = preds[subset_indices].T # shape(n_bgs, len_subset)
        
        # Calculate the p-value
        counter = 0
        for _ in range(n_samples):
            random_preds = subset_preds[np.arange(n_bgs), rng.integers(0, len_subset, size=n_bgs)]
            if random_preds.mean() >= kmer_i_mean:
                counter += 1
        pvals[i] = counter / n_samples
    
    return pvals, single_dinucs


def build_candidate_nodes(g, seqs, preds, pvals):
    """
    Build candidate nodes in the graph from sequences, predictions, and p-values.
    
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph to which nodes will be added.
    seqs: list[str]
        List of k-mer sequences.
    preds: numpy.ndarray
        A 1D numpy array of shape (n_kmers,) containing the prediction for each k-mer.
    pvals: numpy.ndarray
        A 1D numpy array of shape (n_kmers,) containing the p-value for each k-mer.
    
    Returns
    -------
    indices: dict
        A dictionary mapping each k-mer sequence to its index.
    """

    indices = {}
    for i, (seq, pred, pval) in enumerate(zip(seqs, preds, pvals)):
        g.add_node(seq, pred=float(pred), pval=float(pval))
        indices[seq] = i
    return indices


def build_candidate_edges(g, maxk, node_pval_thresh=.01):
    """
    Build candidate edges in the graph based on the k-mer sequences and their p-values.
    
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph to which edges will be added.
    maxk: int
        Maximum k-mer length.
    node_pval_thresh: float, optional
        Threshold for node p-values to consider edges. Defaults to 0.01.
    """

    def _add_edge(s1: str, s2: str, edge_type: str):
        if g.has_edge(s1, s2):
            return # already added
        pred1, pval1 = g.nodes[s1]["pred"], g.nodes[s1]["pval"]
        pred2, pval2 = g.nodes[s2]["pred"], g.nodes[s2]["pval"]
        g.add_edge(s1, s2,
                    pval=-1.,
                    dPred=pred2 - pred1,
                    dPval=pval2 - pval1,
                    edge_type=edge_type)

    nucs = "ACGT"
    print('')
    for seq in tqdm(g.nodes, desc="Building candidate edges", unit="kmer"):
        L = len(seq)
        # Insertions
        if L < maxk:
            for i in range(L + 1):
                for nuc in nucs:
                    nbr = seq[:i] + nuc + seq[i:]
                    if g.nodes[nbr]["pval"] < node_pval_thresh:
                        _add_edge(seq, nbr, edge_type="ins")
        # Deletions
        if L > 1:
            for i in range(L):
                nbr = seq[:i] + seq[i + 1:]
                if g.nodes[nbr]["pval"] < node_pval_thresh:
                    _add_edge(seq, nbr, edge_type="del")


def prune_isolated_nodes(g):
    """
    Remove isolated nodes from the graph.
    
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph from which isolated nodes will be removed.
    """

    g.remove_nodes_from(list(nx.isolates(g)))


def prune_edges_by_pval(g, edge_pval_thresh=.01, drop_isolates=True):
    """
    Remove edges from the graph based on their p-values.
    
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph from which edges will be removed.
    edge_pval_thresh: float, optional
        Threshold for edge p-values to consider edges. Defaults to 0.01.
    drop_isolates: bool, optional
        If True, also remove isolated nodes after pruning edges. Defaults to True.
    """
    
    bad_edges = [(u, v)
                for u, v, attr in g.edges(data=True)
                if attr.get("pval", float("inf")) >= edge_pval_thresh]
    g.remove_edges_from(bad_edges)

    if drop_isolates:
        prune_isolated_nodes(g)


def calculate_edge_pvals(g, indices, rng, preds, n_bgs=100, n_samples=1000):
    """
    Calculate p-values for edges in the graph based on the predictions.
    
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph containing edges for which p-values will be calculated.
    indices: list[dict]
        List of dictionaries that maps each k-mer sequence to a unique index.
    rng: np.random.Generator
        Random number generator for reproducibility.
    preds: list[list[list[float]]] # shape(maxk, n_kmers of length k, n_bgs)
        List of predictions for each k-mer length.
    n_bgs: int, optional
        Number of backgrounds to use for calculating p-values. Defaults to 100.
    n_samples: int, optional
        Number of samples to use for calculating p-values. Defaults to 1000.
    """

    for s, t, data in tqdm(g.edges(data=True), desc="Calculating edge p-values", unit="edge"):
        s_1st_idx = len(s)-1
        s_2nd_idx = indices[s_1st_idx][s]
        edge_pred_s = preds[s_1st_idx][s_2nd_idx] # shape(n_bgs,)
        edge_pred_mean = data['dPred'] # shape(1,)
        
        alt_preds = []
        for _, s_nbr in g.out_edges(s, data=False):
            nbr_1st_idx = len(s_nbr)-1
            nbr_2nd_idx = indices[nbr_1st_idx][s_nbr]
            edge_pred_nbr = preds[nbr_1st_idx][nbr_2nd_idx] # shape(n_bgs,)
            edge_pred_s_nbr = edge_pred_nbr - edge_pred_s # shape(n_bgs,)
            alt_preds.append(edge_pred_s_nbr)
        alt_preds = np.array(alt_preds).T # shape(n_backgrounds, n_neighbors)
        n_neighbors = alt_preds.shape[1]

        counter = 0
        for _ in range(n_samples):
            random_preds = alt_preds[np.arange(n_bgs), rng.integers(0, n_neighbors, size=n_bgs)]
            if random_preds.mean() >= edge_pred_mean:
                counter += 1
        pval = counter / n_samples
        data["pval"] = pval


def annotate_edge_counts(g, edge_pval_thresh=.01):
    """
    Annotate the graph with counts of incoming and outgoing edges for each node.
    
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph to annotate.
    edge_pval_thresh: float, optional
        Threshold for edge p-values to consider edges. Defaults to 0.01.
    """

    # Initialize counters to zero for all nodes
    inc_total  = {n: 0 for n in g.nodes}
    inc_sig    = {n: 0 for n in g.nodes}
    out_total  = {n: 0 for n in g.nodes}
    out_sig    = {n: 0 for n in g.nodes}

    # Iterate over every directed edge u → v
    for u, v, data in g.edges(data=True):
        p = data.get("pval", float("inf"))
        sig = p < edge_pval_thresh

        # Outgoing for u
        out_total[u] += 1
        if sig:
            out_sig[u] += 1

        # Incoming for v
        inc_total[v] += 1
        if sig:
            inc_sig[v] += 1

    # Attach to node attributes
    for n in g.nodes:
        g.nodes[n]["num_incoming"]        = inc_total[n]
        g.nodes[n]["num_sig_incoming"]    = inc_sig[n]
        g.nodes[n]["num_outgoing"]        = out_total[n]
        g.nodes[n]["num_sig_outgoing"]    = out_sig[n]


def nodes_to_csv(g, fname, num_decimals=4):
    """
    Save the nodes of the graph to a CSV file.
    
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph whose nodes will be saved.
    fname: str, optional
        Name of the output CSV file.
    num_decimals: int, optional
        Number of decimal places to format the floating-point numbers. Defaults to 4.
    """
    
    fmt = f"{{:.{num_decimals}f}}".format
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["seq", "pred", "pval",
            "num_incoming", "num_sig_incoming",
            "num_outgoing", "num_sig_outgoing"]
        )
        for seq, data in g.nodes(data=True):
            w.writerow([
                seq,
                fmt(data["pred"]),
                fmt(data["pval"]),
                data["num_incoming"],
                data["num_sig_incoming"],
                data["num_outgoing"],
                data["num_sig_outgoing"],
            ])


def edges_to_csv(g, fname, num_decimals=4):
    """
    Save the edges of the graph to a CSV file.
    
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph whose edges will be saved.
    fname: str, optional
        Name of the output CSV file.
    num_decimals: int, optional
        Number of decimal places to format the floating-point numbers. Defaults to 4.
    """
    
    fmt = f"{{:.{num_decimals}f}}".format
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target",
                    "pval", "dPred", "dPval", "edge_type"])
        for u, v, data in g.edges(data=True):
            w.writerow([
                u, v,
                fmt(data["pval"]),
                fmt(data["dPred"]),
                fmt(data["dPval"]),
                data["edge_type"],
            ])


def save_graph(g, out_path, nodes_csv=None, edges_csv=None):
    """
    Save the graph to a file.
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph to save.
    out_path: str
        Path to the output file where the graph will be saved.
    nodes_csv: str, optional
        Path to the output CSV file for nodes.
    edges_csv: str, optional
        Path to the output CSV file for edges.
    """
    
    pickle.dump(g, open(out_path, 'wb'))
    
    if nodes_csv:
        nodes_to_csv(g, nodes_csv)
    if edges_csv:
        edges_to_csv(g, edges_csv)


def load_graph(in_path):
    """
    Load the graph from a file.
    
    Parameters
    ----------
    in_path: str
        Path to the input file from which the graph will be loaded.
    
    Returns
    -------
    g: nx.DiGraph
        The directed graph loaded from the file.
    """

    return pickle.load(open(in_path, 'rb'))
        

def build_graph(params, marginalization_dir="marginalization", outdir="graph", verbose=False):
    """,
    Build a k-mer graph based on the provided parameters.
    
    Parameters
    ----------
    params: dict
        Dictionary containing parameters for building the graph.
    marginalization_dir: str, optional
        Directory where the marginalization results are stored. Defaults to "marginalization".
    outdir: str, optional
        Output directory where the results will be saved. Defaults to "graph".
    verbose: bool, optional
        If True, print additional information during the graph building process. Defaults to False.
    
    Returns
    -------
    g: nx.DiGraph
        The directed graph containing k-mer nodes and edges.
    """
    
    print(f"\n### Calculate k-mer p-values")

    # Get parameters
    maxk = params['maxk']
    n_backgrounds = params.get('n_backgrounds', 100)
    random_seed = params.get('random_seed', 0)
    node_pval_nsamples = params.get('node_pval_nsamples', 1000)
    edge_pval_nsamples = params.get('edge_pval_nsamples', 1000)
    node_pval_thresh = params.get('node_pval_thresh', .01)
    edge_pval_tresh = params.get('edge_pval_thresh', .01)
    
    graph = nx.DiGraph() # empty directed graph
    kmer_indices = [] # list of dictionaries that maps each kmer sequence to a unique index
    all_preds = []
    rng = np.random.default_rng(seed=random_seed)
    for k in range(1, maxk + 1):
        # Load sequences
        _, seqs = read_fasta(f"{marginalization_dir}/kmers_k{k}.fa")

        # Generate dinucleotide matrices
        dinuc_matrix_npy = f"{outdir}/dinuc_matrix_k{k}.npy"
        if os.path.exists(dinuc_matrix_npy):
            dinuc_matrix = np.load(dinuc_matrix_npy)
        else:
            dinuc_matrix = generate_dinuc_matrices(seqs, verbose=verbose)
            np.save(dinuc_matrix_npy, dinuc_matrix)

        # Generate dinucleotide classes
        dinuc_classes_pkl = f"{outdir}/dinuc_classes_k{k}.pkl"
        if os.path.exists(dinuc_classes_pkl):
            with open(dinuc_classes_pkl, 'rb') as f:
                dinuc_classes = pickle.load(f)
        else:
            dinuc_classes = fast_generate_dinuc_classes(dinuc_matrix, verbose=verbose)
            with open(dinuc_classes_pkl, 'wb') as f:
                pickle.dump(dinuc_classes, f)

        # Load sequences and predictions
        preds = np.load(f"{marginalization_dir}/pred.after.k{k}.npy") # shape(n_kmers, n_backgrounds)
        all_preds.append(preds)

        # Calculate node p-values
        pvals_npy = f"{outdir}/node_pvals_k{k}.npy"
        if not os.path.exists(pvals_npy):
            pvals, _ = calculate_node_pvals(preds, dinuc_classes, rng, n_samples=node_pval_nsamples)
            np.save(pvals_npy, pvals)
            if verbose:
                print('')
        else:
            pvals = np.load(pvals_npy)

        # Build candidate nodes
        preds = preds.mean(axis=1) # shape(n_kmers,)
        k_indices_dict = build_candidate_nodes(graph, seqs, preds, pvals)
        kmer_indices.append(k_indices_dict)

    build_candidate_edges(graph, maxk, node_pval_thresh)
    prune_isolated_nodes(graph)
    calculate_edge_pvals(graph, kmer_indices, rng, all_preds, 
                         n_bgs=n_backgrounds, n_samples=edge_pval_nsamples)
    annotate_edge_counts(graph, edge_pval_thresh=edge_pval_tresh)

    return graph


def find_cores(g, drop_frac=.1, top_n=5):
    """
    Find core k-mers in the graph using a dynamic programming approach.
    
    Parameters
    ----------
    g: nx.DiGraph
        The directed graph containing k-mer nodes and edges.
    drop_frac: float, optional
        Fraction of the initial prediction value to drop when considering core k-mers. Defaults to .1.
    top_n: int, optional
        Number of top k-mers to consider as starting points for finding core k-mers.
    
    Returns
    -------
    all_cores: dict
        A dictionary mapping each k-mer sequence to its best core sequence and prediction value.
    top_n_cores: dict
        A dictionary mapping the top N k-mer sequences to their best core sequence and prediction value.
    """

    print(f"\n### Find core k-mers")

    # Build directional DAG using deletion edges
    dag = nx.DiGraph((u, v) for u, v, d in g.edges(data=True) if d["edge_type"] == "del")
    dag.add_nodes_from(g.nodes)

    # Memoise on (node, init_pred) → (core_seq, core_pred)
    @lru_cache(maxsize=None)
    def _best_core(node, init_pred):
        """
        Find the best core sequence starting from a given node.
        
        Parameters
        ----------
        node: str
            The starting node sequence.
        init_pred: float
            The prediction value for the starting node.
        """
        
        best_seq  = node
        best_pred = g.nodes[node]["pred"]

        for child in dag.successors(node):
            core_seq, core_pred = _best_core(child, init_pred)
            if init_pred - core_pred <= init_pred * drop_frac:
                if len(core_seq) < len(best_seq) or (
                    len(core_seq) == len(best_seq) and core_pred > best_pred
                ):
                    best_seq, best_pred = core_seq, core_pred
        return best_seq, best_pred

    # Evaluate once for every node
    for n in nx.topological_sort(dag):
        _best_core(n, round(g.nodes[n]["pred"], 6))

    # Pretty-print paths for the top-N starters
    starters = sorted(g.nodes, key=lambda s: g.nodes[s]["pred"], reverse=True)[:top_n]

    for start in starters:
        init_pred = g.nodes[start]["pred"]
        max_drop  = init_pred * drop_frac

        path = [(start, init_pred)]
        current = start
        while True:
            next_node = None
            for child in dag.successors(current):
                core_seq_child, _ = _best_core(child, round(init_pred, 6))
                core_seq_curr, _  = _best_core(current, round(init_pred, 6))
                drop = init_pred - g.nodes[child]["pred"]
                if drop <= max_drop and core_seq_child == core_seq_curr:
                    next_node = child
                    break
            if next_node is None:
                break
            path.append((next_node, g.nodes[next_node]["pred"]))
            current = next_node

        print(f"Starting node: {start}")
        for seq, pr in path:
            print(f"  {seq:<12} pred={pr:.4f}  ({pr - init_pred:.4f})")

    all_cores = {n: _best_core(n, round(g.nodes[n]["pred"], 6)) for n in g.nodes}
    top_n_cores = {n: (core_seq, core_pred) for n, (core_seq, core_pred) in all_cores.items() 
                   if n in starters}
    return all_cores, top_n_cores