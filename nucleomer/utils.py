import torch
import itertools
import pyfaidx
import numpy as np
import pandas as pd
from tangermeme.utils import one_hot_encode


def check_accelerator():
    """
    Check for available hardware accelerators (GPU or MPS) and return the appropriate device.
    If no accelerators are available, it defaults to CPU.
    
    Returns
    -------
        torch.device: The device to use for computations (CPU, CUDA, or MPS).
    """
    
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA device available.\n")
    elif (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        device = torch.device('mps')
        print("MPS device available.\n")
    else: 
        device = torch.device('cpu')
        print("No accelerator device available. Using CPU.\n")

    if device.index is not None:
        device = f"{device.type}:{device.index}"
    else:
        device = device.type
    return device


def generate_kmers(l, alphabet=['A', 'C', 'G', 'T']):
    """Generate all possible k-mers of length l from the given alphabet.
    
    Parameters
    ----------
    l: int 
        Length of the k-mers to generate.
    
    alphabet: list, optional 
        List of characters to use for generating k-mers. Defaults to DNA bases.
    
    Returns
    -------
        list: A list of all possible k-mers of length l.
    """
    
    return [''.join(motif) for motif in itertools.product(alphabet, repeat=l)]


def extract_loci(genome_fasta, loci_bed, n_loci, seq_len=2114, 
                validate=True, dtype=torch.float32):
    """
    Extract genomic sequences from a FASTA file based on loci specified in a BED file.
    
    Parameters
    ----------
    genome_fasta: str
        Path to the genome FASTA file.
    
    loci_bed: str
        Path to the BED file containing genomic loci.
    
    n_loci: int
        Number of random loci to extract.
    
    seq_len: int, optional
        Length of the sequences to extract. Default is 2114.
    
    validate: bool, optional
        If True, validate that the sequences contain only A, C, G, T characters. Default is True.

    dtype: torch.dtype, optional
        Data type for the output tensor. Default is torch.float32.
    
    Returns
    -------
    torch.Tensor: A tensor of shape (n_loci, 4, seq_len)
        containing the one-hot encoded sequences.
    """
    
    genome = pyfaidx.Fasta(genome_fasta)
    loci_df = pd.read_csv(loci_bed, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    
    counter, indices = 0, []
    x = torch.full((n_loci, 4, seq_len), np.nan, dtype=dtype)
    while counter < n_loci:
        idx = np.random.randint(0, loci_df.shape[0])
        while idx in indices:
            idx = np.random.randint(0, loci_df.shape[0])
        indices.append(idx)

        chrom, start, end = loci_df.iloc[idx]
        seq = genome[chrom][start:end].seq.upper()
        if validate:
            # Make sure that the only characters are A, C, G, T
            if set(seq) - set("ACGT"):
                continue
        x[counter] = one_hot_encode(seq).to(dtype)
        counter += 1

    assert torch.isnan(x).sum() == 0
    return x


def read_fasta(fasta_path):
    """
    Read a FASTA file and return the sequence names and sequences.
    
    Parameters
    ----------
    fasta_path: str
        Path to the FASTA file.
    
    Returns
    -------
    tuple: A tuple containing two lists:
        - names: List of sequence names (without the '>' character).
        - seqs: List of sequences corresponding to the names.
    """

    names, seqs = [], []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                names.append(line[1:])
                seqs.append('')
            else:
                seqs[-1] += line
    return names, seqs


def ohe(seqs, device="cpu", dtype=torch.float32):
    """
    One-hot encode a list of DNA sequences.

    Parameters
    ----------
    seqs: list of str
        List of DNA sequences to be one-hot encoded. All sequences must have the same length.

    device: str, optional
        Device to place the resulting tensor on (e.g., "cpu", "cuda", "mps"). Defaults to "cpu".

    dtype: torch.dtype, optional
        Data type for the resulting tensor. Defaults to torch.float32.

    Returns
    -------
    torch.Tensor: A tensor of shape (N, 4, k) where N is the number of sequences,
        4 corresponds to the nucleotides A, C, G, T, and k is the length of each sequence.
    """

    k = len(seqs[0])
    assert all(len(s) == k for s in seqs), "All sequences in the batch must have same length."
    
    # Build index tensor [N, k] in CPU then push once
    idx_cpu = torch.empty((len(seqs), k), dtype=torch.int64)
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, s in enumerate(seqs):
        idx_cpu[i] = torch.tensor([mapping[ch] for ch in s], dtype=torch.int64)
    idx = idx_cpu.to(device)

    out = torch.zeros((len(seqs), 4, k), device=device, dtype=dtype)
    
    return out.scatter_(1, idx.unsqueeze(1), 1)