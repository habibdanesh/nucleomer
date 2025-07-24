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


def extract_loci(genome_fasta, loci_bed, n_loci, seq_len=2114, validate=True):
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
    
    Returns
    -------
    torch.Tensor: A tensor of shape (n_loci, 4, seq_len)
        containing the one-hot encoded sequences.
    """
    
    genome = pyfaidx.Fasta(genome_fasta)
    loci_df = pd.read_csv(loci_bed, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    
    counter, indices = 0, []
    x = torch.full((n_loci, 4, seq_len), np.nan, dtype=torch.float32)
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
        x[counter] = one_hot_encode(seq).float()
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