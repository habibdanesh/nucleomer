import sys
import os
import json
import numpy as np
import torch
from tqdm import tqdm

from tangermeme.predict import predict
from tangermeme.ersatz import substitute
from tangermeme.utils import one_hot_encode

from .utils import generate_kmers, extract_loci, read_fasta


def marginalize_kmers(model, params, 
                outdir="marginalization", device="cpu", dtype=torch.float32):
    """
    Marginalize k-mers by substituting them into genomic backgrounds and predicting their effects.
    
    Parameters
    ----------
    model: torch.nn.Module
        A trained model to use for predictions.
    params: dict
        Dictionary containing parameters for the marginalization process.
    outdir: str, optional
        Directory to save the results. Defaults to "marginalization".
    device: str, optional
        Device to run the model on (e.g., "cpu", "cuda", "mps"). Defaults to "cpu".
    dtype: torch.dtype, optional
        Data type for the model predictions. Defaults to torch.float32.
    """

    # Get parameters
    maxk = params['maxk']
    n_backgrounds = params['n_backgrounds']
    in_length = params['in_length']
    n_ctrl_tracks = params['n_ctrl_tracks']
    batch_size = params['batch_size']
    
    # Generate k-mers
    print(f"### Generate k-mer sequences")
    for k in range(1, maxk + 1):
        outfile = f"{outdir}/kmers_k{k}.fa"
        if not os.path.exists(outfile):
            kmers = generate_kmers(k)
            print(f"Generated {len(kmers)} {k}-mers")
            with open(outfile, 'w') as f:
                for i, kmer in enumerate(kmers):
                    f.write(f'>k{k}_kmer{i}\n{kmer}\n')

    # Load backgrounds
    x_before_npy = f"{outdir}/x_before.npy"
    if not os.path.exists(x_before_npy):
        x_before = extract_loci(params["genome_fasta"], params["backgrounds_bed"], 
                                n_backgrounds, seq_len=in_length, validate=True).to(device)
        np.save(x_before_npy, x_before.cpu())
    else:
        x_before = torch.from_numpy(np.load(x_before_npy)).to(device)
    
    # Control # TODO: make sure this works for other models than BPNet
    ctrl_tensor = None
    if n_ctrl_tracks > 0:
        ctrl_tensor = torch.zeros(n_backgrounds, n_ctrl_tracks, in_length).to(device)
    
    # Get before predictions (backgrounds)
    pred_before_npy = f"{outdir}/pred.before.npy"
    if not os.path.exists(pred_before_npy):
        pred_before = predict(model, x_before, args=(ctrl_tensor,), batch_size=batch_size, 
                            device=device, verbose=False).squeeze()
        np.save(pred_before_npy, pred_before)
    
    # Get after predictions (kmers inserted into backgrounds)
    print(f"\n### Marginalize k-mers")
    for k in range(1, maxk + 1):
        pred_after_npy = f"{outdir}/pred.after.k{k}.npy" # shape(4^k, n_backgrounds)
        if not os.path.exists(pred_after_npy):
            fasta_path = f"{outdir}/kmers_k{k}.fa"
            kmer_names, kmer_seqs = read_fasta(fasta_path)
            n_kmers = len(kmer_names)

            pred_after = torch.full((n_kmers, n_backgrounds), float('nan'), dtype=dtype, device=device)
            
            for i, kmer_seq in tqdm(enumerate(kmer_seqs), total=n_kmers,
                                    desc=f"{k}-mers", unit=" kmer"):
                kmer_ohe = one_hot_encode(kmer_seq).unsqueeze(0).to(device)
                x_after = substitute(x_before, kmer_ohe)
                pred_after[i] = predict(model, x_after, args=(ctrl_tensor,), batch_size=batch_size, 
                                        device=device, verbose=False).squeeze()

            np.save(pred_after_npy, pred_after.cpu())