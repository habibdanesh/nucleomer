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
    model_type = params['model_type']
    
    # Generate k-mers
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
                                n_backgrounds, seq_len=in_length, validate=True, dtype=dtype)
        np.save(x_before_npy, x_before)
    else:
        x_before = torch.from_numpy(np.load(x_before_npy))

    x_before = x_before.to(device)
    
    # Control # TODO: make sure this works for other models than BPNet
    ctrl_tensor = None
    if n_ctrl_tracks > 0:
        ctrl_tensor = torch.zeros(n_backgrounds, n_ctrl_tracks, in_length, dtype=dtype).to(device)
    
    # Get before predictions (backgrounds)
    pred_before_npy = f"{outdir}/pred.before.npy"
    if not os.path.exists(pred_before_npy):
        if model_type == "bpnet-lite":
            pred_before = predict(model, x_before, args=(ctrl_tensor,), batch_size=batch_size, 
                                device=device, verbose=False).squeeze()
        elif model_type == "ProCapNet":
            pred_before = predict(model, x_before, batch_size=batch_size, 
                                device=device, verbose=False).squeeze()
        np.save(pred_before_npy, pred_before)
    
    # Get after predictions (kmers inserted into backgrounds)
    print(f"\n### Marginalize k-mers")
    for k in range(1, maxk + 1):
        pred_after_npy = f"{outdir}/pred.after.k{k}.npy" # shape(4^k, n_backgrounds)
        if os.path.exists(pred_after_npy):
            continue

        fasta_path = f"{outdir}/kmers_k{k}.fa"
        kmer_names, kmer_seqs = read_fasta(fasta_path)
        n_kmers = len(kmer_names)

        pred_after = torch.full((n_kmers, n_backgrounds), float('nan'), dtype=dtype, device=device)

        if k <= 5:
            if n_ctrl_tracks > 0:
                ctrl_tensor = torch.zeros(n_backgrounds, n_ctrl_tracks, in_length, dtype=dtype).to(device)
            for i, kmer_seq in tqdm(enumerate(kmer_seqs), total=n_kmers, desc=f"{k}-mers", unit=" kmer"):
                kmer_ohe = one_hot_encode(kmer_seq).unsqueeze(0).to(device)
                x_after = substitute(x_before, kmer_ohe)
                if n_ctrl_tracks > 0:
                    pred_after[i] = predict(model, x_after, args=(ctrl_tensor,), batch_size=batch_size, 
                                        device=device, verbose=False).squeeze()
                else:
                    pred_after[i] = predict(model, x_after, batch_size=batch_size, 
                                        device=device, verbose=False).squeeze()
        else:
            # Batched version
            if n_ctrl_tracks > 0:
                ctrl_tensor = torch.zeros(batch_size * n_backgrounds, n_ctrl_tracks, in_length, dtype=dtype).to(device)
            n_batches = n_kmers // batch_size
            for i in tqdm(range(0, n_kmers, batch_size), 
                            total=n_batches, desc=f"{k}-mers", unit=f"batch({batch_size} kmers)"):
                kmer_seqs_batch = kmer_seqs[i:min(i+batch_size, n_kmers)]
                x_after_batch = torch.stack([substitute(x_before, one_hot_encode(kmer_seq).unsqueeze(0).to(device))
                                                for kmer_seq in kmer_seqs_batch]).to(device) # (batch_size, n_backgrounds, 4, in_length)
                x_after_batch = x_after_batch.view(-1, 4, in_length) # (batch_size * n_backgrounds, 4, in_length)
                if n_ctrl_tracks > 0:
                    pred_after_batch = predict(model, x_after_batch, args=(ctrl_tensor,), batch_size=batch_size, 
                                            device=device, verbose=False).squeeze() # (batch_size * n_backgrounds)
                else:
                    pred_after_batch = predict(model, x_after_batch, batch_size=batch_size, 
                                            device=device, verbose=False).squeeze()
                
                pred_after_batch = pred_after_batch.view(batch_size, n_backgrounds) # (batch_size, n_backgrounds)
                pred_after[i:min(i+batch_size, n_kmers)] = pred_after_batch

        np.save(pred_after_npy, pred_after.cpu())