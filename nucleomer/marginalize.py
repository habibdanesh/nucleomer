import sys
import os
import json
import numpy as np
import torch
from tqdm import tqdm

from .utils import generate_kmers, extract_loci, read_fasta, ohe


def _build_substituted_batch(x_before, kmers_ohe, insert_pos):
    """
    Create [K*B, 4, L] by repeating backgrounds K times and writing the k-mers
    into the same slice [insert_pos:insert_pos+k] for every background.
    
    Parameters
    ----------
    x_before: torch.Tensor
        Tensor of shape [B, 4, L] containing background sequences.

    kmers_ohe: torch.Tensor
        Tensor of shape [K, 4, k] containing one-hot encoded k-mers

    insert_pos: int
        Position to insert the k-mers into the backgrounds.

    Returns
    -------
    torch.Tensor
        Tensor of shape [K*B, 4, L] containing the backgrounds with k-mers inserted.
    """
    B, C, L = x_before.shape
    K, _, k = kmers_ohe.shape
    assert C == 4 and kmers_ohe.shape[1] == 4
    assert 0 <= insert_pos <= L - k

    # Repeat backgrounds: [K, B, 4, L] → [K*B, 4, L]
    x = x_before.unsqueeze(0).expand(K, B, C, L).contiguous().view(K * B, C, L)

    # Tile k-mers for all backgrounds: [K,4,k] → [K*B,4,k]
    patch = kmers_ohe.repeat_interleave(B, dim=0)

    # In-place write of the patch slice for all K*B
    x[:, :, insert_pos:insert_pos + k] = patch
    return x


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
    
    # Get before predictions (backgrounds)
    pred_before_npy = f"{outdir}/pred.before.npy"
    if not os.path.exists(pred_before_npy):
        print(f"Get background predictions..")
        if n_ctrl_tracks > 0:
            ctrl_tensor = torch.zeros(n_backgrounds, n_ctrl_tracks, in_length, dtype=dtype).to(device)
            pred_before = model(x_before, ctrl_tensor).squeeze()
        else:
            pred_before = model(x_before).squeeze()
        np.save(pred_before_npy, pred_before.cpu())
    
    # Get after predictions (kmers inserted into backgrounds)
    print(f"\n### Marginalize k-mers")
    for k in range(1, maxk + 1):
        pred_after_npy = f"{outdir}/pred.after.k{k}.npy" # shape(4^k, n_backgrounds)
        if os.path.exists(pred_after_npy):
            continue

        fasta_path = f"{outdir}/kmers_k{k}.fa"
        kmer_names, kmer_seqs = read_fasta(fasta_path)
        n_kmers = len(kmer_names)

        ins_pos_k = (in_length - k) // 2

        pred_after = torch.full((n_kmers, n_backgrounds), float('nan'), dtype=dtype, device=device)

        if k <= 4:
            kmers_ohe = ohe(kmer_seqs, device=device, dtype=dtype)
            x_after_batch = _build_substituted_batch(x_before, kmers_ohe, ins_pos_k).to(device)  # (n_kmers * n_backgrounds, 4, in_length)

            if n_ctrl_tracks > 0:
                ctrl_tensor = torch.zeros(n_backgrounds, n_ctrl_tracks, in_length, dtype=dtype).to(device)
            for i in tqdm(range(n_kmers), total=n_kmers, desc=f"{k}-mers", unit=" kmer"):

                start_idx = i * n_backgrounds
                end_idx = start_idx + n_backgrounds
                x_after = x_after_batch[start_idx:end_idx]  # (n_backgrounds, 4, in_length)
                
                if n_ctrl_tracks > 0:
                    pred_after[i] = model(x_after, ctrl_tensor).squeeze()
                else:
                    pred_after[i] = model(x_after).squeeze()
        else:
            # Batched version
            if n_ctrl_tracks > 0:
                ctrl_tensor = torch.zeros(batch_size * n_backgrounds, n_ctrl_tracks, in_length, dtype=dtype).to(device)
            n_batches = n_kmers // batch_size
            for i in tqdm(range(0, n_kmers, batch_size), 
                            total=n_batches, desc=f"{k}-mers", unit=f" batch({batch_size} kmers)"):
                
                kmers_ohe = ohe(kmer_seqs[i:min(i+batch_size, n_kmers)], device=device, dtype=dtype)
                x_after_batch = _build_substituted_batch(x_before, kmers_ohe, ins_pos_k).to(device)  # (n_kmers * n_backgrounds, 4, in_length)
                
                if n_ctrl_tracks > 0:
                    pred_after_batch = model(x_after_batch, ctrl_tensor).squeeze()
                else:
                    pred_after_batch = model(x_after_batch).squeeze()
                
                pred_after_batch = pred_after_batch.view(batch_size, n_backgrounds) # (batch_size, n_backgrounds)
                pred_after[i:min(i+batch_size, n_kmers)] = pred_after_batch

        np.save(pred_after_npy, pred_after.cpu())