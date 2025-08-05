import sys
import torch
from torch import inference_mode
from torch.amp import autocast
import json
import os

# Warnings
import warnings
warnings.filterwarnings(
    "ignore",
    message="In MPS autocast, but the target dtype is not supported",
    module="torch.amp.autocast_mode",
)

os.environ['KMP_WARNINGS'] = 'off'

# Command-line arguments
params_json = sys.argv[1]

# Load parameters
with open(params_json, 'r') as f:
    params = json.load(f)

#
from nucleomer.models_io import load_bpnet, load_procapnet
from nucleomer.utils import check_accelerator
from nucleomer.marginalize import marginalize_kmers
from nucleomer.kmer_graph import build_graph, save_graph, load_graph, find_cores

outdir = f"nucleomer-results-{params['run_id']}"
os.makedirs(outdir, exist_ok=True)

verbose = params.get("verbose", False)

marginalization_dir = f"{outdir}/marginalization"
os.makedirs(marginalization_dir, exist_ok=True)

graph_dir = f"{outdir}/graph"
os.makedirs(graph_dir, exist_ok=True)

cores_dir = f"{outdir}/cores"
os.makedirs(cores_dir, exist_ok=True)

tomtom_dir = f"{outdir}/tomtom"
os.makedirs(tomtom_dir, exist_ok=True)

device = check_accelerator()

dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
dtype = params['data_type']
dtype = dtypes.get(dtype, torch.float32)

graph_pkl = f"{graph_dir}/kmer_graph.maxk{params['maxk']}.pkl"
if os.path.exists(graph_pkl):
    graph = load_graph(graph_pkl)
else:
    # Run marginalization ##################################################
    with inference_mode():
        with autocast(device, dtype=dtype):
            if params["model_type"] == "bpnet-lite":
                model = load_bpnet(params["model_path"], device=device)
            elif params["model_type"] == "ProCapNet":
                model = load_procapnet(params["model_path"], device=device)
            else:
                raise ValueError(f"Unknown model type: {params['model_type']}")
            
            marginalize_kmers(model=model, params=params, outdir=marginalization_dir, 
                                device=device, dtype=dtype)

    # Build graph ##################################################
    graph = build_graph(params=params, marginalization_dir=marginalization_dir, outdir=graph_dir, verbose=verbose)

    nodes_csv = f"{graph_dir}/graph_nodes.maxk{params['maxk']}.csv"
    edges_csv = f"{graph_dir}/graph_edges.maxk{params['maxk']}.csv"
    save_graph(graph, graph_pkl, nodes_csv, edges_csv)
print(f"\nGraph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

# Find core k-mers ##################################################
n_cores = params.get("n_core_kmers", 5)
all_cores, top_n_cores = find_cores(graph, top_n=n_cores, drop_frac=.2)

cores_json = f"{cores_dir}/all_cores.maxk{params['maxk']}.json"
with open(cores_json, 'w') as f:
    json.dump(all_cores, f, indent=4)

top_n_cores_json = f"{cores_dir}/top_{n_cores}_cores.maxk{params['maxk']}.json"
with open(top_n_cores_json, 'w') as f:
    json.dump(top_n_cores, f, indent=4)

# Run TOMTOM ##################################################
print(f"\n### TOMTOM")
for seq in top_n_cores.keys():
    out_file = f"{tomtom_dir}/tomtom.{seq}.txt"
    if not os.path.exists(out_file):
        print(f"Running TOMTOM on {seq}")
        cmd = f"ttl -q {seq} -t {params['motifs_meme']} > {out_file}"
        os.system(cmd)