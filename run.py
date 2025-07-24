import sys
import torch
import json
import os

# Command-line arguments
params_json = sys.argv[1]

# Load parameters
with open(params_json, 'r') as f:
    params = json.load(f)

#
from nucleomer.models_io import load_bpnet
from nucleomer.utils import check_accelerator
from nucleomer.marginalize import marginalize_kmers
from nucleomer.kmer_graph import build_graph, save_graph, load_graph, find_cores

outdir = f"nucleomer-results-{params['exp_id']}"
os.makedirs(outdir, exist_ok=True)

csv_dir = f"{outdir}/csv"
os.makedirs(csv_dir, exist_ok=True)

pkl_dir = f"{outdir}/pkl"
os.makedirs(pkl_dir, exist_ok=True)

json_dir = f"{outdir}/json"
os.makedirs(json_dir, exist_ok=True)

device = check_accelerator()

graph_pkl = f"{pkl_dir}/kmer_graph.maxk{params['maxk']}.pkl"
if os.path.exists(graph_pkl):
    graph = load_graph(graph_pkl)
else:
    # Run marginalization ##################################################
    model = load_bpnet(params["model_path"], device=device)
    marginalize_kmers(model=model, params=params, outdir=outdir, device=device)

    # Build graph ##################################################
    graph = build_graph(params=params, outdir=outdir)

    nodes_csv = f"{csv_dir}/graph_nodes.maxk{params['maxk']}.csv"
    edges_csv = f"{csv_dir}/graph_edges.maxk{params['maxk']}.csv"
    save_graph(graph, graph_pkl, nodes_csv, edges_csv)
print(f"\nGraph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

# Find core k-mers ##################################################
n_cores = params.get("n_core_kmers", 5)
all_cores, top_n_cores = find_cores(graph, top_n=n_cores, drop_frac=.4)

cores_json = f"{json_dir}/all_cores.maxk{params['maxk']}.json"
with open(cores_json, 'w') as f:
    json.dump(all_cores, f, indent=4)

top_n_cores_json = f"{json_dir}/top_{n_cores}_cores.maxk{params['maxk']}.json"
with open(top_n_cores_json, 'w') as f:
    json.dump(top_n_cores, f, indent=4)

# Run TOMTOM ##################################################
for seq in top_n_cores.keys():
    print(f"\nRunning TOMTOM for {seq}")
    cmd = f"ttl -q {seq} -t {params['motifs_meme']} | head -n 5"
    os.system(cmd)