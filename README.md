# MembraneABMSim

A Julia package implementing an agent-based model for the Gram-negative outer membrane. 

_This codebase is still in active development so implementation may change. Feel free to leave issues or get in contact with bugs, queries, or feature requests_

> This branch is the version of the codebase used for Williams _et al._ 2026. Config files used to generate the simulation data can be found in `config/PAPER_1`. Note that simulations generated from these configs may differ slightly from those presented in the paper, since these were continued from checkpoints a few times, and RNG checkpointing has not been implemented in the codebase yet.

## Quick start:

MembraneABMSim is not yet a registered Julia package, but setup is still straightforward. From the repo root, launch Julia, then run the following commands in the Julia REPL:

```julia
]activate .
]instantiate
```

Instantiation may take a few minutes on first run. You can ignore errors associated with the Metal and CUDA packages - these may not be available for your device. Once the `instantiate` command has finished, you can now run:

```julia
include("src/MembraneABMSim.jl")
```

Again, this may take a few minutes on first run. MembraneABMSim is now ready to use.

Simulations in MembraneABMSim are specified by a JSON config file (which you probably want to keep in the `config/simulation` directory) and run with the `run_sim` function. To run the demo config provided with the repo, run:

```julia
MembraneABMSim.run_sim("config/simulation/demo.json")
```

Output from the simulation is written to the `out` directory. You can visualise the simulation with the command:

```julia
MembraneABMSim.make_membrane_movie("demo")
```

This creates a movie of the simulation in `out/demo`.

To run a simultion _sweep_, you will need a sweep config (different to a simulation config), which is passed to the `run_sweep` function. To run the demo config provided with the repo, run:

```julia
MembraneABMSim.run_sweep("config/sweep/demo.json")
```

Output is written to structured sub-directories of `out`. Sweeps can be analysed using the `analyse_sweep` function with a metric function (many such functions are in `src/postprocessing/metrics.jl`, or you can write your own). Processed data is saved to each of the structured output sub-directories to be used in plots, for example.

## Setting up your own simulations:

A tutorials page is forthcoming. For now, it is probably easiest to try copying the demo configs and tinkering with the parameters, and referring to the documentation.