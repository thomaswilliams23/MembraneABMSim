# MembraneABMSim

A Julia package implementing an agent-based model for the Gram-negative outer membrane. 

_This codebase is still in active development so implementation may change. Feel free to leave issues or get in contact with bugs, queries, or feature requests_

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

## Running tests:

Unit tests live in `test/` (`runtests.jl` includes one file per source area, e.g. `test_utils.jl`, `test_forces.jl`). Run them from the repo root with:

```julia
]activate .
]test
```

or `julia --project -e 'using Pkg; Pkg.test()'`. To run only some test files, name them (any of `"utils"`, `"forces"`, `"initialise"`):

```sh
julia --project -e 'using Pkg; Pkg.test(test_args=["forces"])'
```

Each test file can also be run on its own from a REPL (quick to repeat once the package has loaded):

```julia
include("test/test_forces.jl")
```

Please run the tests before merging into `dev`, and add tests alongside new or changed functions. Tests marked `@test_broken` document known issues; switch them to `@test` once fixed.

## Setting up your own simulations:

A tutorials page is forthcoming. For now, it is probably easiest to try copying the demo configs and tinkering with the parameters, and referring to the documentation.