# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MembraneABMSim is an unregistered Julia package (Julia 1.12): a 2D agent-based model of the Gram-negative outer membrane. Agents are OMPs (OmpA, OmpCF, BamA, LptD) and LPS discs on a periodic domain. They interact through pairwise attraction-repulsion forces, and the membrane grows as BamA inserts new OMPs and LptD inserts new LPS.

## Commands

The package is loaded with `include`, not `using` (see README):

```julia
]activate .
]instantiate
include("src/MembraneABMSim.jl")
MembraneABMSim.run_sim("config/simulation/demo.json")   # output -> out/<system.output_dir>/raw_data/*.jld2
MembraneABMSim.make_membrane_movie("demo")
MembraneABMSim.run_sweep("config/sweep/demo.json")
```

Tests:

```sh
julia --project -e 'using Pkg; Pkg.test()'                              # fast + gpu tiers (default, ~1 min)
julia --project -e 'using Pkg; Pkg.test(test_args=["slow"])'            # one tier: fast, slow or gpu ("all" for every tier)
julia --project -e 'using Pkg; Pkg.test(test_args=["biology", "step"])' # named files, from any tier
julia --project -e 'include("test/fast/test_step.jl")'                   # one file directly (also works from a REPL)
```

- New test files go in `test/<tier>/test_<name>.jl` and are picked up automatically. Start each with `include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))`.
- `Metal` and `CUDA` are hard dependencies loaded at module load. Errors from them on machines without that GPU are expected and can be ignored.

## Architecture

**Simulation loop** (`run_sim` and `step!`, src/main.jl; `initialise_sim` and the per-backend step operations, src/sim_state.jl; `SimState`, src/types/SystemTypes.jl; backend types, src/types/BackendTypes.jl):
1. Parse the JSON config into `AllParams` (src/types/ParamTypes.jl, via JSON3/StructTypes).
2. `initialise_sim` precomputes the nascent-insertion lookup tables (`build_nascent_added_area_lookup`, `compute_nascent_incs`), initialises the system (random / specify_positions / from_checkpoint) and returns a `SimState`. The state holds the agents, the grid, the backend (`SimBackendCPU` / `SimBackendMetal` / `SimBackendCUDA`) and the `rng`.
3. `run_sim` calls `step!(state, params, t)` once per step and writes output every `vis_dt`. Device-specific work inside a step is dispatched on the backend type (`copy_to_host!`, `apply_growth_and_nonforce!`, `advance_positions!`, ...), not branched on a device string.
4. Each step (`step!`):
   - Update the BAM and Lpt insertion subsystems (src/biology.jl).
   - Rebuild the grid if the agent count changed (`sync_grid!`, src/update_grid.jl).
   - Update tethering and assembly, then grow the nascent agents (src/nascent.jl).
   - Rescale the domain by the area those insertions added (src/update_grid.jl).
   - Apply diffusion and forces.
   - Write output every `vis_dt`.

**Two representations of the same system** must be kept in sync:
- `AllAgents` (src/types/AgentTypes.jl): mutable structs per agent type. The biology, I/O and checkpointing code use these.
- `AllAgentsFlat` (src/types/SystemTypes.jl): flat arrays (positions interleaved x,y) in *sorted* order. The force code uses these.
- `agent.index` is the agent index. `flat.agent_ix_to_sorted_ix` / `sorted_ix_to_agent_ix` map between the two orders.
- `compile_flat_system_data_cpu!` (src/forces.jl) rebuilds the flat data from the structs. Code that edits one side directly must update the other.

**Spatial grid** (src/update_grid.jl):
- Cell width is about `force.sensing_radius`. Cells are numbered in dense Morton (z) order (`build_dense_morton`, src/utils.jl), and agents are sorted by cell so neighbours sit close in memory.
- The grid is rebuilt only when agents are added, or every `max_steps_between_grid_sync(params)` steps (optional config key `system.max_steps_between_grid_sync`, default 100). Between rebuilds, the neighbour search widens its window by `agg_dist_since_grid_sync`, the distance agents have moved since the last rebuild.
- Square domains only: non-square Morton indexing is a known `@test_broken`.

**Compact identifiers** (`make_identifier` / `parse_identifier`, src/utils.jl) pack each agent's type, tethered flag, inserting/nascent state and nascent index into one Int for the flat arrays:
- The sign means tethered.
- Odd means OMP and even means LPS. The force code picks `mu_attr` using `id & 1`.
- Inserting/nascent pairs are linked through `nascent_to_inserting_ixs` / `nascent_to_substrate_ixs`, which are excluded from each other's attraction-repulsion forces and instead feel `compute_inserting_substrate_force`.

**Three backends**, chosen by `system.device`:
- CPU: src/forces.jl, src/initialise.jl.
- Metal: src/metal/.
- CUDA: src/cuda/.
- The GPU kernels (KernelAbstractions, Float32) mirror the CPU force and position-update logic almost line for line. A fix to force or neighbour-search logic usually has to be applied in all three. `copy_data_to_cpu_yn` decides when GPU data must be copied back for host-side biology or output.

**Timing:** interval events (output, insertion attempts) use `is_interval_step(t, interval, dt)`, which counts whole steps. Don't compare floating-point times directly.

**Postprocessing** (src/postprocessing/): metrics, sweep analysis (`analyse_sweep`) and movie making, which read the JLD2 output in `out/`.

## Tests

Tests check contracts that should survive refactors and changes to the force laws: invariants, conservation, symmetries, and agreement with slow brute-force references. They never pin force values, identifier integers, grid or sort order, or RNG-dependent outcomes such as "N agents after k steps with seed X".

**Tiers** (test/<tier>/):
- `fast/`: pure functions, components against oracles, biology made deterministic by forcing rates to 0 or effectively infinity, `step!`, and short `run_sim`/sweep runs.
- `slow/`: statistical rate checks, physical sanity checks (diffusion, relaxation, packing), a long run with an exact area balance, and plot smoke tests.
- `gpu/`: CPU-vs-GPU agreement. Each backend runs only if its device is present (`Metal.functional()` / `CUDA.functional()`) and otherwise reports a skip. The GPU backends only support `init.method = "random"`.

**Helpers** (test/helpers/, loaded by `setup.jl`):
- builders.jl: tests describe scenarios and never call agent, grid or flat constructors.
  - `params_with(; section=(field=value,)...)` builds params from the demo config with overrides, writing output to a `mktempdir()`. `quiet_params` (no biology) and `busy_params` (everything happening fast) are the usual starting points. `single_insertion_params(type)` sets up one prompt insertion.
  - `build_state(params; OmpA=[pos...], ..., L)` (CPU only) and `random_state(params; OmpA=n, ..., device)` return `(state, params)` through `initialise_sim`.
  - `run_steps!` and `step_until!` advance a state.
- consistency.jl: `consistency_errors(state, params)` returns every disagreement between the structs, the flat data and the grid. Use it as `@test consistency_errors(state, params) == String[]`, between whole steps only. It is the one place tests read the flat layout: other tests use its accessors (`flat_radius`, `flat_tether`, `moved_since_rebuild`, `sorted_ix`).
- oracles.jl: `expected_displacements` / `force_step_matches_oracle` (brute-force all-pairs force step, built from the package's pairwise force functions), `largest_free_disk` (hole search), `disk_area`.

**Conventions:**
- Use explicit RNGs (`rng=Xoshiro(k)`). Stochastic behaviour is either forced deterministic or tested statistically in slow/.
- Known bugs are recorded as `@test_broken` and switched to `@test` once fixed. The consistency checker may tolerate a known issue (listed at its top) while a matching `@test_broken` tracks it.
- Tests that write output use a `mktempdir()` output directory. The path resolves via `joinpath("out", abs_path)`, so nothing lands in the repo's `out/`.

## Repo conventions

- `out/`, `workbooks/`, `TMP_*.jl` scratch scripts and all configs except the two `demo.json` files are gitignored.
- README asks that tests be run before merging into `dev`.
