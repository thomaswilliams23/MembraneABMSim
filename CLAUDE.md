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
julia --project -e 'using Pkg; Pkg.test()'
```

- There is no per-test runner. To run a single test file quickly, include `test/runtests.jl`'s preamble (everything above the `@testset` block: package include, `M = MembraneABMSim`, shared helpers), then `include("test/test_<area>.jl")`.
- `Metal` and `CUDA` are hard dependencies loaded at module load. Errors from them on machines without that GPU are expected and can be ignored.

## Architecture

**Simulation loop** (`run_sim`, src/main.jl):
1. Parse the JSON config into `AllParams` (src/types/ParamTypes.jl, via JSON3/StructTypes).
2. Precompute the nascent-insertion lookup tables (`build_nascent_added_area_lookup`, `compute_nascent_incs`).
3. Initialise the system (random / specify_positions / from_checkpoint).
4. Each step:
   - Update the BAM and Lpt insertion subsystems (src/biology.jl).
   - Rebuild the grid if the agent count changed.
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
- The grid is rebuilt only when agents are added, or every `MAX_STEPS_BETWEEN_GRID_SYNC` steps. Between rebuilds, the neighbour search widens its window by `agg_dist_since_grid_sync`, the distance agents have moved since the last rebuild.
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

- `test/runtests.jl` holds the shared helpers: `DEMO_CONFIG`; `test_params(; section=(field=value,)...)`, which builds demo params with fields overridden (immutable structs are edited with `change_field`); `agent_list`, `flat_pos`, `sorted`.
- There is one test file per source file, with testsets in source order and a header listing the functions covered.
- Physics tests check invariants only: zeros, saturation, sign, bounds, continuity, Newton's third law, periodic wrap, and "aggregate = sum of pairwise calls". They must not pin exact force-law values, because the functional forms may change.
- Known bugs are recorded as `@test_broken` and switched to `@test` once fixed.
- Tests that write output set `system.output_dir` to a `mktempdir()`. The path resolves via `joinpath("out", abs_path)`, so nothing lands in the repo's `out/`.

## Repo conventions

- `out/`, `workbooks/`, `TMP_*.jl` scratch scripts and all configs except the two `demo.json` files are gitignored.
- README asks that tests be run before merging into `dev`.
