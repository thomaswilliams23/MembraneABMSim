# MembraneABMSim Documentation

MembraneABMSim is a Julia package implementing an agent-based model of growth of the Gram-negative outer membrane. It simulates the spatial dynamics of outer membrane proteins (OMPs) and lipopolysaccharide (LPS) molecules in a growing patch of membrane.

---

## Installation

MembraneABMSim is not yet a registered Julia package. To set it up, clone the repository, launch Julia from the repo root, and run:

```julia
]activate .
]instantiate
```

Instantiation may take a few minutes on first run. You can safely ignore errors from the Metal and CUDA packages — these only matter if you intend to use a GPU. Once complete, load the package with:

```julia
include("src/MembraneABMSim.jl")
```

This also takes a moment on first run due to compilation. Once it returns, you're ready to go.

---

## Basic Workflow

### Running a Single Simulation

Simulations are configured via a JSON file (by convention kept in `config/simulation/`). An example config is given in `config/simulation/demo.json`, and detailed config options are provided below. To run a simulation, pass the path to your config file to `run_sim`:

```julia
MembraneABMSim.run_sim("<path/to/your_config.json>")
```

Output is written to `out/<output_dir>/`, where `output_dir` is set in your config. Raw simulation state is saved at regular intervals to `out/<output_dir>/raw_data/` as `.jld2` files. A copy of the config is also saved to the output directory.

`run_sim` accepts a few optional keyword arguments:

- `clear_existing_output=true` — overwrites any existing output in the output directory (default to `false`, in which case, if the output directory is non-empty, the user is asked to confirm whether to wipe its contents)
- `suppress_prints=true` — silences progress output (useful when running many simulations programmatically)

### Running a Parameter Sweep

Sweeps are configured via a sweep config JSON (by convention in `config/sweep/`). Note that sweep config files have a different structure to config files for single simulations, which is described below. An example can be found in `config/sweep/demo.json`. Run a sweep with:

```julia
MembraneABMSim.run_sweep("<path/to/your_sweep_config.json>")
```

The sweep runs simulations across a specified grid of parameter values, with a specified number of simulation replicates per combination. By default, simulations are parallelised across CPU threads using Julia's `Threads.@threads`. Output is written to a structured directory tree under `out/<output_base_dir>/`, organised as:

```
out/<output_base_dir>/<param1_name><val_index>/<param2_name><val_index>/.../rep<rep_index>/
```

`run_sweep` also accepts `run_serially_yn=true` to force serial execution (this is set automatically when using a GPU device).

---

## Post-Processing

### Making Simulation Movies

To generate a movie of a completed simulation, use `make_membrane_movie` with the simulation's output directory name (relative to `out/`). For example, if the output directory is `my_sim`, run:

```julia
MembraneABMSim.make_membrane_movie("my_sim")
```

This reads frames from `out/my_sim/raw_data/` (generated during the simulation) and writes a movie to `out/my_sim/sim.mp4`.

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `fps` | `24` | Frames per second |
| `output_extension` | `".mp4"` | Output format: `".mp4"`, `".avi"`, or `".gif"` |
| `plot_size` | `500` | Side length of the output video in pixels |
| `frame_inds` | all frames | A range or vector of frame indices to include |
| `plot_time_series` | `false` | Show a live membrane size plot alongside the visualisation |
| `centre_on_agent` | `nothing` | Keep a specific agent centred, e.g. `("BamA", 1)` |
| `centre_on_agents` | `nothing` | Keep the centroid of a list of agents centred, e.g. `[("BamA", 1), ("LptD", 1)]` |
| `centre_at_point` | `[0.5, 0.5]` | If centering on an agent or agents, this specifies the normalised coordinates (within the membrane) to put that centre |
| `show_tethers_yn` | `true` | Draw tether lines for tethered agents |
| `show_insertion_states_yn` | `true` | Draw outlines indicating BamA/LptD insertion states |
| `demarcate_new_LPS_yn` | `false` | Show LPS added during the simulation in a darker colour |

**Example — centred movie at 12 fps:**
```julia
MembraneABMSim.make_membrane_movie("my_sim";
    fps = 12,
    centre_on_agents = [("BamA", 1), ("LptD", 1)]
)
```

### Snapshots

To save a single frame as an image, use `make_snapshot` with the output directory (relative to `./out`), and the time from which to generate the snapshot:

```julia
MembraneABMSim.make_snapshot("my_sim", 15.0)  # snapshot at t=15.0
```

This saves both a raster (`.png`) and a vector (`.svg`) image to `out/my_sim/snapshots/` and `out/my_sim/vector_snapshots/` respectively.

To make snapshots at multiple time points:

```julia
MembraneABMSim.make_snapshots("my_sim", [5.0, 10.0, 15.0, 20.0])
```

To make consistently-sized snapshots across multiple simulations, especially across a sweep (useful for comparison figures), use:

```julia
paths = ["<output_dir_1>", "output_dir_2"]
MembraneABMSim.make_snapshots_across_sweep(paths, [15.0, 30.0])
```

This automatically determines the largest membrane across all systems and uses it as a common frame size.

### Analysing Sweeps

The `analyse_sweep` function (in `src/postprocessing/postprocessing_utils.jl`) applies a metric function to every simulation in a sweep and saves the results (this is a wrapper to `analyse_sim`, which is the single-simulation analogue of this function). Several pre-built metric functions are available in `src/postprocessing/metrics.jl`, or you can write your own.

_If writing your own metric function, it's probably easiest to take a look at the existing metric functions and modify them. Note that metric functions for a single time point and metric functions over an entire simulation trajectory have different call signatures._

Metric functions which apply to individual time points of a simulation (like the number of OMPs, membrane size etc) can be called with one of the following calls, where `my_metric` is the name of your metric function you want to use:

```julia
MembraneABMSim.analyse_sweep(
    my_metric, 
    "output_file_name", 
    "<path/to/your_sweep_config.json>",
    get_time_series_yn=true
) #evaluates my_metric at all time points across the sweep
MembraneABMSim.analyse_sweep(
    my_metric, 
    "output_file_name", 
    "<path/to/your_sweep_config.json>",
    evaluate_final_state_yn=true
) #evaluates my_metric at the final time state only
MembraneABMSim.analyse_sweep(
    my_metric, 
    "output_file_name", 
    "<path/to/your_sweep_config.json>",
    evaluate_initial_state_yn=true
) #evaluates my_metric at the initial time state only
```


Metric functions which apply to an entire simulation trajectory (like the time at which the BamA first becomes stalled etc) are called as:

```julia
MembraneABMSim.analyse_sweep(
    my_metric, 
    "output_file_name", 
    "<path/to/your_sweep_config.json>",
    analyse_whole_traj_yn=true
)
```

**Built-in metrics:**

Metrics on a single time point:

| Function | Returns |
|----------|---------|
| `get_membrane_size` | Area of the membrane (product of dimensions) |
| `get_num_agents` | Dict of agent counts by type |
| `get_num_OmpA_agents` | Number of OmpA agents |
| `get_num_LPS_agents` | Number of LPS agents |
| `get_domain_coverage` | Fraction of membrane area covered by agents |
| `get_ideal_domain_coverage` | Theoretical coverage assuming no overlaps |
| `get_num_LPS_bordering_OMP` | Number of LPS adjacent to any OMP |
| `get_prop_LPS_bordering_OMP` | Proportion of LPS adjacent to any OMP |
| `get_BAM_states` | List of insertion states of BamA agents |
| `get_BAM_stalled_status` | Boolean vector: which BamA agents are in the "stalled" state |
| `get_distance_between_BAMs` | Pairwise distance matrix between BamA agents |
| `get_LPS_clusters` | List of LPS clusters (groups of touching LPS) |

Metrics on entire simulation trajectories:

| Function | Returns |
|----------|---------|
| `get_cutoff_time` | Time at which each BamA first loses access to LPS |
| `get_squared_displacement` | Squared displacement trajectories for all agents |
| `get_num_OmpA_agents_by_BAM` | Count of OmpA inserted by each BamA over time |
| `get_LptD_insertion_times` | Insertion times of all LptD agents |

---

## Using Other Devices (Metal and CUDA)

By default, simulations run on the CPU. To use a GPU, set `"device"` in the `"system"` section of your config:

```json
"system": {
    "device": "metal",
    ...
}
```

Valid values are `"cpu"` (default), `"metal"` (Apple Silicon GPU), and `"cuda"` (NVIDIA GPU). The corresponding hardware and Julia packages must be available; if a GPU is specified but not found, the simulation will error at startup.

**Notes on GPU usage:**

- Explicit diffusion (Brownian motion) is **not currently supported** for GPU computation.
- Aside from diffusion, Metal and CUDA implementations are functionally equivalent to the CPU version, but offload the computationally intensive force calculations to the GPU.
- When running a sweep with a GPU device, simulations are automatically run serially (no thread parallelism), since the GPU is a shared resource.
- GPU support requires `Metal.jl` (Apple) or `CUDA.jl` (NVIDIA) to be functional — errors from these packages during `instantiate` can be safely ignored if you only plan to use the CPU.

---

## Simulation Config Reference

Simulation configs are JSON files with the following top-level sections.

### `OmpA`

Parameters for OmpA: small outer membrane protein. Can form a tether which limits its motion.

| Key | Type | Description |
|-----|------|-------------|
| `radius` | float | Physical radius of the agent |
| `insertion_prob` | float | Probability that a new OMP insertion event produces an OmpA, in the range [0, 1] |
| `tether_rate` | float | Rate at which OmpA agents become tethered to the BAM complex |
| `tether_radius` | float | Maximum distance that the OmpA can move from its tether point|

### `OmpCF`

Parameters for OmpC/OmpF monomer: large outer membrane protein which forms a homotrimer (not yet implemented).

| Key | Type | Description |
|-----|------|-------------|
| `radius` | float | Physical radius of the agent |
| `insertion_prob` | float | Probability that a new OMP insertion event produces an OmpCF |

### `LptD`

Parameters for LptD: outer membrane protein component of the Lpt complex, responsible for inserting LPS into the membrane. The assembled Lpt complex is considered tethered in place.

| Key | Type | Description |
|-----|------|-------------|
| `radius` | float | Physical radius of the agent |
| `insertion_prob` | float | Probability that a new OMP insertion event produces an LptD|
| `complex_assembly_rate` | float | Rate at which LptD is incorporated into a complete Lpt complex |
| `tether_radius` | float | Maximum distance that the LptD can move from its tether point |

### `BamA`

Parameters for BamA: outer membrane protein component of the BAM complex, responsible for folding OMPs into the membrane.

| Key | Type | Description |
|-----|------|-------------|
| `radius` | float | Physical radius of the agent|
| `insertion_prob` | float | Probability that a new OMP insertion event produces a BamA|
| `complex_assembly_rate` | float | Rate at which BamA is incorporated into a complete BAM complex|

### `LPS`

Parameters for lipopolysaccharide (LPS) molecules.

| Key | Type | Description |
|-----|------|-------------|
| `radius` | float | Physical radius of the agent|
| `arrival_rate` | float | Rate at which new LPS molecules arrive at the membrane |
| `insertion_time` | float | Time taken for a newly arrived LPS molecule to fully insert |

### `init`

Initial conditions.

| Key | Type | Description |
|-----|------|-------------|
| `dim_x` | float | Initial width of the membrane domain |
| `dim_y` | float | Initial height of the membrane domain (if not equal to `dim_x` you might encounter some unexpected behaviour) |
| `num_OmpA` | int | Number of OmpA agents at t=0 |
| `num_OmpCF` | int | Number of OmpCF agents at t=0 |
| `num_LptD` | int | Number of LptD agents at t=0 |
| `num_BamA` | int | Number of BamA agents at t=0 |
| `num_LPS` | int | Number of LPS agents at t=0 |
| `num_PP` | int | Number of substrate polypeptides at t=0 |
| `equilibration_time` | float | Length of time to equilibrate the system before shrinking the domain to size (optional) and running simulation|
| `complexes_assembled` | bool | If `true`, all complexes and tethers for existing agents are considered assembled at t=0 |
| `method` | string | Initialisation method for agent positions. Current options are: `"random"` - places all agents randomly within the domain, `"specify_positions"` - place some agents at specified positions using the optional `"positions"` field (the rest are positioned randomly) (**Note:** agents may move during equilibration), `"from_checkpoint"` - resumes simulation from a specified `"checkpoint_file"` (this option skips equilibration); you can optionally set the `"checkpoint_time"` to get consistent time values for the output of this simulation.|
| `"positions"` | (see description, optional) | If `"method"` is set to `"specify_positions"`, specifies where certain agents should be placed. Under this field, you can add fields for `"OmpA"`, `"OmpCF"`, `"BamA"`, `"LptD"` and `"LPS"`, each of which should be given as a vector of two-element position vectors. |
| `checkpoint_file` | string (optional) | Path to a `.jld2` checkpoint file to resume from (must be specified if `"method"` is set to `"from_checkpoint"`) |
| `checkpoint_time` | float (optional) | Final time corresponding to the checkpoint file|
| `shrink_to_size` | bool (optional) | If `true`, iteratively shrinks the domain and re-equilibrates to remove any holes|
| `shrinkage_factor` | float (optional) | If `shrink_to_size` is set to `true`, specifies the factor by which the domain is shrunk in each stage of shrinkage (defaults to 0.01) |
| `max_num_shrinkage_rounds` | int (optional) | If `shrink_to_size` is set to `true`, specifies the maximum number of rounds to try shrinking the domain and re-equilibrating to remove holes before giving up (defaults to 10)|
| `shrinkage_equilibration_time` | float (optional) | If `shrink_to_size` is set to `true`, specifies the length of time to re-equilibrate the system after each round of domain shrinkage (defaults to 3.0)|

### `insertion`

Controls the OMP insertion process, including the nascent-inserting force.

| Key | Type | Description |
|-----|------|-------------|
| `type` | string | Insertion mode. Current options are: `"lipid-dependent"` - OMP insertion is only possible when LPS is within the sensing radius of the BAM, `"guaranteed"` - OMP insertion happens regardless. |
| `attempt_dt` | float | Time interval between insertion attempts |
| `mu_rep` | float | Repulsion strength for the nascent-inserting force |
| `mu_attr` | float | Attraction strength for the nascent-inserting force |
| `k_C` | float | Rate at which attraction force decays over distance for the nascent-inserting force |
| `PP_arrival_rate` | float | Rate at which substrate polypeptides arrive in the system |
| `PP_bind_rate` | float | Rate at which substrate polypeptides bind to the BAM complex |
| `OMP_assembly_rate` | float | Rate of OMP embedding by the BAM relative to OMP size |

### `force`

Controls attraction-repulstion forces and diffusion.

| Key | Type | Description |
|-----|------|-------------|
| `temperature` | float | Controls the extent of explicit diffusion (Brownian motion); set to 0.0 for no Brownian motion (**note:** if setting `temperature` to greater than zero, you must specify the `diffusion_mode` to be `"LPS_only"`) |
| `diffusion_mode` | string (optional) | Specifies which agents undergo explicit diffusion (Brownian motion) (**note:** `"LPS_only"` is the only currently supported mode for diffusion if you set `temperature` to be greater than zero) |
| `mu_rep` | float | Repulsion coefficient for attraction-repulsion force |
| `mu_attr_OMP_OMP` | float | Attraction coefficient between OMP agents |
| `mu_attr_OMP_LPS` | float | Attraction coefficient between OMP and LPS agents |
| `mu_attr_LPS_LPS` | float | Attraction coefficient between LPS agents |
| `rho` | float | Controls how sharply the repulsion force drops for overlapping agents (vertical asymptote at rho times the sum of agent radii) |
| `max_repulsion` | float | Cap on the maximum repulsive force magnitude (negative value) |
| `k_C` | float | Rate at which attraction force decays over distance for the nascent-inserting force |
| `eta` | float | Viscous drag coefficient (effective mobility = 1/eta) |
| `sensing_radius` | float | Distance within which agents can "sense" each other: defines maximum distance (beyond agent circumferences) at which the attraction force is experienced, and the maximum distance an LPS can be from a BamA to permit OMP insertion. |

### `system`

General simulation settings.

| Key | Type | Description |
|-----|------|-------------|
| `density` | float | Target packing density of the membrane (total agent area per domain area); controls the equilibrium domain size |
| `dt` | float | Integration timestep |
| `t_max` | float | Total simulation duration |
| `vis_dt` | float | Interval between output frames (should be a multiple of `dt`) |
| `output_dir` | string | Name of the output directory (written to `out/<output_dir>/`) |
| `seed` | int (optional) | Random seed for reproducibility |
| `device` | string (optional) | Hardware device: `"cpu"` (default), `"metal"`, or `"cuda"` |
| `max_hole_radius` | float (optional) | If set, membrane growth is paused whenever a hole larger than this radius is detected. Also tells the domain shrinkage procedure (if active) to stop when a hole of this size can no longer be found in the domain. |

---

## Sweep Config Reference

Sweep configs are separate JSON files with the following keys.

| Key | Type | Description |
|-----|------|-------------|
| `sweep_params` | object | Dictionary of parameters to sweep (see below) |
| `num_reps` | int | Number of independent replicates per parameter combination |
| `default_config` | string | Path to the base simulation config (relative to the repo root) |
| `output_base_dir` | string | Base directory for sweep output (written to `out/<output_base_dir>/`) |
| `checkpoint_sweep_config` | string (optional) | Path to a previous sweep config to resume from |

### `sweep_params`

Each entry in `sweep_params` is keyed by the parameter path (using `/` to navigate the config hierarchy, e.g. `"force/mu_attr_LPS_LPS"`), with the following fields:

| Key | Type | Description |
|-----|------|-------------|
| `short_name` | string | A short label used in the output directory names |
| `values` | array | List of values to sweep over |

**Example:**

```json
{
    "sweep_params": {
        "force/mu_attr_LPS_LPS": {
            "short_name": "mu_attr_LPS_LPS",
            "values": [0.01, 0.1, 0.5]
        },
        "insertion/OMP_assembly_rate": {
            "short_name": "assembly_rate",
            "values": [0.01, 0.05, 0.1]
        }
    },
    "num_reps": 5,
    "default_config": "config/simulation/demo.json",
    "output_base_dir": "my_sweep"
}
```

This would run 3 × 3 × 5 = 45 simulations in total. Output paths would be structured like `out/my_sweep/mu_attr_LPS_LPS1/assembly_rate2/rep3/`.

Each simulation in a sweep is automatically assigned a unique random seed (overriding the seed in the default config), ensuring independent replicates and reproducibility. A copy of the exact parameter settings, including random seeds used, for each simulation in the sweep are saved in its output directory.
