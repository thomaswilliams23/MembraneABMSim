

module MembraneABMSim

    using Accessors
    using CairoMakie
    using Colors
    using Distributions
    using GeometryBasics
    using JLD2
    using JSON3
    using KernelAbstractions
    using LinearAlgebra
    using Metal
    using Morton
    using Printf
    using Random
    using StaticArrays
    using StatsBase
    using StructTypes

    include("types/ParamTypes.jl")
    include("types/AgentTypes.jl")
    include("types/GridTypes.jl")
    include("types/SystemTypes.jl")

    include("metal/MetalTypes.jl")
    include("metal/data_transfer_metal.jl")
    include("metal/forces_metal.jl")
    include("metal/initialise_metal.jl")
    include("metal/other_position_updates_metal.jl")
    include("metal/utils_metal.jl")

    include("biology.jl")
    include("initialise.jl")
    include("file_io.jl")
    include("forces.jl")
    include("main.jl")
    include("nascent.jl")
    include("postprocessing.jl")
    include("sweep.jl")
    include("update_grid.jl")
    include("utils.jl")
    include("vis_membrane.jl")

    export parse_config
    export parse_sweep_config
    export run_sim
    export run_sweep
    export make_membrane_movie
    export get_time_series
    export get_membrane_size
    export analyse_sim
    export analyse_sweep

end # module MembraneABMSim
