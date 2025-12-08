

module MembraneABMSim

    using Accessors
    using CairoMakie
    using Colors
    using CUDA
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

    include("cuda/cudaTypes.jl")
    include("cuda/data_transfer_cuda.jl")
    include("cuda/forces_cuda.jl")
    include("cuda/initialise_cuda.jl")
    include("cuda/other_position_updates_cuda.jl")
    include("cuda/utils_cuda.jl")

    include("postprocessing/postprocessing_utils.jl")
    include("postprocessing/metrics.jl")
    include("postprocessing/vis_membrane.jl")

    include("biology.jl")
    include("initialise.jl")
    include("file_io.jl")
    include("forces.jl")
    include("main.jl")
    include("nascent.jl")
    include("sweep.jl")
    include("update_grid.jl")
    include("utils.jl")

end # module MembraneABMSim
