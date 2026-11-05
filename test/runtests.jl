using Test
using StaticArrays
using LinearAlgebra
using Random

# The package is used via `include` rather than `using` (see README), so do the same here.
if !isdefined(Main, :MembraneABMSim)
    include(joinpath(@__DIR__, "..", "src", "MembraneABMSim.jl"))
end
const M = MembraneABMSim

# demo simulation config, used as a base parameter set by several test files
const DEMO_CONFIG = joinpath(@__DIR__, "..", "config", "simulation", "demo.json")

@testset "MembraneABMSim" begin
    include("test_utils.jl")
    include("test_forces.jl")
end
