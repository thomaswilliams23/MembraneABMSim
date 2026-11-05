using Test
using StaticArrays
using LinearAlgebra

# The package is used via `include` rather than `using` (see README), so do the same here.
if !isdefined(Main, :MembraneABMSim)
    include(joinpath(@__DIR__, "..", "src", "MembraneABMSim.jl"))
end
const M = MembraneABMSim

@testset "MembraneABMSim" begin
    include("test_utils.jl")
    include("test_forces.jl")
end
