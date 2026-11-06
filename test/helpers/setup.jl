# Shared setup for every test file: loads the package and the test helpers.
#
# Included by runtests.jl and at the top of each test file, so a single file can be run on its
# own (e.g. `include("test/fast/test_step.jl")` from the repo root or a REPL). Safe to include
# repeatedly: the package and helpers are only loaded once.

using Test
using Random
using LinearAlgebra
using StaticArrays
using Accessors
using JLD2
using JSON3

# the package is used via `include` rather than `using` (see README), so do the same here
if !isdefined(Main, :MembraneABMSim)
    include(joinpath(@__DIR__, "..", "..", "src", "MembraneABMSim.jl"))
end

if !isdefined(Main, :TEST_HELPERS_LOADED)
    const M = MembraneABMSim
    include(joinpath(@__DIR__, "builders.jl"))
    include(joinpath(@__DIR__, "consistency.jl"))
    include(joinpath(@__DIR__, "oracles.jl"))
    const TEST_HELPERS_LOADED = true
end
