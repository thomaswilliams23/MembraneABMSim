# Shared setup for the test files: loads the package and defines helpers used by several files.
# Included by runtests.jl and at the top of each test file, so a single test file can be run
# on its own (e.g. `include("test/test_forces.jl")` in a REPL). Safe to include repeatedly.

using Test
using StaticArrays
using LinearAlgebra
using Random

# The package is used via `include` rather than `using` (see README), so do the same here.
if !isdefined(Main, :MembraneABMSim)
    include(joinpath(@__DIR__, "..", "src", "MembraneABMSim.jl"))
end

if !isdefined(@__MODULE__, :TEST_SETUP_LOADED)

    const M = MembraneABMSim

    # demo simulation config, used as a base parameter set
    const DEMO_CONFIG = joinpath(@__DIR__, "..", "config", "simulation", "demo.json")

    # replace fields of an immutable struct by keyword
    with_fields(obj; kw...) = foldl((o, (k, v)) -> M.change_field(o, String(k), v), pairs(kw); init=obj)

    # demo parameters with selected fields replaced, by section, e.g. test_params(system=(dt=1e-3,))
    function test_params(; sections...)
        p = M.parse_config(DEMO_CONFIG)
        for (section, fields) in pairs(sections)
            p = M.change_field(p, String(section), with_fields(getfield(p, section); fields...))
        end
        return p
    end

    # all agents, in the same order the source iterates them
    agent_list(agents) = collect(Iterators.flatten((agents.OMP.OmpA, agents.OMP.OmpCF, agents.OMP.LptD,
        agents.OMP.BamA, agents.LPS, agents.nascent.nascent_OMP, agents.nascent.nascent_LPS)))

    # position of sorted agent `s` in the flat data, and an agent's sorted index
    flat_pos(flat, s) = SVector(flat.positions[2s-1], flat.positions[2s])
    sorted(flat, agent) = flat.agent_ix_to_sorted_ix[agent.index]

    const TEST_SETUP_LOADED = true
end
