# Runs the test suite. Tests live in tier folders, and every test/<tier>/test_*.jl file is run:
#   fast/  invariants, oracles and short runs (about a minute)
#   slow/  statistical checks and longer runs
#   gpu/   CPU-vs-GPU agreement; each backend runs only if its device is present
#
# With no arguments, runs fast and gpu. Arguments select tiers and/or single files:
#   julia --project -e 'using Pkg; Pkg.test()'                                # fast + gpu
#   julia --project -e 'using Pkg; Pkg.test(test_args=["slow"])'              # one tier
#   julia --project -e 'using Pkg; Pkg.test(test_args=["all"])'               # every tier
#   julia --project -e 'using Pkg; Pkg.test(test_args=["biology", "step"])'   # named files, any tier
#   julia --project -e 'include("test/fast/test_step.jl")'                    # one file directly

include(joinpath(@__DIR__, "helpers", "setup.jl"))

const TIERS = ["fast", "slow", "gpu"]
const DEFAULT_TIERS = ["fast", "gpu"]

tier_files(tier) = sort(filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(joinpath(@__DIR__, tier))))

# resolve the arguments to (tier, file) pairs, keeping tier order
function select_tests(args)
    isempty(args) && (args = DEFAULT_TIERS)
    "all" in args && (args = TIERS)
    selected = Tuple{String, String}[]
    for tier in TIERS, file in tier_files(tier)
        name = chopsuffix(chopprefix(file, "test_"), ".jl")
        if tier in args || name in args || file in args || "test_$name" in args
            push!(selected, (tier, file))
        end
    end
    known = vcat(TIERS, [chopsuffix(chopprefix(f, "test_"), ".jl") for t in TIERS for f in tier_files(t)])
    unknown = filter(a -> !(chopsuffix(chopprefix(a, "test_"), ".jl") in known), args)
    isempty(unknown) || error("Unknown test tier or file(s) $unknown; choose from $known")
    return selected
end

@testset "MembraneABMSim" begin
    for tier in TIERS
        files = [f for (t, f) in select_tests(ARGS) if t == tier]
        isempty(files) && continue
        @testset "$tier" begin
            for file in files
                include(joinpath(@__DIR__, tier, file))
            end
        end
    end
end
