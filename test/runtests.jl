# Runs the test suite. Pass file names to run a subset, e.g.
#   julia --project -e 'using Pkg; Pkg.test(test_args=["forces"])'

include("setup.jl")

const TEST_FILES = ["utils", "forces", "initialise"]   # test/test_<name>.jl, run in this order
selected = isempty(ARGS) ? TEST_FILES : ARGS
unknown = setdiff(selected, TEST_FILES)
isempty(unknown) || error("Unknown test file(s) $unknown; choose from $TEST_FILES")

@testset "MembraneABMSim" begin
    for name in selected
        include("test_$name.jl")
    end
end
