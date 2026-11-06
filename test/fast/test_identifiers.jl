# Compact agent identifiers (src/utils.jl): make_identifier, parse_identifier
#
# Contracts: every valid combination of properties round-trips, and the identifiers have the two
# arithmetic properties the force code relies on (sign = tethered, parity = OMP vs LPS). The
# actual integer values are not tested, so the encoding can change.

include(joinpath(@__DIR__, "..", "helpers", "setup.jl"))

@testset "identifiers" begin

    # every valid combination: an agent is at most one of inserting/nascent, and only an
    # inserting or nascent agent has a nascent index
    combos = [(; is_tethered, agent_type, is_inserting, is_nascent, nascent_ix)
              for is_tethered in (false, true)
              for agent_type in ("OmpA", "OmpCF", "BamA", "LptD", "LPS")
              for (is_inserting, is_nascent) in ((false, false), (true, false), (false, true))
              for nascent_ix in ((is_inserting || is_nascent) ? (1, 2, 17, 98) : (0,))]

    @testset "round trip" begin
        for c in combos
            id = M.make_identifier(; c...)
            is_tethered, agent_type, is_nascent, is_inserting, nascent_ix = M.parse_identifier(id)
            @test (is_tethered, agent_type, is_nascent, is_inserting, nascent_ix) ==
                  (c.is_tethered, c.agent_type, c.is_nascent, c.is_inserting, c.nascent_ix)
        end
        # distinct properties give distinct identifiers
        @test allunique(M.make_identifier(; c...) for c in combos)
    end

    @testset "properties used by the force code" begin
        for c in combos
            id = M.make_identifier(; c...)
            @test (id < 0) == c.is_tethered                       # sign: tethered
            @test isodd(id & 1) == (c.agent_type != "LPS")        # parity: OMP (odd) vs LPS (even)
        end
    end

    @testset "invalid combinations are rejected" begin
        @test_throws AssertionError M.make_identifier(; is_tethered=false, agent_type="OmpA",
                                                      is_inserting=true, is_nascent=true, nascent_ix=1)
        @test_throws Exception M.make_identifier(; is_tethered=false, agent_type="NotAType",
                                                 is_inserting=false, is_nascent=false, nascent_ix=0)
    end
end
