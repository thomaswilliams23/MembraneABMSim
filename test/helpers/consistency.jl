# Consistency checker: do the agent structs, the flat (sorted) arrays and the grid all describe
# the same system?
#
# Use as `@test consistency_errors(state, params) == String[]`, so a failure prints every
# violation. Valid between whole steps (after `step!`, `sync_grid!` or initialisation): within a
# step the biology code deliberately adds agents before the flat data is rebuilt.
#
# This is the one helper that reads the flat-array layout. If the flat representation changes,
# update the accessors in the first section and the checks below should still apply.
#
# Known issues tolerated here (each has a @test_broken that flips when it is fixed): none.

# ---- flat-layout accessors (the only place tests read AllAgentsFlat / SimGrid directly) ----

flat_position(flat, s) = SVector(flat.positions[2s-1], flat.positions[2s])
flat_tether(flat, s) = SVector(flat.tether_points[2s-1], flat.tether_points[2s])
sorted_ix(state, a) = state.flat.agent_ix_to_sorted_ix[a.index]
flat_radius(state, a) = state.flat.effective_radii[sorted_ix(state, a)]
flat_tether(state, a::M.AbstractAgent) = flat_tether(state.flat, sorted_ix(state, a))

"distance each agent has moved since the last grid rebuild (the neighbour search's margin)"
moved_since_rebuild(state) = state.flat.agg_dist_since_grid_sync[1:state.grid_size.num_agents]

"decoded identifier of sorted agent `s`: (tethered, type, nascent, inserting, nascent_ix)"
function flat_identity(flat, s)
    is_tethered, type, is_nascent, is_inserting, nascent_ix = M.parse_identifier(flat.identifiers[s])
    return (; is_tethered, type, is_nascent, is_inserting, nascent_ix)
end

"rectangle (lower corner, width) of grid cell `z`"
function cell_rect(state, z)
    g, gs = state.grid, state.grid_size
    w = gs.dims ./ gs.num_cells
    cx, cy = g.z_ix_to_coords[2z-1], g.z_ix_to_coords[2z]
    return SVector((cx - 1) * w[1], (cy - 1) * w[2]), w
end

"periodic distance from point `p` to the rectangle of grid cell `z` (0 if inside)"
function distance_to_cell(state, p, z)
    lo, w = cell_rect(state, z)
    d = abs.(M.shortest_vec(p, lo .+ w ./ 2, state.grid_size.dims))
    return norm(max.(d .- w ./ 2, 0.0))
end


# ---- the checker ----

"""
    consistency_errors(state, params; tol=1e-9) -> Vector{String}

Every way in which the structs, flat data and grid of `state` disagree (empty if consistent).
"""
function consistency_errors(state::M.SimState, params::M.AllParams; tol::Float64=1e-9)
    errs = String[]
    fail(msg) = push!(errs, msg)
    agents, flat, grid, gs = state.agents, state.flat, state.grid, state.grid_size
    dims = gs.dims
    as = all_agents(state)
    N = gs.num_agents

    # pending updates mean we are mid-step
    gs.num_agents_changed && fail("num_agents_changed is set (grid not rebuilt after insertion)")
    gs.nascent_promoted && fail("nascent_promoted is set (flat data not updated after promotion)")

    # agent count and indices
    length(as) == N || fail("grid_size.num_agents = $N but there are $(length(as)) agents")
    sort([a.index for a in as]) == 1:length(as) || fail("agent indices are not 1:N")
    isempty(errs) || return errs   # nothing below is meaningful without valid indices

    # index maps are inverse permutations
    a2s = flat.agent_ix_to_sorted_ix[1:N]
    s2a = flat.sorted_ix_to_agent_ix[1:N]
    isperm(a2s) || return push!(errs, "agent_ix_to_sorted_ix is not a permutation of 1:N")
    all(s2a[a2s[i]] == i for i in 1:N) || fail("sorted_ix_to_agent_ix is not the inverse of agent_ix_to_sorted_ix")

    # domain
    all(dims .> 0) && all(isfinite, dims) || fail("bad domain dims $dims")

    # each agent: struct and flat data agree
    inserter_of = Dict{Int, Any}()   # inserting agent index => its nascent agent
    for a in as
        s = a2s[a.index]
        id = flat_identity(flat, s)
        tag = "$(agent_type(a)) $(a.index)"

        all(isfinite, a.position) || fail("$tag: non-finite position $(a.position)")
        all(0 .<= a.position .< dims) || fail("$tag: position $(a.position) outside domain $dims")
        flat_position(flat, s) == a.position || fail("$tag: flat position $(flat_position(flat, s)) != struct position $(a.position)")
        r_flat, r_expected = flat.effective_radii[s], current_radius(params, a)
        isapprox(r_flat, r_expected; atol=tol) ||
            fail("$tag: flat radius $(r_flat) != expected $(r_expected)")

        id.type == agent_type(a) || fail("$tag: flat identifier says type $(id.type)")
        id.is_nascent == is_nascent(a) || fail("$tag: flat nascent flag is $(id.is_nascent)")

        # tethering
        if hasproperty(a, :is_tethered)
            id.is_tethered == a.is_tethered || fail("$tag: flat tethered flag $(id.is_tethered) != struct $(a.is_tethered)")
            if a.is_tethered
                flat_tether(flat, s) == a.tether_point || fail("$tag: flat tether point != struct tether point")
                d = M.shortest_distance(a.position, a.tether_point, dims)
                d <= tether_radius(params, a) + tol || fail("$tag: $(d) from its tether point, beyond tether radius $(tether_radius(params, a))")
            end
        else
            id.is_tethered && fail("$tag: flat identifier says tethered, but this type cannot be")
        end

        # insertion state: an embedding BamA/LptD is the flat "inserting" agent
        inserting = hasproperty(a, :insertion_state) && a.insertion_state == "embedding"
        id.is_inserting == inserting || fail("$tag: flat inserting flag $(id.is_inserting) but insertion_state is $(hasproperty(a, :insertion_state) ? a.insertion_state : "n/a")")
        if hasproperty(a, :insertion_state)
            a.insertion_state in ("free", "stalled", "embedding") || fail("$tag: unknown insertion_state $(a.insertion_state)")
            a isa M.LptDAgent && a.insertion_state == "stalled" && fail("$tag: LptD cannot be stalled")
            a.insertion_state != "free" && !a.is_part_of_assembled_complex && fail("$tag: unassembled but insertion_state is $(a.insertion_state)")
        end
    end

    # nascent agents: linked one-to-one with an embedding inserter, in flat order OMPs then LPS
    nascents = vcat(agents.nascent.nascent_OMP, agents.nascent.nascent_LPS)
    by_index = Dict(a.index => a for a in as)
    for (k, n) in enumerate(nascents)
        tag = "nascent $(agent_type(n)) $(n.index)"
        ins = get(by_index, n.inserting_agent_index, nothing)
        expected_inserter = n isa M.NascentLPSAgent ? M.LptDAgent : M.BamAAgent
        if !(ins isa expected_inserter)
            fail("$tag: inserting agent $(n.inserting_agent_index) is not a $(expected_inserter)")
            continue
        end
        ins.insertion_state == "embedding" || fail("$tag: its inserter is $(ins.insertion_state), not embedding")
        haskey(inserter_of, ins.index) && fail("$tag: inserter $(ins.index) has more than one substrate")
        inserter_of[ins.index] = n

        sn, si = a2s[n.index], a2s[ins.index]
        flat.nascent_to_substrate_ixs[k] == sn || fail("$tag: flat substrate link $(flat.nascent_to_substrate_ixs[k]) != $sn")
        flat.nascent_to_inserting_ixs[k] == si || fail("$tag: flat inserting link $(flat.nascent_to_inserting_ixs[k]) != $si")
        flat_identity(flat, sn).nascent_ix == k || fail("$tag: flat nascent index != $k")
        flat_identity(flat, si).nascent_ix == k || fail("$tag: its inserter's flat nascent index != $k")
        isapprox(flat.substrate_inserting_ideal_dists[k], n.ideal_dist_from_inserting_agent; atol=tol) ||
            fail("$tag: flat ideal distance != struct ideal distance")

        # nascent geometry is physical (the upper bound, final size, is checked in test_nascent.jl:
        # it is currently exceeded by one growth increment just before promotion)
        n isa M.NascentOMPAgent && !(n.effective_radius > 0) &&
            fail("$tag: non-positive effective radius $(n.effective_radius)")
        n.ideal_dist_from_inserting_agent >= 0 ||
            fail("$tag: negative ideal distance $(n.ideal_dist_from_inserting_agent)")
    end
    for a in as
        if hasproperty(a, :insertion_state) && a.insertion_state == "embedding" && !haskey(inserter_of, a.index)
            fail("$(agent_type(a)) $(a.index) is embedding but has no nascent substrate")
        end
    end

    # grid: every agent belongs to exactly one cell's contiguous sorted range, and that cell is
    # within the distance the agent has moved since the last rebuild of its true cell
    tot = gs.tot_num_cells
    sum(grid.num_agents_in_cell[1:tot]) == N || fail("grid cell counts sum to $(sum(grid.num_agents_in_cell[1:tot])), not $N")
    covered = falses(N)
    for z in 1:tot
        n_in = grid.num_agents_in_cell[z]
        n_in == 0 && continue
        r = grid.start_agents_in_cell[z]:(grid.start_agents_in_cell[z] + n_in - 1)
        if first(r) < 1 || last(r) > N
            fail("grid cell $z covers sorted indices $r, outside 1:$N")
            continue
        end
        any(covered[r]) && fail("grid cell $z overlaps another cell's sorted range")
        covered[r] .= true
        for s in r
            grid.agent_cell_z_ixs[s] == z || fail("sorted agent $s is in cell $z's range but records cell $(grid.agent_cell_z_ixs[s])")
            slack = flat.agg_dist_since_grid_sync[s] + tol
            d = distance_to_cell(state, flat_position(flat, s), z)
            d <= slack || fail("sorted agent $s is $(d) from its grid cell $z (allowed $(slack))")
        end
    end
    all(covered) || fail("$(count(!, covered)) agents are not in any grid cell")

    agents.num_PP >= 0 || fail("negative polypeptide count $(agents.num_PP)")

    return errs
end
