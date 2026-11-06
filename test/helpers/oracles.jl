# Oracles: slow, obviously-correct references for the optimised parts of the simulation.
#
# They work from the agent structs and the parameters only (never the flat arrays or the grid),
# and call the package's pairwise force functions rather than restating the force laws, so they
# stay valid when a force law changes. What they do encode is the model's update rule
# (overdamped motion, displacement = dt / (eta * radius) * force, tethered moves that would leave
# the tether radius are rejected). If that rule changes, update `expected_displacements`.

"""
    mu_attr_for(params, a, b)

Attraction strength for the pair: OMP-OMP, OMP-LPS or LPS-LPS.
"""
function mu_attr_for(params::M.AllParams, a, b)
    n_OMP = is_OMP(a) + is_OMP(b)
    return n_OMP == 2 ? params.force.mu_attr_OMP_OMP :
           n_OMP == 1 ? params.force.mu_attr_OMP_LPS : params.force.mu_attr_LPS_LPS
end

"""
    insertion_partner(state, a) -> (partner agent or nothing, ideal distance)

The nascent agent's inserter, or the embedding inserter's nascent substrate.
"""
function insertion_partner(state::M.SimState, a)
    nascents = vcat(state.agents.nascent.nascent_OMP, state.agents.nascent.nascent_LPS)
    if is_nascent(a)
        ins = only(filter(b -> b.index == a.inserting_agent_index, all_agents(state)))
        return ins, a.ideal_dist_from_inserting_agent
    end
    for n in nascents
        n.inserting_agent_index == a.index && return n, n.ideal_dist_from_inserting_agent
    end
    return nothing, 0.0
end

"""
    brute_force_forces(state, params) -> Dict(agent index => force)

Net force on every agent by summing the pairwise force over *all* other agents (no grid): the
attraction-repulsion force with everyone except an insertion partner, plus the insertion
spring force with the partner.
"""
function brute_force_forces(state::M.SimState, params::M.AllParams)
    as = all_agents(state)
    dims = state.grid_size.dims
    f, ins = params.force, params.insertion
    forces = Dict{Int, SVector{2, Float64}}()
    for a in as
        partner, ideal = insertion_partner(state, a)
        F = SVector(0.0, 0.0)
        for b in as
            (b === a || b === partner) && continue
            F += M.compute_attr_rep_force(a.position, current_radius(params, a), b.position, current_radius(params, b),
                    dims, f.sensing_radius, mu_attr_for(params, a, b), f.mu_rep, f.max_repulsion, f.rho, f.k_C)
        end
        if !isnothing(partner)
            F += M.compute_inserting_substrate_force(a.position, partner.position, dims, ideal,
                    ins.mu_attr, ins.mu_rep, f.max_repulsion, f.rho, ins.k_C)
        end
        forces[a.index] = F
    end
    return forces
end

"""
    expected_displacements(state, params) -> Dict(agent index => displacement)

Displacement of every agent over one force step (no diffusion), from the brute-force forces.
A nascent OMP moves with the drag of its full-size type. A tethered agent whose move would take
it beyond its tether radius does not move.
"""
function expected_displacements(state::M.SimState, params::M.AllParams)
    dims = state.grid_size.dims
    forces = brute_force_forces(state, params)
    disp = Dict{Int, SVector{2, Float64}}()
    for a in all_agents(state)
        drag_radius = type_radius(params, agent_type(a))
        d = (params.system.dt / (params.force.eta * drag_radius)) * forces[a.index]
        if is_tethered(a)
            next = mod.(a.position + d, dims)
            M.shortest_distance(next, a.tether_point, dims) > tether_radius(params, a) && (d = SVector(0.0, 0.0))
        end
        disp[a.index] = d
    end
    return disp
end

"""
    observed_displacements(before, state) -> Dict(agent index => displacement)

Minimum-image displacement of each agent since `before` (from `positions_by_index`).
"""
function observed_displacements(before::Dict, state::M.SimState)
    now = positions_by_index(state)
    return Dict(i => M.shortest_vec(p, now[i], state.grid_size.dims) for (i, p) in before)
end

"""
    force_step_matches_oracle(state, params; t=dt) -> Bool

Takes one force step (the backend's `advance_positions!`, no diffusion) and checks that every
agent moved as `expected_displacements` says, to a tolerance relative to the largest move.
Mutates `state`.
"""
function force_step_matches_oracle(state, params; t=params.system.dt)
    before = positions_by_index(state)
    expected = expected_displacements(state, params)
    M.advance_positions!(state, params, t)
    observed = observed_displacements(before, state)
    scale = max(1e-12, maximum(norm, values(expected)))
    worst = maximum(norm(observed[i] - expected[i]) for i in keys(expected))
    return worst <= 1e-9 * scale
end

"""
    largest_free_disk(state, params; spacing) -> radius

Approximately the radius of the largest disk that overlaps no agent, by testing candidate
centres on a lattice with the given spacing (the true value is within `spacing / sqrt(2)` of
this).
"""
function largest_free_disk(state::M.SimState, params::M.AllParams; spacing::Float64)
    dims = state.grid_size.dims
    as = all_agents(state)
    best = 0.0
    for x in 0:spacing:dims[1], y in 0:spacing:dims[2]
        c = SVector(x, y)
        clearance = minimum(M.shortest_distance(c, a.position, dims) - current_radius(params, a) for a in as)
        best = max(best, clearance)
    end
    return best
end

"exact area of a disk"
disk_area(r) = π * r^2
