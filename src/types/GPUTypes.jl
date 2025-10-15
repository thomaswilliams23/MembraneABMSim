
# essentially, implements GPU versions of the `AllAgentsFlat` struct

mutable struct AllAgentsFlatMetal
    positions::MtlVector{Float32, Metal.PrivateStorage}
    next_positions::MtlVector{Float32, Metal.PrivateStorage}
    effective_radii::MtlVector{Float32, Metal.PrivateStorage}
    actual_radii::MtlVector{Float32, Metal.PrivateStorage}
    is_OMP::MtlVector{Bool, Metal.PrivateStorage}
    is_tethered::MtlVector{Bool, Metal.PrivateStorage}
    tether_points::MtlVector{Float32, Metal.PrivateStorage}
    tether_lengths::MtlVector{Float32, Metal.PrivateStorage}
    substrate_inserting_ixs::MtlVector{Int, Metal.PrivateStorage}
    substrate_inserting_ideal_dists::MtlVector{Float32, Metal.PrivateStorage}
    successors::MtlVector{Int, Metal.PrivateStorage}
    agent_ix_to_sorted_ix::MtlVector{Int, Metal.PrivateStorage}
    sorted_ix_to_agent_ix::MtlVector{Int, Metal.PrivateStorage}
end

