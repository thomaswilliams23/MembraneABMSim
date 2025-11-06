
# essentially, implements GPU versions of the `AllAgentsFlat` struct (plus a few extra fields we need for force calculation)

#immutable struct, lives on CPU, holds params the GPU needs for force calculations
struct ParamsMetal
    dt::Float32
    OmpA_radius::Float32
    OmpCF_radius::Float32
    BamA_radius::Float32
    LptD_radius::Float32
    LPS_radius::Float32
    OmpA_tether_radius::Float32
    LptD_tether_radius::Float32
    insertion_mu_attr::Float32
    insertion_mu_rep::Float32
    insertion_k_C::Float32
    sensing_radius::Float32
    mu_attr_LPS_LPS::Float32
    mu_attr_OMP_LPS::Float32
    mu_attr_OMP_OMP::Float32
    mu_rep::Float32
    max_repulsion::Float32
    rho::Float32
    k_C::Float32
    eta::Float32
end

#mutable struct, lives on CPU, holds scalar and small fixed-size vector data from the SimGrid
#this stuff is small and changes every timestep, so lives on CPU
mutable struct GridSizeMetal
    dims::SVector{2, Float32}
    num_cells::SVector{2, Int}
    tot_num_cells::Int
    num_agents::Int
    has_changed::Bool
end



#mutable struct, lives on GPU, gathers all vector data the GPU needs for force calculation
mutable struct AllDataMetal
    #these fields from AllAgentsFlat
    positions::MtlVector{Float32, Metal.PrivateStorage}
    next_positions::MtlVector{Float32, Metal.PrivateStorage}
    effective_radii::MtlVector{Float32, Metal.PrivateStorage}
    identifiers::MtlVector{Int, Metal.PrivateStorage}
    tether_points::MtlVector{Float32, Metal.PrivateStorage}
    nascent_to_inserting_ixs::MtlVector{Int, Metal.PrivateStorage}
    nascent_to_substrate_ixs::MtlVector{Int, Metal.PrivateStorage}
    substrate_inserting_ideal_dists::MtlVector{Float32, Metal.PrivateStorage}

    #these fields from SimGrid
    coords_to_z_ix::MtlVector{Int, Metal.PrivateStorage}
    z_ix_to_coords::MtlVector{Int, Metal.PrivateStorage}
    agent_cell_z_ixs::MtlVector{Int, Metal.PrivateStorage}
    num_agents_in_cell::MtlVector{Int, Metal.PrivateStorage}
    start_agents_in_cell::MtlVector{Int, Metal.PrivateStorage}
end

