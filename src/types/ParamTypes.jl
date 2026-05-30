"""
    OmpAParams

Parameters for OmpA agents, including radius, insertion probability, tether rate, and tether radius.
"""
struct OmpAParams
    radius::Float64
    insertion_prob::Float64
    tether_rate::Float64
    tether_radius::Float64
end

"""
    OmpCFParams

Parameters for OmpCF agents, including radius and insertion probability.
"""
struct OmpCFParams
    radius::Float64
    insertion_prob::Float64
end

"""
    LptDParams

Parameters for LptD agents, including radius, insertion probability, complex assembly rate, and tether radius.
"""
struct LptDParams
    radius::Float64
    insertion_prob::Float64
    complex_assembly_rate::Float64
    tether_radius::Float64
end

"""
    BamAParams

Parameters for BamA agents, including radius, insertion probability, and complex assembly rate.
"""
struct BamAParams
    radius::Float64
    insertion_prob::Float64
    complex_assembly_rate::Float64
end

"""
    LPSParams

Parameters for LPS agents, including radius, arrival rate, and insertion time.
"""
struct LPSParams
    radius::Float64
    arrival_rate::Float64
    insertion_time::Float64
end


"""
    InitPositions

Initial positions (optional) for different agent types in the simulation.
"""
struct InitPositions
    OmpA::Union{Nothing, Vector{SVector{2, Float64}}}
    OmpCF::Union{Nothing, Vector{SVector{2, Float64}}}
    LptD::Union{Nothing, Vector{SVector{2, Float64}}}
    BamA::Union{Nothing, Vector{SVector{2, Float64}}}
    LPS::Union{Nothing, Vector{SVector{2, Float64}}}
end


"""
    InitConditions

Initial conditions for the simulation, including domain size, agent counts, method, and equilibration settings.
"""
struct InitConditions
    dim_x::Float64
    dim_y::Float64
    num_OmpA::Int
    num_OmpCF::Int
    num_LptD::Int
    num_BamA::Int
    num_LPS::Int
    num_PP::Int
    positions::Union{Nothing, InitPositions}
    method::String
    equilibration_time::Union{Nothing, Float64}
    complexes_assembled::Union{Nothing, Bool}
    checkpoint_file::Union{Nothing, String}
    checkpoint_time::Union{Nothing, Float64}
    shrink_to_size::Union{Nothing, Bool}
    shrinkage_factor::Union{Nothing, Float64}
    max_num_shrinkage_rounds::Union{Nothing, Int}
    shrinkage_equilibration_time::Union{Nothing, Float64}
end

"""
    InsertionParams

Parameters controlling agent insertion dynamics, including rates and force constants.
"""
struct InsertionParams
    type::String
    attempt_dt::Float64
    mu_rep::Float64
    mu_attr::Float64
    k_C::Float64
    PP_arrival_rate::Float64
    PP_bind_rate::Float64
    OMP_assembly_rate::Float64
end

"""
    ForceParams

Parameters for force calculations, including temperature, attraction/repulsion coefficients, and sensing radius.
"""
struct ForceParams
    temperature::Float64
    mu_rep::Float64
    mu_attr_OMP_OMP::Float64
    mu_attr_OMP_LPS::Float64
    mu_attr_LPS_LPS::Float64
    rho::Float64
    max_repulsion::Float64
    k_C::Float64
    eta::Float64
    sensing_radius::Float64
    diffusion_mode::Union{Nothing, String}
end

"""
    SystemParams

General simulation parameters, including density, timestep, output settings, and device configuration.
"""
struct SystemParams
    density::Float64
    dt::Float64
    t_max::Float64
    vis_dt::Float64
    output_dir::String
    max_hole_radius::Union{Nothing, Float64}
    seed::Union{Nothing, Int}
    device::Union{Nothing, String}
end

"""
    AllParams

Container struct holding all simulation parameters for agents, initial conditions, insertion, force, and system.
"""
struct AllParams
    OmpA::OmpAParams
    OmpCF::OmpCFParams
    LptD::LptDParams
    BamA::BamAParams
    LPS::LPSParams
    init::InitConditions
    insertion::InsertionParams
    force::ForceParams
    system::SystemParams
end



"""
    SweepParamValues

Holds values for a parameter sweep, including a short name and a vector of values.
"""
struct SweepParamValues
    short_name::Union{Nothing, String}
    values::Vector{Any}
end


"""
    SweepParams

Container for parameter sweep configuration, including sweep parameters, replicate number, and output directory.
"""
struct SweepParams
    sweep_params::Dict{String, SweepParamValues}
    num_reps::Int
    default_config::String
    output_base_dir::String
    checkpoint_sweep_config::Union{Nothing, String}
end


#register all the struct types with StructTypes
StructTypes.StructType(::Type{OmpAParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{OmpCFParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{LptDParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{BamAParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{LPSParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{InitPositions}) = StructTypes.Struct()
StructTypes.StructType(::Type{InitConditions}) = StructTypes.Struct()
StructTypes.StructType(::Type{InsertionParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{SystemParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{ForceParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{AllParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{SweepParamValues}) = StructTypes.Struct()
StructTypes.StructType(::Type{SweepParams}) = StructTypes.Struct()