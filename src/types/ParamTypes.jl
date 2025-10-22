

#simulation-related parameters
struct OmpAParams
    radius::Float64
    insertion_prob::Float64
    tether_rate::Float64
    tether_radius::Float64
end

struct OmpCFParams
    radius::Float64
    insertion_prob::Float64
end

struct LptDParams
    radius::Float64
    insertion_prob::Float64
    complex_assembly_rate::Float64
    tether_radius::Float64
end

struct BamAParams
    radius::Float64
    insertion_prob::Float64
    complex_assembly_rate::Float64
end

struct LPSParams
    radius::Float64
    arrival_rate::Float64
    insertion_time::Float64
end

struct InitConditions
    dim_x::Float64
    dim_y::Float64
    num_OmpA::Int
    num_OmpCF::Int
    num_LptD::Int
    num_BamA::Int
    num_LPS::Int
    num_PP::Int
    method::String
    equilibration_time::Union{Nothing, Float64}
    complexes_assembled::Union{Nothing, Bool}
end

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
end


struct SystemParams
    density::Float64
    dt::Float64
    num_sub_timesteps::Int
    t_max::Float64
    vis_dt::Float64
    output_dir::String
    seed::Union{Nothing, Int}
    device::Union{Nothing, String}
end


#container
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



#sweep-related structures
struct SweepParamValues
    short_name::Union{Nothing, String}
    values::Vector{Any}
end


struct SweepParams
    sweep_params::Dict{String, SweepParamValues}
    num_reps::Int
    default_config::String
    output_base_dir::String
end


#register all the struct types with StructTypes
StructTypes.StructType(::Type{OmpAParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{OmpCFParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{LptDParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{BamAParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{LPSParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{InitConditions}) = StructTypes.Struct()
StructTypes.StructType(::Type{InsertionParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{SystemParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{ForceParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{AllParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{SweepParamValues}) = StructTypes.Struct()
StructTypes.StructType(::Type{SweepParams}) = StructTypes.Struct()