

#abstract types for hierarchy
abstract type AbstractAgent end

abstract type AbstractOMP <: AbstractAgent end
abstract type AbstractNascentAgent <: AbstractAgent end


#LPS
mutable struct LPSAgent <: AbstractAgent
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
end


#OMPs
mutable struct OmpAAgent <: AbstractOMP
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    is_tethered::Bool
    tether_point::SVector{2, Float64}
end

mutable struct OmpCFAgent <: AbstractOMP
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
end

mutable struct LptDAgent <: AbstractOMP
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    is_tethered::Bool
    tether_point::SVector{2, Float64}
    is_part_of_assembled_complex::Bool
    insertion_state::String
end

mutable struct BamAAgent <: AbstractOMP
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    is_part_of_assembled_complex::Bool
    insertion_state::String
end


#Nascent types
mutable struct NascentLPSAgent <: AbstractNascentAgent
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    ideal_dist_from_inserting_agent::Float64
    inserting_agent_index::Int
end

mutable struct NascentOMPAgent <: AbstractNascentAgent
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    effective_radius::Float64
    ideal_dist_from_inserting_agent::Float64
    OMP_type::String
    inserting_agent_index::Int
end


#containers for agents
mutable struct AllOMPs
    OmpA::Vector{OmpAAgent}
    OmpCF::Vector{OmpCFAgent}
    LptD::Vector{LptDAgent}
    BamA::Vector{BamAAgent}
end

mutable struct AllNascent
    nascent_OMP::Vector{NascentOMPAgent}
    nascent_LPS::Vector{NascentLPSAgent}
end


#main container used, comprising all agents (and also the number of polypeptides)
mutable struct AllAgents
    OMP::AllOMPs
    LPS::Vector{LPSAgent}
    nascent::AllNascent
    num_PP::Int
end