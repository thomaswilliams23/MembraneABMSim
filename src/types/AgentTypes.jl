"""
    AbstractAgent

Abstract base type for all agent types in the simulation.
"""
abstract type AbstractAgent end

"""
    AbstractOMP <: AbstractAgent

Abstract type for all OMP (outer membrane protein) agents.
"""
abstract type AbstractOMP <: AbstractAgent end

"""
    AbstractNascentAgent <: AbstractAgent

Abstract type for all nascent agents.
"""
abstract type AbstractNascentAgent <: AbstractAgent end

"""
    LPSAgent <: AbstractAgent

Represents a lipopolysaccharide (LPS) agent with index, position, and arrival time.
"""
mutable struct LPSAgent <: AbstractAgent
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
end

"""
    OmpAAgent <: AbstractOMP

Represents an OmpA protein agent, including tethering information.
"""
mutable struct OmpAAgent <: AbstractOMP
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    is_tethered::Bool
    tether_point::SVector{2, Float64}
end

"""
    OmpCFAgent <: AbstractOMP

Represents an OmpCF protein agent.
"""
mutable struct OmpCFAgent <: AbstractOMP
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
end

"""
    LptDAgent <: AbstractOMP

Represents an LptD protein agent, including tethering and complex assembly state.
"""
mutable struct LptDAgent <: AbstractOMP
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    is_tethered::Bool
    tether_point::SVector{2, Float64}
    is_part_of_assembled_complex::Bool
    insertion_state::String
end

"""
    BamAAgent <: AbstractOMP

Represents a BamA protein agent, including complex assembly state.
"""
mutable struct BamAAgent <: AbstractOMP
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    is_part_of_assembled_complex::Bool
    insertion_state::String
end

"""
    NascentLPSAgent <: AbstractNascentAgent

Represents a nascent LPS agent, including ideal distance and associated inserting agent.
"""
mutable struct NascentLPSAgent <: AbstractNascentAgent
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    ideal_dist_from_inserting_agent::Float64
    inserting_agent_index::Int
end

"""
    NascentOMPAgent <: AbstractNascentAgent

Represents a nascent OMP agent, including effective radius, ideal distance, and OMP type.
"""
mutable struct NascentOMPAgent <: AbstractNascentAgent
    index::Int
    position::SVector{2, Float64}
    arrival_time::Float64
    effective_radius::Float64
    ideal_dist_from_inserting_agent::Float64
    OMP_type::String
    inserting_agent_index::Int
end

"""
    AllOMPs

Container for all OMP agents, grouped by type.
"""
mutable struct AllOMPs
    OmpA::Vector{OmpAAgent}
    OmpCF::Vector{OmpCFAgent}
    LptD::Vector{LptDAgent}
    BamA::Vector{BamAAgent}
end

"""
    AllNascent

Container for all nascent agents, grouped by type.
"""
mutable struct AllNascent
    nascent_OMP::Vector{NascentOMPAgent}
    nascent_LPS::Vector{NascentLPSAgent}
end

"""
    AllAgents

Main container for all agents in the simulation, including OMPs, LPS, nascent agents, and polypeptide count.
"""
mutable struct AllAgents
    OMP::AllOMPs
    LPS::Vector{LPSAgent}
    nascent::AllNascent
    num_PP::Int
end