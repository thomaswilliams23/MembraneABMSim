"""
    shortest_vec_CUDA(pos1::SVector{2, Float32}, pos2::SVector{2, Float32}, dims::SVector{2, Float32}) :: SVector{2, Float32}

Returns the shortest vector between two positions, accounting for periodic boundaries.
"""
@inline function shortest_vec_CUDA(pos1::SVector{2, Float32}, pos2::SVector{2, Float32}, dims::SVector{2, Float32}) :: SVector{2, Float32}
    raw_vec = pos2 - pos1
    # Replace broadcast operations with component-wise
    wrapped_vec = SVector{2, Float32}(
        raw_vec[1] - round(raw_vec[1] / dims[1]) * dims[1],
        raw_vec[2] - round(raw_vec[2] / dims[2]) * dims[2]
    )
    return wrapped_vec
end


"""
    shortest_distance_CUDA(pos1::SVector{2, Float32}, pos2::SVector{2, Float32}, dims::SVector{2, Float32}) :: Float32

Returns the shortest distance between two positions, accounting for periodic boundaries.
"""
@inline function shortest_distance_CUDA(pos1::SVector{2, Float32}, pos2::SVector{2, Float32}, dims::SVector{2, Float32}) :: Float32
    vec = shortest_vec_CUDA(pos1, pos2, dims)
    return sqrt(vec[1]*vec[1] + vec[2]*vec[2])
end


"""
    fast_floor_int32(x::Float32):: Int32

Efficiently computes the floor of a Float32 value and returns it as Int32. GPU-safe.
"""
@inline function fast_floor_int32(x::Float32):: Int32
    i = unsafe_trunc(Int32, x)        # GPU-safe, direct LLVM fptosi
    i -= (x < Float32(i))             # subtract 1 if x < i (emulates floor)
    return i
end



"""
    parse_identifier_CUDA(identifier::Int)

Parses an agent identifier integer into its constituent properties for CUDA GPU kernels.
Returns a SVector of agent properties.
"""
@inline function parse_identifier_CUDA(identifier::Int)

    is_tethered = identifier < 0
    if is_tethered
        identifier *= -1
    end
    is_nascent = identifier > 1000
    if is_nascent
        is_inserting = false
        identifier -= 1000
    else
        is_inserting = identifier > 10
    end
    nascent_ix = fast_floor_int32(identifier / 10.0f0)
    agent_type_num = identifier - nascent_ix * 10

    agent_data = SVector{5, Int}(is_tethered, agent_type_num, is_nascent, is_inserting, nascent_ix)

    return agent_data
end