
"""
    shortest_vec(P1::SVector{2, Float64}, P2::SVector{2, Float64}, dims::SVector{2, Float64})

Simple helper function to calculate the vector for the shortest path between P1 and P2 on a 
2D periodic domain with dimensions `dims`.
"""
@inline function shortest_vec(P1::SVector{2, Float64}, P2::SVector{2, Float64}, dims::SVector{2, Float64})

    shortest_vec = P2-P1
    shortest_vec -= dims.* round.(shortest_vec./dims)

    return shortest_vec
end


"""
    shortest_distance(P1::SVector{2, Float64}, P2::SVector{2, Float64}, dims::SVector{2, Float64})

Simple helper function to calculate the shortest distance between P1 and P2 on a 2D 
periodic domain with dimensions dim_x x dim_y.
"""
@inline function shortest_distance(P1::SVector{2, Float64}, P2::SVector{2, Float64}, dims::SVector{2, Float64})
    return norm(shortest_vec(P1, P2, dims))
end


"""
    build_dense_morton(num_cells::SVector{2, Int})

Precompute dense Morton mappings for a grid.
Returns two vectors:
- coords_to_z_ix[(y-1)*cols + x] = (dense) Morton index
- z_ix_to_coords[(2*dense-1):(2*dense)] = x, y coords
"""
function build_dense_morton(num_cells::SVector{2, Int})

    #unpack
    rows = num_cells[1]
    cols = num_cells[2]
    tot_cells = rows*cols

    #generate coords, sort by (actual) Morton
    coords = [(x,y) for y in 1:rows for x in 1:cols]
    sort!(coords, by = c -> cartesian2morton([c[1], c[2]]))

    #initialise output vectors
    coords_to_z_ix = Vector{Int}(undef, tot_cells)
    z_ix_to_coords = Vector{Int}(undef, 2*tot_cells)

    #build mappings
    for (dense, (x,y)) in enumerate(coords)
        coords_to_z_ix[(y-1)*cols + x] = dense
        z_ix_to_coords[2*dense-1] = x
        z_ix_to_coords[2*dense] = y
    end

    return coords_to_z_ix, z_ix_to_coords
end



"""
    exposed_area(rad_ins::Float64, rad_sub::Float64, dist::Float64)

Computes the exposed area of a substrate agent with radius `rad_sub`, at a distance `dist`
from an inserting agent with radius `rad_ins`. Exposed area is the total area of the substrate 
agent minus segments of both the inserting and substrate agents, cut by the chord connecting 
the points of intersection.
"""
function exposed_area(rad_ins::Float64, rad_sub::Float64, dist::Float64)

    dist_err = 1e-6
    if dist<dist_err
        return 0.0
    end

    x_int = (rad_ins^2 - rad_sub^2 + dist^2)/(2*dist)
    ins_sect_angle = 2*acos(clamp(x_int/rad_ins, -1.0, 1.0))
    sub_sect_angle = 2*acos(clamp((dist-x_int)/rad_sub, -1.0, 1.0))
    ins_sect_area = 0.5 * rad_ins^2 * (ins_sect_angle - sin(ins_sect_angle))
    sub_sect_area = 0.5 * rad_sub^2 * (sub_sect_angle - sin(sub_sect_angle))
    sub_full_area = pi * rad_sub^2

    return sub_full_area - ins_sect_area - sub_sect_area

end


"""
    change_field(obj, field::String, val)

Changes a field in a struct `obj` to a new value `val`, returning a new struct instance. (Workaround for changing fields in an immutable struct.)
"""
function change_field(obj, field::String, val)

    #read this into a named tuples structure
    nt = (; zip(fieldnames(typeof(obj)), getfield.(Ref(obj), fieldnames(typeof(obj))))...)

    #make sure the requested field is actually there
    if !haskey(nt, Symbol(field))
        error("Object $(typeof(obj)) does not have a field $(field)")
    end

    #update the named tuples
    updated_nt = merge(nt, [Pair(Symbol(field), val)])

    #rebuild
    return typeof(obj)(updated_nt...)
end



"""
    make_identifier(;is_tethered::Bool, agent_type::String, is_inserting::Bool, is_nascent::Bool, nascent_ix::Int)

Simple function to generate an compact identifier for an agent, encoding its properties
(this is a little bit janky but is memory efficient and fast).
"""
@inline function make_identifier(;is_tethered::Bool, agent_type::String, is_inserting::Bool, is_nascent::Bool, nascent_ix::Int)

    @assert nascent_ix<990 "There are too many nascent agents being inserted at once ($nascent_ix), cannot encode in identifier!"

    agent_type_to_num = Dict(
        "OmpA" => 1,
        "OmpCF" => 3,
        "BamA" => 5,
        "LptD" => 7,
        "LPS" => 2
    )

    identifier = agent_type_to_num[agent_type]
    if is_tethered
        identifier *= -1
    end

    if is_inserting
        identifier += 10*nascent_ix
    end

    if is_nascent
        identifier += 10*nascent_ix
        identifier += 1000
    end


    #BUGFIX
    if abs(identifier)>1e6
        error("Generated junk identifier: $identifier")
    end


    return identifier
end



"""
    parse_identifier(identifier::Int)

Recovers agent properties from compact identifier.
"""
@inline function parse_identifier(identifier::Int)
    num_to_agent_type = Dict(
        1 => "OmpA",
        3 => "OmpCF",
        5 => "BamA",
        7 => "LptD",
        2 => "LPS"
    )
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
    nascent_ix = floor(Int, identifier / 10)
    identifier -= nascent_ix * 10


    #BUGFIX
    if identifier ∉ keys(num_to_agent_type)
        return false, "Unknown", true, true, -1
    end


    agent_type = num_to_agent_type[identifier]
    return is_tethered, agent_type, is_nascent, is_inserting, nascent_ix
end