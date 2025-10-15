

"""
    compute_overhang(position::Vector{Float64}, radius::Float64, dims::SVector{2, Float64}) :: Vector{Int}

Given a position in the membrane, a radius, and the dimensions of the membrane, calculates
which, if any, of the edges the object hangs over. Relies on the assumption that an object
cannot overlap both the top and bottom edge, or left and right edge simultaneously.

Output is a two element vector `overhang` such that 
    `overhang[1]=-1` means the object lies over the left edge
    `overhang[1]=1` means the objects lies over the right edge
    `overhang[2]=-1` means the object lies over the bottom edge
    `overhang[2]=1` means the objects lies over the top edge
"""
function compute_overhang(position::Vector{Float64}, radius::Float64, dims::SVector{2, Float64}) :: Vector{Int}
    overhang_l = (position[1] < radius)
    overhang_r = ((dims[1]-position[1]) < radius)
    overhang_b = (position[2] < radius)
    overhang_t = ((dims[2]-position[2]) < radius)

    return [-overhang_l + overhang_r, -overhang_b + overhang_t]
end


"""
    build_plot_objects(agents::AllAgents, dims::SVector{2, Float64}, max_dims::SVector{2, Float64}, params::AllParams)

Given a system state (specifically, an AllAgents structure, `agents`, and a grid dimensions 
vector `dims`), extracts and returns vectors of data necessary for plotting, centered within
the maximal membrane domain
(
    membrane_obj,           # a Point2f vector comprising the current vertices of the membrane (centered)
    agent_x_coords,         # x coordinates of each agent
    agent_y_coords,         # y coordinates of each agent
    agent_radii,            # (effective) radii of each agent
    agent_colours,          # `RGBA{Float64}` colour of each agent
    insertion_states,       # insertion states of each agent as string ("NA" if not an inserting-type agent)
    tether_x_coords,        # x coordinates of the tethers of each tethered agent
    tether_y_coords,        # y coordinates of the tethers of each tethered agent
    tether_end_x_coords,    # x coordinates of each tethered agent (for plotting tethers)
    tether_end_y_coords.    # y coordinates of each tethered agent (for plotting tethers)
)
"""
function build_plot_objects(agents::AllAgents, dims::SVector{2, Float64}, max_dims::SVector{2, Float64}, params::AllParams)

    #colour defaults
    non_nascent_alpha = 0.7
    nascent_alpha = 1.0
    OmpA_colour = RGBA(0.4, 0.76, 0.65, non_nascent_alpha)
    OmpCF_colour = RGBA(0.22, 0.49, 0.72, non_nascent_alpha)
    BamA_colour = RGBA(0.89, 0.10, 0.11, non_nascent_alpha)
    LptD_colour = RGBA(0.30, 0.69, 0.29, non_nascent_alpha)
    LPS_colour = RGBA(0.49, 0.49, 0.49, non_nascent_alpha)
    nascent_OmpA_colour = RGBA(0.4, 0.76, 0.65, nascent_alpha)
    nascent_OmpCF_colour = RGBA(0.22, 0.49, 0.72, nascent_alpha)
    nascent_BamA_colour = RGBA(0.89, 0.10, 0.11, nascent_alpha)
    nascent_LptD_colour = RGBA(0.30, 0.69, 0.29, nascent_alpha)
    nascent_LPS_colour = RGBA(0.49, 0.49, 0.49, nascent_alpha)

    #make vectors to store all agent representations
    agent_x_coords = Vector{Float64}()
    agent_y_coords = Vector{Float64}()
    agent_radii = Vector{Float64}()
    agent_colours = Vector{RGBA{Float64}}()
    insertion_states = Vector{String}()
    tether_x_coords = Vector{Float64}()
    tether_y_coords = Vector{Float64}()
    tether_end_x_coords = Vector{Float64}()
    tether_end_y_coords = Vector{Float64}()

    #convenience functions for writing data
    function _push_agent_data!(x_coord::Float64, y_coord::Float64, agent_radius::Float64, 
                               agent_colour::RGBA{Float64}, insertion_state::String)
        shifted_x_coord = x_coord + membrane_bl[1]
        shifted_y_coord = y_coord + membrane_bl[2]
        push!(agent_x_coords, shifted_x_coord)
        push!(agent_y_coords, shifted_y_coord)
        push!(agent_radii, agent_radius)
        push!(agent_colours, agent_colour)
        push!(insertion_states, insertion_state)
    end

    function _push_tether_data!(tether_x_coord::Float64, tether_y_coord::Float64, 
                                tether_end_x_coord::Float64, tether_end_y_coord::Float64)
        shortest_path_to_tether = shortest_vec(
            SVector{2, Float64}(tether_end_x_coord, tether_end_y_coord), 
            SVector{2, Float64}(tether_x_coord, tether_y_coord), 
            dims
        )
        shifted_tether_x_coord = tether_end_x_coord + shortest_path_to_tether[1] + membrane_bl[1]
        shifted_tether_y_coord = tether_end_y_coord + shortest_path_to_tether[2] + membrane_bl[2]
        shifted_tether_end_x_coord = tether_end_x_coord + membrane_bl[1]
        shifted_tether_end_y_coord = tether_end_y_coord + membrane_bl[2]
        push!(tether_x_coords, shifted_tether_x_coord)
        push!(tether_y_coords, shifted_tether_y_coord)
        push!(tether_end_x_coords, shifted_tether_end_x_coord)
        push!(tether_end_y_coords, shifted_tether_end_y_coord)
    end

    function _push_overhang_agent_data!(overhang::Vector{Int}, x_coord::Float64, y_coord::Float64, agent_radius::Float64, 
                                        agent_colour::RGBA{Float64}, insertion_state::String)
        #count num overhangs
        num_overhangs = (overhang[1]!=0) + (overhang[2]!=0)

        #single overhang
        if num_overhangs == 1
            periodic_shifts = [overhang]
        #x and y overhangs, need THREE periodic images
        elseif num_overhangs == 2
            periodic_shifts = [[overhang[1], 0], [0, overhang[2]], overhang]
        else
            error("More than two overhangs - impossible.")
        end
        #for each image, generate a new agent position, radius, etc
        for periodic_shift in periodic_shifts
            img_position = [x_coord, y_coord] - periodic_shift .* dims
            _push_agent_data!(img_position[1], img_position[2], agent_radius, agent_colour, insertion_state)
        end
    end

    function _push_overhang_tether_data!(overhang::Vector{Int}, tether_x_coord::Float64, tether_y_coord::Float64, 
                                         tether_end_x_coord::Float64, tether_end_y_coord::Float64)
        #count number overhangs
        num_overhangs = (overhang[1]!=0) + (overhang[2]!=0)

        #single overhang
        if num_overhangs == 1
            periodic_shifts = [overhang]
        #x and y overhangs, need THREE periodic images
        elseif num_overhangs == 2
            periodic_shifts = [[overhang[1], 0], [0, overhang[2]], overhang]
        else
            error("More than two overhangs - impossible.")
        end
        #for each image, generate a new agent position, radius, etc
        for periodic_shift in periodic_shifts
            img_position = [tether_end_x_coord, tether_end_y_coord] - periodic_shift .* dims
            img_tether = [tether_x_coord, tether_y_coord] - periodic_shift .* dims
            _push_tether_data!(img_tether[1], img_tether[2], img_position[1], img_position[2])
        end
    end

    function _push_all_agent_data(agent::AbstractAgent, agent_radius::Float64, agent_colour::RGBA{Float64}, insertion_state::String, test_tether::Bool)
        
        #check for overhang on membrane edges
        overhang = compute_overhang(agent.position, agent_radius, dims)
        num_overhangs = (overhang[1]!=0) + (overhang[2]!=0)

        #push this agent's data
        x_coord, y_coord = agent.position
        _push_agent_data!(x_coord, y_coord, agent_radius, agent_colour, insertion_state)
        
        #tether
        agent_tethered = false
        if test_tether
            if agent.is_tethered
                tether_x_coord, tether_y_coord = agent.tether_point
                _push_tether_data!(tether_x_coord, tether_y_coord, x_coord, y_coord)
                agent_tethered = true
            end
        end

        #go through periodic images
        if num_overhangs>0
            _push_overhang_agent_data!(overhang, x_coord, y_coord, agent_radius, agent_colour, insertion_state)
            if agent_tethered
                _push_overhang_tether_data!(overhang, tether_x_coord, tether_y_coord, x_coord, y_coord)
            end
        end

    end


    #work out where the bottom left of the membrane goes 
    membrane_bl = [0.5*(max_dims[1]-dims[1]), 0.5*(max_dims[2]-dims[2])]

    #membrane object
    membrane_obj = Point2f[membrane_bl, membrane_bl + [dims[1], 0], membrane_bl + dims, membrane_bl + [0, dims[2]]]

    #loop through agents and populate vectors
    no_insertion_state = "NA"
    for OmpA in agents.OMP.OmpA
        _push_all_agent_data(OmpA, params.OmpA.radius, OmpA_colour, no_insertion_state, true)
    end
    for OmpCF in agents.OMP.OmpCF
        _push_all_agent_data(OmpCF, params.OmpCF.radius, OmpCF_colour, no_insertion_state, false)
    end
    for BamA in agents.OMP.BamA
        _push_all_agent_data(BamA, params.BamA.radius, BamA_colour, BamA.insertion_state, false)
    end
    for LptD in agents.OMP.LptD
        _push_all_agent_data(LptD, params.LptD.radius, LptD_colour, LptD.insertion_state, true)
    end
    for LPS in agents.LPS
        _push_all_agent_data(LPS, params.LPS.radius, LPS_colour, no_insertion_state, false)
    end
    for nascent_LPS in agents.nascent.nascent_LPS
        _push_all_agent_data(nascent_LPS, params.LPS.radius, nascent_LPS_colour, no_insertion_state, false)
    end
    for nascent_OMP in agents.nascent.nascent_OMP
        if nascent_OMP.OMP_type == "OmpA"
            _push_all_agent_data(nascent_OMP, nascent_OMP.effective_radius, nascent_OmpA_colour, no_insertion_state, false)
        elseif nascent_OMP.OMP_type == "OmpCF"
            _push_all_agent_data(nascent_OMP, nascent_OMP.effective_radius, nascent_OmpCF_colour, no_insertion_state, false)
        elseif nascent_OMP.OMP_type == "BamA"
            _push_all_agent_data(nascent_OMP, nascent_OMP.effective_radius, nascent_BamA_colour, no_insertion_state, false)
        elseif nascent_OMP.OMP_type == "LptD"
            _push_all_agent_data(nascent_OMP, nascent_OMP.effective_radius, nascent_LptD_colour, no_insertion_state, false)
        else
            error("Nascent OMP type $(nascent_OMP.OMP_type) not recognised.")
        end
    end
    
    return (
        membrane_obj,
        agent_x_coords,
        agent_y_coords,
        agent_radii,
        agent_colours,
        insertion_states,
        tether_x_coords,
        tether_y_coords,
        tether_end_x_coords,
        tether_end_y_coords
    )
end


"""
    make_membrane_movie(out_path::String)

Given an `out_path` - which must be a directory within the `out` directory - makes a movie
of the simulation data in `out_path/raw_data`.
"""
function make_membrane_movie(out_path::String)

    #check that the out_path exists and contains data
    raw_data_dir = joinpath("out", out_path, "raw_data")
    if !isdir(raw_data_dir)
        error("Can't find output directory $data_dir")
    end
    if isempty(readdir(raw_data_dir))
        error("Data directory exists, but is empty.")
    end

    #read in the copy of the config in the out_path
    config_path = joinpath("out", out_path, "config.json")
    params = parse_config(config_path)

    #find the maximum dimensions (final data file)
    num_time_steps = round(Int, params.system.t_max/params.system.vis_dt)
    final_data_fname = joinpath("out", out_path, "raw_data", "sys_data_$(num_time_steps).jld2")
    @load final_data_fname dims
    max_dims = dims

    #initialise figure
    fig = Figure()
    ax = Axis(fig[1,1];
        backgroundcolor = :transparent,
        xgridvisible = false,
        ygridvisible = false,
        xticksvisible = false,
        yticksvisible = false,
        aspect = DataAspect(),
        limits = (0, max_dims[1], 0, max_dims[2]))
    hidedecorations!(ax; grid=false)
    hidespines!(ax)

    #make a circle marker of unit size
    unit_circle = BezierPath([MoveTo(Point(1,0)), EllipticalArc(Point(0, 0), 1, 1, 0, 0, 2pi)])

    #set up movie
    movie_path = joinpath("out", out_path, "sim.mp4")

    #loop over each output data file and generate a snapshot for the movie
    record(fig, movie_path, 1:2988) do time_ix

        print("Rendering frame $time_ix of $num_time_steps\r")
        
        #load in data for this time step
        time_val = time_ix * params.system.vis_dt
        data_fname = joinpath("out", out_path, "raw_data", "sys_data_$(time_ix).jld2")
        @load data_fname agents dims
        
        #get agent representations for this time step
        (
            membrane_obj, 
            agent_x_coords, 
            agent_y_coords, 
            agent_radii, 
            agent_colours,
            insertion_states,
            tether_x_coords,
            tether_y_coords,
            tether_end_x_coords,
            tether_end_y_coords
        ) = build_plot_objects(agents, dims, max_dims, params)

        #clear current plot and plot representations for this time step
        empty!(ax)

        #plot membrane background
        poly!(ax, membrane_obj;
            color = RGBA(0.5, 0.5, 0.5, 0.2)
        )

        #agents
        scatter!(ax, agent_x_coords, agent_y_coords; 
            marker=unit_circle, 
            markersize = agent_radii, 
            markerspace=:data, 
            color = agent_colours
        )

        #plot outlines to indicate insertion states
        for (i, ins_state) in enumerate(insertion_states)
            if ins_state=="bound"
                agent_colour = agent_colours[i]
                poly!(ax, Circle(Point2f(agent_x_coords[i], agent_y_coords[i]), agent_radii[i]);
                    strokecolor = RGBA(agent_colour.r, agent_colour.g, agent_colour.b, 1.0),
                    linestyle = :dash,
                    strokewidth = 4,
                    color = :transparent
                )
            elseif ins_state=="embedding"
                agent_colour = agent_colours[i]
                poly!(ax, Circle(Point2f(agent_x_coords[i], agent_y_coords[i]), agent_radii[i]);
                    strokecolor = RGBA(agent_colour.r, agent_colour.g, agent_colour.b, 1.0),
                    strokewidth = 4,
                    color = :transparent
                )
            end
        end

        #tethers to plot?
        if length(tether_x_coords)>0
            
            #tether points
            scatter!(ax, tether_x_coords, tether_y_coords;
                marker = :xcross,
                markersize = 10,
                color = :red
            )

            #tethers
            for ix = 1:length(tether_x_coords)
                lines!(
                    [tether_x_coords[ix], tether_end_x_coords[ix]], [tether_y_coords[ix], tether_end_y_coords[ix]];
                    color = :black,
                    linestyle = (:dash, :dense),
                    linewidth = 1
                )
            end
        end

        #now plot a mask to cover everything outside the membrane
        membrane_mask = Polygon(
            Point2f[[0.0, 0.0], [max_dims[1], 0.0], max_dims, [0.0, max_dims[2]]],
            [membrane_obj]
        )
        poly!(ax, membrane_mask;
            color=:white
        )

        
    end

    println("\nDone!")
end

