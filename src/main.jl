

"""
    run_sim(config_pathname::String)

Main driver function. Given the path of a config JSON file, parses the config, then runs a 
simulation according to the settings given in the config.
"""
function run_sim(config_pathname::String)

    #parse the config and add a copy to the output directory
    params = parse_config(config_pathname)
    copy_config_to_output_dir(config_pathname, params.system.output_dir)

    #if specified, set the random seed
    if !isnothing(params.system.seed)
        println("Running with random seed $(params.system.seed)")
        Random.seed!(params.system.seed)
    end

    #determine device to use
    device = "cpu"
    if params.system.device == "metal"
        println("Running on Metal GPU")
        device = "metal"
    else
        if !isnothing(params.system.device)
            error("Device type $(params.system.device) not recognised")
        end
    end

    #build nascent added area lookup for speed
    nascent_added_area_lookup = build_nascent_added_area_lookup(params)

    #intialise the system
    t=0.0
    if device=="cpu"
        agents, grid, system_flat_cpu = initialise_system_cpu(params)
    elseif device=="metal"
        (
            force_kernel,
            agents, 
            grid, 
            system_flat_cpu, 
            all_data_metal, 
            grid_metal,
            params_metal
        ) = initialise_system_metal(params)
    end


    #write out initial state
    time_ix = 0
    write_system_state(agents, grid.dims, params.system.output_dir, time_ix)


    #main loop
    time_err = 0.1*params.system.dt
    while t<params.system.t_max - time_err

        #update BAM subsystem
        update_BAM_subsystem!(agents, grid, system_flat_cpu, params, t)

        #update Lpt subsystem
        update_Lpt_subsystem!(agents, grid, params, t)

        #check for any new tethering or assembly
        update_tethering_and_assembly!(agents, params)

        #update any nascent agents
        update_nascent_agents!(agents, params, t)

        #rescale the domain and every agent's position
        rescale_domain!(agents, grid, params, nascent_added_area_lookup, t)

        #compute diffusion of every agent
        compute_diffusion!(agents, grid, params)

        #update the grid and flat system data
        rebuild_grid!(grid, agents, params.force.sensing_radius)
        compile_flat_system_data_cpu!(system_flat_cpu, agents, grid, params)
        put_grid_in_sorted_order!(grid, system_flat_cpu)

        #resolve inter-agent forces
        if device=="cpu"
            resolve_forces_cpu!(agents, grid, system_flat_cpu, params)
        elseif device=="metal"
            copy_data_to_metal!(all_data_metal, grid_metal, system_flat_cpu, grid)
            resolve_forces_metal!(force_kernel, agents, system_flat_cpu, all_data_metal, grid_metal, params_metal)
        end

        #step time
        t += params.system.dt


        #optionally write out data
        if abs(t/params.system.vis_dt - round(t/params.system.vis_dt))<time_err
            time_ix = round(Int, t/params.system.vis_dt)
            write_system_state(agents, grid.dims, params.system.output_dir, time_ix)

            @printf "Running: t=%5.2f\r" t
        end
    end

    return
end



"""
run sweep
"""
function run_sweep(sweep_config_pathname::String)

    #parse sweep config (file_io.jl)
    sweep_config = parse_sweep_config(sweep_config_pathname)

    #build a list of simulations to do with parameter information for each
    sim_dict = build_sim_dict(sweep_config)

    #load in the default parameters
    def_params = parse_config(sweep_config.default_config)

    #iterate through changes to be made as specified in the sweep config (and also reps)
    for (sim_path, sim_param_changes) in sim_dict

        #set up output directory for this sim
        out_path = joinpath(sweep_config.output_base_dir, sim_path)
        set_up_output_directory(out_path)

        #make a config for each
        params_this_sim = make_config_this_sim(def_params, sim_param_changes, out_path)

        #save a copy of the config
        config_fname = joinpath("out", out_path, "config.json")
        open(config_fname, "w") do f
            JSON3.pretty(f, JSON3.write(params_this_sim))
        end

        #run the sim
        run_sim(config_fname)
    end

end