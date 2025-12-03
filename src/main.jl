

"""
    run_sim(config_pathname::String)

Main driver function. Given the path of a config JSON file, parses the config, then runs a 
simulation according to the settings given in the config.
"""
function run_sim(config_pathname::String; clear_existing_output::Bool=false)

    #parse the config and add a copy to the output directory
    params = parse_config(config_pathname)
    check_params(params)
    copy_config_to_output_dir(config_pathname, params.system.output_dir)

    #if specified, set the random seed
    if !isnothing(params.system.seed)
        println("Running with random seed $(params.system.seed)")
        Random.seed!(params.system.seed)
    end

    #determine device to use
    device = "cpu"
    if params.system.device == "metal"
        @assert Metal.functional() "Metal device specified but no Metal-compatible GPU found."
        println("Running on Metal GPU")
        device = "metal"
    elseif params.system.device == "cuda"
        @assert CUDA.has_cuda() "CUDA device specified but no CUDA-compatible GPU found."
        println("Running on CUDA GPU")
        device = "cuda"
    elseif params.system.device == "cpu"
        println("Running on CPU")
        device = "cpu"
    else
        if !isnothing(params.system.device)
            error("Device type $(params.system.device) not recognised")
        end
    end

    #build nascent lookups for speed
    nascent_added_area_lookup = build_nascent_added_area_lookup(params)
    effective_rad_incs, ideal_dist_incs = compute_nascent_incs(params)


    #intialise the system
    if device=="cpu"
        (
            agents, 
            grid_size, 
            grid, 
            system_flat_cpu
        ) = initialise_system_cpu(params; clear_existing_output=clear_existing_output)
    elseif device=="metal"
        (
            non_force_position_kernel,
            force_kernel, 
            agents, 
            grid_size, 
            grid, 
            system_flat_cpu, 
            all_data_metal, 
            grid_size_metal, 
            params_metal
        ) = initialise_system_metal(params; clear_existing_output=clear_existing_output)
        effective_rad_incs_metal = Float32.(effective_rad_incs)
        ideal_dist_incs_metal = Float32.(ideal_dist_incs)
    elseif device=="cuda"
        (
            non_force_position_kernel,
            force_kernel, 
            agents, 
            grid_size, 
            grid, 
            system_flat_cpu, 
            all_data_CUDA, 
            grid_size_CUDA, 
            params_CUDA
        ) = initialise_system_CUDA(params; clear_existing_output=clear_existing_output)
        effective_rad_incs_CUDA = Float32.(effective_rad_incs)
        ideal_dist_incs_CUDA = Float32.(ideal_dist_incs)
    end


    #write out initial state
    output_ix = 0
    write_system_state(agents, grid_size.dims, params.system.output_dir, output_ix)


    #main loop
    steps_since_grid_sync = 0
    MAX_STEPS_BETWEEN_GRID_SYNC = 100 #temporary
    max_time_ix = round(Int, params.system.t_max/params.system.dt)
    output_ix_interval = round(Int, params.system.vis_dt/params.system.dt)
    for time_ix in 0:max_time_ix

        t = time_ix * params.system.dt

        #update BAM subsystem
        update_BAM_subsystem!(agents, grid_size, grid, system_flat_cpu, params, t)

        #update Lpt subsystem
        update_Lpt_subsystem!(agents, grid_size, system_flat_cpu, params, t)

        #if we added new agents, need to update grid and flat data structure
        if grid_size.num_agents_changed

            #if using metal, copy data back to CPU to rebuild grid
            if device=="metal"
                copy_data_to_cpu_from_metal!(system_flat_cpu, agents, all_data_metal, grid_size_metal)
            elseif device=="cuda"
                copy_data_to_cpu_from_CUDA!(system_flat_cpu, agents, all_data_CUDA, grid_size_CUDA)
            end

            #rebuild grid and flat data structures
            rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
            compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
            put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)

            #if using metal, copy data to GPU
            if device=="metal"
                copy_data_to_metal!(all_data_metal, grid_size_metal, system_flat_cpu, grid_size, grid)
            elseif device=="cuda"
                copy_data_to_CUDA!(all_data_CUDA, grid_size_CUDA, system_flat_cpu, grid_size, grid)
            end

            steps_since_grid_sync = 0
        end

        #if any nascent agents have been promoted, need to update flat data structure on gpu
        if device=="metal" && grid_size.nascent_promoted
            update_nascent_promotion_metal!(all_data_metal, system_flat_cpu, grid_size)
        end

        #check for any new tethering or assembly
        newly_tethered_agent_ixs = update_tethering_and_assembly!(agents, system_flat_cpu, params)

        #if using metal, update newly tethered agent ixs on GPU
        if device=="metal" && length(newly_tethered_agent_ixs)>0
            update_newly_tethered_agent_ixs_metal!(all_data_metal, newly_tethered_agent_ixs)
        elseif device=="cuda" && length(newly_tethered_agent_ixs)>0
            update_newly_tethered_agent_ixs_CUDA!(all_data_CUDA, newly_tethered_agent_ixs)
        end

        #update any nascent agents
        update_nascent_agents!(agents, system_flat_cpu, effective_rad_incs, ideal_dist_incs)

        #rescale the domain and all agent positions
        if device=="cpu"
            rescale_domain!(agents, system_flat_cpu, grid_size, params, nascent_added_area_lookup, t)
        elseif device=="metal"
            compute_non_force_position_changes!(
                non_force_position_kernel, 
                agents, 
                all_data_metal, 
                grid_size, 
                grid_size_metal, 
                effective_rad_incs_metal, 
                ideal_dist_incs_metal, 
                nascent_added_area_lookup, 
                newly_tethered_agent_ixs, 
                params, 
                t
            )
        elseif device=="cuda"
            compute_non_force_position_changes_CUDA!(
                non_force_position_kernel, 
                agents, 
                all_data_CUDA, 
                grid_size, 
                grid_size_CUDA, 
                effective_rad_incs_CUDA, 
                ideal_dist_incs_CUDA, 
                nascent_added_area_lookup, 
                newly_tethered_agent_ixs, 
                params, 
                t
            )
        end


        #if it has been too long since last grid sync, rebuild grid and flat data structures
        if steps_since_grid_sync >= MAX_STEPS_BETWEEN_GRID_SYNC

            #if using metal, copy data back to CPU to rebuild grid
            if device=="metal"
                copy_data_to_cpu_from_metal!(system_flat_cpu, agents, all_data_metal, grid_size_metal)
            elseif device=="cuda"
                copy_data_to_cpu_from_CUDA!(system_flat_cpu, agents, all_data_CUDA, grid_size_CUDA)
            end

            #rebuild grid and flat data structures
            rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
            compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
            put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)

            #if using metal, copy data to GPU
            if device=="metal"
                copy_data_to_metal!(all_data_metal, grid_size_metal, system_flat_cpu, grid_size, grid)
            elseif device=="cuda"
                copy_data_to_CUDA!(all_data_CUDA, grid_size_CUDA, system_flat_cpu, grid_size, grid)
            end

            steps_since_grid_sync = 0
        else
            steps_since_grid_sync += 1
        end


        #resolve inter-agent forces and diffusion
        if device=="cpu"
            compute_diffusion!(agents, system_flat_cpu, grid_size, params)
            resolve_forces_cpu!(agents, grid_size, grid, system_flat_cpu, params)
        elseif device=="metal"
            resolve_forces_metal!(force_kernel, all_data_metal, grid_size_metal, params_metal)
            copy_data_to_cpu = copy_data_to_cpu_yn(agents.OMP.BamA, params, t)
            if copy_data_to_cpu
                copy_data_to_cpu_from_metal!(system_flat_cpu, agents, all_data_metal, grid_size_metal)
            end
        elseif device=="cuda"
            resolve_forces_CUDA!(force_kernel, all_data_CUDA, grid_size_CUDA, params_CUDA)
            copy_data_to_cpu = copy_data_to_cpu_yn(agents.OMP.BamA, params, t)
            if copy_data_to_cpu
                copy_data_to_cpu_from_CUDA!(system_flat_cpu, agents, all_data_CUDA, grid_size_CUDA)
            end
        end

        #optionally write out data
        if mod(time_ix, output_ix_interval) == 0
            output_ix = round(Int, time_ix/output_ix_interval)
            write_system_state(agents, grid_size.dims, params.system.output_dir, output_ix)

            @printf "Running: t=%5.2f\r" t
        end
    end

    println("\nSimulation complete.")
    return
end



"""
    run_sweep(sweep_config_pathname::String)

Wrapper around `run_sim` to run a parameter sweep as specified in a sweep config JSON file.
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
