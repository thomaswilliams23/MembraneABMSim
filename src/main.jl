

"""
    run_sim(config_pathname::String)

Main driver function. Given the path of a config JSON file, parses the config, then runs a 
simulation according to the settings given in the config.
"""
function run_sim(config_pathname::String; clear_existing_output::Bool=false, suppress_prints::Bool=false)

    #parse the config and add a copy to the output directory
    params = parse_config(config_pathname)
    check_params(params)
    copy_config_to_output_dir(config_pathname, params.system.output_dir)

    #if specified, set the random seed
    if !isnothing(params.system.seed)
        if !suppress_prints
            println("Running with random seed $(params.system.seed)")
        end
        Random.seed!(params.system.seed)
    end

    #determine device to use
    device = "cpu"
    if params.system.device == "metal"
        @assert Metal.functional() "Metal device specified but no Metal-compatible GPU found."
        if !suppress_prints
            println("Running on Metal GPU")
        end
        device = "metal"
    elseif params.system.device == "cuda"
        @assert CUDA.has_cuda() "CUDA device specified but no CUDA-compatible GPU found."
        if !suppress_prints
            println("Running on CUDA GPU")
        end
        device = "cuda"
    elseif params.system.device == "cpu"
        if !suppress_prints
            println("Running on CPU")
        end
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
        ) = initialise_system_cpu(params; clear_existing_output=clear_existing_output, suppress_prints=suppress_prints)
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
        ) = initialise_system_metal(params; clear_existing_output=clear_existing_output, suppress_prints=suppress_prints)
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
            grid_size.num_agents_changed = false
        end

        #if any nascent agents have been promoted, need to update flat data structure and pass to gpu
        if grid_size.nascent_promoted

            update_flat_data_nascent_agents!(agents, system_flat_cpu)

            #if using metal, update nascent promotion on GPU
            if device=="metal"
                update_nascent_promotion_metal!(all_data_metal, system_flat_cpu, grid_size)
            elseif device=="cuda"
                update_nascent_promotion_CUDA!(all_data_CUDA, system_flat_cpu, grid_size)
            end

            grid_size.nascent_promoted = false
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
            compute_non_force_position_changes_metal!(
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

            if !suppress_prints
                @printf "Running: t=%5.2f\r" t
            end
        end
    end

    if !suppress_prints
        println("\nSimulation complete.")
    end
    return
end



"""
    run_sweep(sweep_config_pathname::String)

Wrapper around `run_sim` to run a parameter sweep as specified in a sweep config JSON file.
"""
function run_sweep(sweep_config_pathname::String; run_serially_yn::Bool=false)

    #helper for running one of the simulations in the sweep
    function _run_sim_in_sweep(sim_ix::Int, all_sims::Vector{Pair{String, Dict{String, Any}}}, def_params::AllParams)

        #get sim info
        (sim_path, sim_param_changes) = all_sims[sim_ix]
        
        #set up output directory for this sim
        out_path = joinpath(sweep_config.output_base_dir, sim_path)
        set_up_output_directory(out_path; suppress_prints=true)

        #make a config for each
        params_this_sim = make_config_this_sim(def_params, sim_param_changes, out_path)

        #save a copy of the config
        config_fname = joinpath("out", out_path, "config.json")
        open(config_fname, "w") do f
            JSON3.pretty(f, JSON3.write(params_this_sim))
        end

        #run the sim
        run_sim(config_fname; suppress_prints=true)
    end

    #helper for printing progress
    function _print_progress!(completed_sims::Vector{Int}, sim_ix::Int, num_sims::Int)
        completed_sims[sim_ix] = 1
        num_completed = sum(completed_sims)
        print("Completed $num_completed of $num_sims simulations\r")
    end


    #parse sweep config (file_io.jl)
    sweep_config = parse_sweep_config(sweep_config_pathname)

    #build a list of simulations to do with parameter information for each
    sim_dict = build_sim_dict(sweep_config)

    #load in the default parameters
    def_params = parse_config(sweep_config.default_config)

    #iterate through changes to be made as specified in the sweep config (and also reps)
    all_sims = collect(sim_dict)
    completed_sims = zeros(length(all_sims))

    #decide if we can parallelise based on device type
    if def_params.system.device == "cuda" || def_params.system.device == "metal"
        println("Running sweep serially due to GPU device usage")
        run_serially_yn = true
    end

    #run all the simulations
    completed_sims = zeros(length(all_sims))
    if run_serially_yn
        for sim_ix in eachindex(all_sims)
            _run_sim_in_sweep(sim_ix, all_sims, def_params)
            _print_progress!(completed_sims, sim_ix, length(all_sims))
        end
    else
        Threads.@threads for sim_ix in eachindex(all_sims)
            _run_sim_in_sweep(sim_ix, all_sims, def_params)
            _print_progress!(completed_sims, sim_ix, length(all_sims))
        end
    end

end
