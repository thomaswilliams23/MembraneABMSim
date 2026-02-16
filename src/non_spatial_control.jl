
"""
    run_quick_non_spatial_sim(config_pathname::String; clear_existing_output::Bool=false, suppress_prints::Bool=false)

Simulates a non-spatial membrane based on the provided config file. Notably this removes the requirement for local LPS
for BAM insertion. Note that this bypasses force calculation and spatial structure entirely and only simulates arrivals,
wait times etc. This is intented as a negative control to the spatial model.
"""
function run_quick_non_spatial_sim(config_pathname::String; clear_existing_output::Bool=false, suppress_prints::Bool=false)

    #parse the config and add a copy to the output directory
    params = parse_config(config_pathname)

    #adjust some parameters to avoid unnecessary calculations
    @reset params.insertion.type = "guaranteed"
    @reset params.init.method = "random"
    @reset params.init.equilibration_time = 0.0

    #just to make sure you don't accidentally overwrite a spatial sim, add a non-spatial prefix to the output direction
    @reset params.system.output_dir="NON_SPATIAL_" * params.system.output_dir

    #if specified, set the random seed
    if !isnothing(params.system.seed)
        if !suppress_prints
            println("Running with random seed $(params.system.seed)")
        end
        Random.seed!(params.system.seed)
    end

    #check device - only support CPU
    if !isnothing(params.system.device) && params.system.device != "cpu"
        error("Only CPU is supported currently for non-spatial simulation")
    end

    #set up output directory and add a copy of the config to the output directory
    set_up_output_directory(params.system.output_dir; clear_existing_output=clear_existing_output, suppress_prints=suppress_prints)
    copy_config_to_output_dir(config_pathname, params.system.output_dir)

    #build nascent lookups for speed
    nascent_added_area_lookup = build_nascent_added_area_lookup(params)
    effective_rad_incs, ideal_dist_incs = compute_nascent_incs(params)

    #initialise
    (
        agents, 
        grid_size, 
        grid, 
        system_flat_cpu
    ) = initialise_system_cpu(params; clear_existing_output=clear_existing_output, suppress_prints=suppress_prints)

    #write out the initial state    
    init_output_ix = 0
    write_system_state(agents, grid_size.dims, params.system.output_dir, init_output_ix)


    #main loop
    steps_since_grid_sync = 0
    MAX_STEPS_BETWEEN_GRID_SYNC = 100 #temporary
    output_ix_interval = round(Int, params.system.vis_dt/params.system.dt)
    max_time_ix = round(Int, params.system.t_max/params.system.dt)
    for time_ix in 1:max_time_ix

        t = time_ix * params.system.dt

        #update BAM subsystem
        update_BAM_subsystem!(agents, grid_size, grid, system_flat_cpu, params, t)

        #update Lpt subsystem
        update_Lpt_subsystem!(agents, grid_size, system_flat_cpu, params, t)

         #if we added new agents, need to update grid and flat data structure
        if grid_size.num_agents_changed

            #rebuild grid and flat data structures
            rebuild_grid!(grid_size, grid, agents, params.force.sensing_radius)
            compile_flat_system_data_cpu!(system_flat_cpu, agents, grid_size, grid, params)
            put_grid_in_sorted_order!(grid_size, grid, system_flat_cpu)

            steps_since_grid_sync = 0
            grid_size.num_agents_changed = false
        end

        #if any nascent agents have been promoted, need to update flat data structure
        if grid_size.nascent_promoted
            update_flat_data_nascent_agents!(agents, system_flat_cpu)
            grid_size.nascent_promoted = false
        end

        #check for any new tethering or assembly
        update_tethering_and_assembly!(agents, system_flat_cpu, params)

        #update any nascent agents
        update_nascent_agents!(agents, system_flat_cpu, effective_rad_incs, ideal_dist_incs)

        #rescale the domain (dims only)
        added_area_this_timestep = compute_added_area(agents, params, nascent_added_area_lookup, t)
        added_area_err = 1e-10
        if added_area_this_timestep>added_area_err
            prev_area = prod(grid_size.dims)
            scaled_added_area = added_area_this_timestep/params.system.density
            scale_factor = sqrt((prev_area + scaled_added_area)/prev_area)
            grid_size.dims *= scale_factor
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
    run_reps_quick_non_spatial_sim(config_pathname::String, nreps::Int)

Wrapper around `run_quick_non_spatial_sim` for running multiple replicates of a non-spatial simulation based on a simulation config file. This is intended for running multiple replicates of the non-spatial control simulation.
"""
function run_reps_quick_non_spatial_sim(config_pathname::String, nreps::Int)
    
    #helper for running one of the simulations with a unique seed
    function _run_quick_non_spatial_sim_rep(def_params::AllParams, sim_ix::Int)

        #get sim info
        sim_path = "rep$(sim_ix)"
        sim_param_changes = Dict{String, Any}("system/seed" => sim_ix)
        
        #set up output directory for this sim
        base_dir = "multi_run_" * def_params.system.output_dir
        raw_out_path = joinpath(base_dir, sim_path)
        actual_out_path = joinpath("NON_SPATIAL_" * base_dir, sim_path)
        set_up_output_directory(actual_out_path; suppress_prints=true)

        #make a config for each
        params_this_sim = make_config_this_sim(def_params, sim_param_changes, raw_out_path)

        #save a copy of the config
        config_fname = joinpath("out", actual_out_path, "config.json")
        open(config_fname, "w") do f
            JSON3.pretty(f, JSON3.write(params_this_sim))
        end

        #run the sim
        run_quick_non_spatial_sim(config_fname; suppress_prints=true)
    end

    #helper for printing progress
    function _print_progress!(completed_sims::Vector{Int}, sim_ix::Int, num_sims::Int)
        completed_sims[sim_ix] = 1
        num_completed = sum(completed_sims)
        print("Completed $num_completed of $num_sims simulations\r")
    end

    #load in the default parameters
    def_params = parse_config(config_pathname)

    #run all the simulations
    completed_sims = zeros(Int, nreps)
    Threads.@threads for sim_ix in eachindex(completed_sims)
        _run_quick_non_spatial_sim_rep(def_params, sim_ix)
        _print_progress!(completed_sims, sim_ix, nreps)
    end

end
