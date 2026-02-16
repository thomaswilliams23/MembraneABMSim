
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