
"""
    get_time_series(func::Function, out_path::String, params::AllParams)

Extracts time series data for a specified function `func` from simulation output files located in `out_path`.
`func` takes as input an AllAgents object, dimensions (SVector{2, Float64}), and AllParams, and returns a
value which can be converted to Float64.
"""
function get_time_series(func::Function, out_path::String, params::AllParams)
    
    #raw data directory
    raw_data_dir = joinpath(out_path, "raw_data")

    #allocate memory
    max_time_ix = round(Int, params.system.t_max / params.system.vis_dt)
    time_series = Vector{Float64}(undef, max_time_ix+1)

    #iterate through time points
    Threads.@threads for time_ix in 0:max_time_ix

        #load in system state at this time point
        fname_this_time_ix = joinpath(raw_data_dir, @sprintf("sys_data_%d.jld2", time_ix))
        @load fname_this_time_ix agents dims

        #compute function value
        time_series[time_ix+1] = func(agents, dims, params)

    end

    return time_series

end



"""
    analyse_sim(func::Function, data_name::String, simulation_config_fname::String; get_time_series_yn::Bool=false, evaluate_final_state_yn::Bool=false)

Applies a function `func` to a single simulation specified by `simulation_config`,
then saves the results to a JLD2 file in the simulation output directory.
"""
function analyse_sim(func::Function, data_name::String, simulation_config_fname::String;
                     get_time_series_yn::Bool=false, evaluate_final_state_yn::Bool=false,
                     analyse_whole_traj_yn::Bool=false)


    @assert !(get_time_series_yn && evaluate_final_state_yn) "Cannot set both get_time_series and evaluate_final_state to true"

    # Load the simulation configuration
    params_this_sim = parse_config(simulation_config_fname)

    out_path = joinpath("out", params_this_sim.system.output_dir)

    # Compute the desired data
    if get_time_series_yn
        #apply the function to each time point
        result = get_time_series(func, out_path, params_this_sim)
    elseif evaluate_final_state_yn
        #apply the function only to the final system state

        #load in final state
        raw_data_dir = joinpath(out_path, "raw_data")
        max_time_ix = round(Int, params_this_sim.system.t_max / params_this_sim.system.vis_dt)
        fname_final = joinpath(raw_data_dir, @sprintf("sys_data_%d.jld2", max_time_ix))
        @load fname_final agents dims

        #apply function
        result = func(agents, dims, params_this_sim)
    elseif analyse_whole_traj_yn
        #apply the function to the whole trajectory at once 
        #(NOTE requires function to have different arguments)
        result = func(params_this_sim)
    else
        #TODO: implement other analysis types
        error("Only time series and final state evaluation are currently implemented")
    end

    # Save the result to a JLD2 file
    proc_data_dir = joinpath(out_path, "processed_data")
    if !isdir(proc_data_dir)
        mkpath(proc_data_dir)
    end
    analysis_fname = joinpath(proc_data_dir, "$(data_name).jld2")
    @save analysis_fname result

end



"""
    analyse_sweep(func::Function, data_name::String, sweep_config_fname::String; get_time_series_yn::Bool=false, evaluate_final_state_yn::Bool=false)

Applies a function `func` to the results of all simulations in a parameter sweep specified by `sweep_config_fname`,
then saves the results to a JLD2 file in the sweep output directory.
"""
function analyse_sweep(func::Function, data_name::String, sweep_config_fname::String;
                       get_time_series_yn::Bool=false, evaluate_final_state_yn::Bool=false,
                       analyse_whole_traj_yn::Bool=false)

    @assert !(get_time_series_yn && evaluate_final_state_yn) "Cannot set both get_time_series and evaluate_final_state to true"

    # Load the sweep configuration
    sweep_params = parse_sweep_config(sweep_config_fname)
    sim_dict = build_sim_dict(sweep_params)

    # Iterate over all parameter combinations
    sim_ix = 0
    num_sims = length(keys(sim_dict))
    Threads.@threads for sim_path in [_ for _ in keys(sim_dict)]
    
        sim_config_fname = joinpath("out", sweep_params.output_base_dir, sim_path, "config.json")

        analyse_sim(func, data_name, sim_config_fname;
                    get_time_series_yn=get_time_series_yn, 
                    evaluate_final_state_yn=evaluate_final_state_yn,
                    analyse_whole_traj_yn=analyse_whole_traj_yn)

        print("Processed sim $sim_ix of $num_sims\r")

        sim_ix += 1
    end

    println("Finished processing all simulations.")

end