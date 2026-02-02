

"""
    build_sim_dict(sweep_config::SweepParams)

Helper function to build a dictionary of simulation parameter combinations for a given sweep configuration.
"""
function build_sim_dict(sweep_config::SweepParams)
    
    #calculate number of parameter combinations
    param_array_size = vcat([length(sweep_param.values) for (_,sweep_param) in sweep_config.sweep_params], [sweep_config.num_reps])
    num_param_combs = prod(param_array_size)

    #check if resuming from checkpoint and that the checkpoint sweep matches
    running_from_checkpoint_yn = false
    checkpoint_config = nothing
    final_ix = nothing
    if !isnothing(sweep_config.checkpoint_sweep_config)
        checkpoint_config = parse_sweep_config(sweep_config.checkpoint_sweep_config)

        #check compatibility
        @assert sweep_config.num_reps <= checkpoint_config.num_reps "Not enough checkpoint replicates to resume from"
        @assert length(sweep_config.sweep_params) == length(checkpoint_config.sweep_params) "Sweep parameters do not match checkpoint sweep"
        for (param_key, param_vals) in sweep_config.sweep_params
            @assert haskey(checkpoint_config.sweep_params, param_key) "Sweep parameters do not match checkpoint sweep"
            @assert length(param_vals.values) == length(checkpoint_config.sweep_params[param_key].values) "Sweep parameter values do not match checkpoint sweep"
            for (i, val) in enumerate(param_vals.values)
                if typeof(val)<:Float
                    @assert isapprox(val, checkpoint_config.sweep_params[param_key].values[i]) "Sweep parameter values do not match checkpoint sweep"
                else
                    @assert val == checkpoint_config.sweep_params[param_key].values[i] "Sweep parameter values do not match checkpoint sweep"
                end
            end
        end

        #get the final time index from the checkpoint config
        def_config = parse_config(checkpoint_config.default_config)
        final_ix = Int(def_config.system.t_max / def_config.system.vis_dt)

        running_from_checkpoint_yn = true
    end

    #build the simulation dictionary
    cartesian_inds = CartesianIndices(tuple(param_array_size...))
    sim_dict = Dict{String, Dict{String, Any}}()
    for ix=1:num_param_combs

        #construct the sim path name for this sim
        cartesian = cartesian_inds[ix]
        sim_path_name = ""
        for (param_ix, param) in enumerate(sweep_config.sweep_params)
            param_vals = param.second
            #TODO: requires short_name to exist, which not be the case
            sim_path_name *= "$(param_vals.short_name)$(cartesian[param_ix])/"
        end
        sim_path_name *= "rep$(cartesian[length(param_array_size)])"
        
        #make a dictionary of parameter values for this sim
        param_vals_this_sim = Dict{String, Any}()
        for (param_ix, param) in enumerate(sweep_config.sweep_params)
            param_path = param.first
            param_vals = param.second
            param_vals_this_sim[param_path] = param_vals.values[cartesian[param_ix]]
        end

        #also set the random seed to a unique value for each simulation
        #TODO: this assumes the default config has a seed parameter set
        param_vals_this_sim["system/seed"] = ix

        #if the sweep is being resumed from a checkpoint, set the checkpoint file path for this sim
        if running_from_checkpoint_yn
            checkpoint_file_path = joinpath("out", checkpoint_config.output_base_dir, sim_path_name, "raw_data/sys_data_$(final_ix).jld2")
            param_vals_this_sim["init/checkpoint_file"] = checkpoint_file_path
        end

        #read into the overall dictionary
        sim_dict[sim_path_name] = param_vals_this_sim
    end

    return sim_dict
end


"""
    make_config_this_sim(params::AllParams, sim_param_changes::Dict{String, Any}, output_dir::String)

Helper function to make a config for a specific simulation, given a base parameter set, a dictionary of parameter 
changes for this simulation, and an output directory.
"""
function make_config_this_sim(params::AllParams, sim_param_changes::Dict{String, Any}, output_dir::String)

    for (param_path, param_val) in sim_param_changes
        params = modify_params(params, param_path, param_val)
    end

    @reset params.system.output_dir = output_dir

    return params
end


"""
    modify_params(params, param_path::String, param_val)

Helper function to modify a parameter in the params structure, given its path as a string (with '/' separators) and the new value.
"""
function modify_params(params, param_path::String, param_val)

    #change the parameter specified by param_path to param_val
    param_branches = split(param_path, "/")
    if length(param_branches)==1
        return change_field(params, String(param_branches[1]), param_val)
    else
        sub_params = getfield(params, Symbol(param_branches[1]))
        return change_field(params, String(param_branches[1]), 
                            modify_params(sub_params, joinpath(param_branches[2:end]), param_val))
    end
end


