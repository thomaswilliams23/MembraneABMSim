


function build_sim_dict(sweep_config::SweepParams)
    
    #calculate number of parameter combinations
    param_array_size = vcat([length(sweep_param.values) for (_,sweep_param) in sweep_config.sweep_params], [sweep_config.num_reps])
    num_param_combs = prod(param_array_size)

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

        #read into the overall dictionary
        sim_dict[sim_path_name] = param_vals_this_sim
    end

    return sim_dict
end



function make_config_this_sim(params::AllParams, sim_param_changes::Dict{String, Any}, output_dir::String)

    for (param_path, param_val) in sim_param_changes
        params = modify_params(params, param_path, param_val)
    end

    @reset params.system.output_dir = output_dir

    return params
end


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


