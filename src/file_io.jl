


"""
    write_system_state(agents::AllAgents, dims::SVector{2, Float64}, output_dir::String, time_ix::Int)

Writes the agent data and current dimensions to file using the JLD2 format.
"""
function write_system_state(agents::AllAgents, dims::SVector{2, Float64}, output_dir::String, time_ix::Int)
    out_fname = joinpath("out", output_dir, "raw_data", "sys_data_$(time_ix).jld2")
    @save out_fname agents dims
end



"""
    parse_config(file_path::String)

Parses a configuration file and populates an `AllParams` struct.
"""
function parse_config(file_path::String) :: AllParams
    
    # confirm that the file_path is an existing JSON file
    if !isfile(file_path)
        error("Configuration file does not exist: $file_path")
    end
    if !endswith(file_path, ".json")
        error("Configuration file must be a JSON file: $file_path")
    end

    #now read the file in
    config = JSON3.read(open(file_path, "r"), AllParams)

    return config
end


"""
    parse_sweep_config(file_path::String)

Parses a sweep configuration file directly as a dictionary
"""
function parse_sweep_config(file_path::String) :: SweepParams
    
    # confirm that the file_path is an existing JSON file
    if !isfile(file_path)
        error("Configuration file does not exist: $file_path")
    end
    if !endswith(file_path, ".json")
        error("Configuration file must be a JSON file: $file_path")
    end

    #now read the file in
    sweep_config = JSON3.read(open(file_path, "r"), SweepParams)

    return sweep_config
end


"""
    function copy_config_to_output_dir(file_path::String, output_dir::String)

Adds a copy of the specified config JSON to the output dir for post-processing.
"""
function copy_config_to_output_dir(file_path::String, output_dir::String)
    out_path = joinpath("out", output_dir)
    if !isdir(out_path)
        mkpath(out_path)
    end
    config_file_copy=joinpath(out_path, "config.json")
    if file_path == config_file_copy
        return
    end
    cp(file_path, config_file_copy, force=true)
end


"""
    set_up_output_directory(out_dir_path::String; clear_existing_output::Bool=false)

Sets up the specified output directory and optionally clears it if there is already data present.
"""
function set_up_output_directory(out_dir_path::String; clear_existing_output::Bool=false)
    raw_data_dir = joinpath("out", out_dir_path, "raw_data")
    if isdir(raw_data_dir)
        if !isempty(readdir(raw_data_dir))
            if clear_existing_output
                println("Wiping existing output directory: $raw_data_dir")
                rm(raw_data_dir; force=true, recursive=true)
                mkpath(raw_data_dir)
                return
            else
                println("CAUTION: there is already data in the output directory, $raw_data_dir. Do you want to delete this? (y/n)")
                wipe_dir_yn = readline()
                while !(wipe_dir_yn in ["y", "n"])
                    println("Please enter 'y' or 'n'")
                    wipe_dir_yn = readline()
                end
                if wipe_dir_yn == "y"
                    println("Wiping output directory...")
                    rm(raw_data_dir; force=true, recursive=true)
                    mkpath(raw_data_dir)
                else
                    error("Clear or change the output directory before running the simulation.")
                end
            end
        end
    else
        mkpath(raw_data_dir)
        println("Created output directory: $raw_data_dir")
    end
end