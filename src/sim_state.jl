"""
    resolve_device(params::AllParams; suppress_prints::Bool=false)

Returns the device to run on ("cpu", "metal" or "cuda") from `params.system.device`, checking
that a requested GPU is available. A missing device defaults to "cpu".
"""
function resolve_device(params::AllParams; suppress_prints::Bool=false)
    if params.system.device == "metal"
        @assert Metal.functional() "Metal device specified but no Metal-compatible GPU found."
        if !suppress_prints
            println("Running on Metal GPU")
        end
        return "metal"
    elseif params.system.device == "cuda"
        @assert CUDA.has_cuda() "CUDA device specified but no CUDA-compatible GPU found."
        if !suppress_prints
            println("Running on CUDA GPU")
        end
        return "cuda"
    elseif params.system.device == "cpu"
        if !suppress_prints
            println("Running on CPU")
        end
        return "cpu"
    elseif !isnothing(params.system.device)
        error("Device type $(params.system.device) not recognised")
    end
    return "cpu"
end


"""
    initialise_sim(params::AllParams; rng::AbstractRNG=Random.default_rng(), clear_existing_output::Bool=false, suppress_prints::Bool=false) :: SimState

Builds the nascent lookups and initialises the system on the device given by the params,
returning a `SimState` ready for `step!`. Sets up the output directory but writes no state.
"""
function initialise_sim(params::AllParams; rng::AbstractRNG=Random.default_rng(), clear_existing_output::Bool=false, suppress_prints::Bool=false)

    #determine device to use
    device = resolve_device(params; suppress_prints=suppress_prints)

    #build nascent lookups for speed
    nascent_added_area_lookup = build_nascent_added_area_lookup(params)
    effective_rad_incs, ideal_dist_incs = compute_nascent_incs(params)

    #initialise the system (device-specific)
    if device == "cpu"
        (
            agents,
            grid_size,
            grid,
            system_flat_cpu
        ) = initialise_system_cpu(params; clear_existing_output=clear_existing_output, suppress_prints=suppress_prints, rng=rng)
        backend = SimBackendCPU()
    elseif device == "metal"
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
        ) = initialise_system_metal(params; clear_existing_output=clear_existing_output, suppress_prints=suppress_prints, rng=rng)
        backend = SimBackendMetal(force_kernel, non_force_position_kernel, all_data_metal, grid_size_metal, params_metal,
                                  Float32.(effective_rad_incs), Float32.(ideal_dist_incs))
    elseif device == "cuda"
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
        ) = initialise_system_CUDA(params; clear_existing_output=clear_existing_output, suppress_prints=suppress_prints, rng=rng)
        backend = SimBackendCUDA(force_kernel, non_force_position_kernel, all_data_CUDA, grid_size_CUDA, params_CUDA,
                                 Float32.(effective_rad_incs), Float32.(ideal_dist_incs))
    end

    return SimState(
        agents,
        grid_size,
        grid,
        system_flat_cpu,
        backend,
        nascent_added_area_lookup,
        effective_rad_incs,
        ideal_dist_incs,
        true,   #only permit the membrane to grow if no holes are present - checked at every grid sync
        0,
        max_steps_between_grid_sync(params),
        rng
    )
end


"""
    copy_to_host!(state::SimState)

Copies device data back into the CPU structs and flat arrays (no-op on CPU).
"""
copy_to_host!(state::SimState{SimBackendCPU}) = nothing
copy_to_host!(state::SimState{<:SimBackendMetal}) =
    copy_data_to_cpu_from_metal!(state.flat, state.agents, state.backend.all_data, state.backend.grid_size_dev)
copy_to_host!(state::SimState{<:SimBackendCUDA}) =
    copy_data_to_cpu_from_CUDA!(state.flat, state.agents, state.backend.all_data, state.backend.grid_size_dev)

"""
    copy_to_device!(state::SimState)

Copies the CPU flat data and grid onto the device (no-op on CPU).
"""
copy_to_device!(state::SimState{SimBackendCPU}) = nothing
copy_to_device!(state::SimState{<:SimBackendMetal}) =
    copy_data_to_metal!(state.backend.all_data, state.backend.grid_size_dev, state.flat, state.grid_size, state.grid)
copy_to_device!(state::SimState{<:SimBackendCUDA}) =
    copy_data_to_CUDA!(state.backend.all_data, state.backend.grid_size_dev, state.flat, state.grid_size, state.grid)

"""
    push_nascent_promotion!(state::SimState)

Sends updated nascent agent information to the device after a promotion (no-op on CPU).
"""
push_nascent_promotion!(state::SimState{SimBackendCPU}) = nothing
push_nascent_promotion!(state::SimState{<:SimBackendMetal}) =
    update_nascent_promotion_metal!(state.backend.all_data, state.flat, state.grid_size)
push_nascent_promotion!(state::SimState{<:SimBackendCUDA}) =
    update_nascent_promotion_CUDA!(state.backend.all_data, state.flat, state.grid_size)

"""
    push_newly_tethered!(state::SimState, newly_tethered_agent_ixs::Vector{Int})

Sends newly tethered agent indices to the device (no-op on CPU, or when there are none).
"""
push_newly_tethered!(state::SimState{SimBackendCPU}, newly_tethered_agent_ixs::Vector{Int}) = nothing
function push_newly_tethered!(state::SimState{<:SimBackendMetal}, newly_tethered_agent_ixs::Vector{Int})
    if length(newly_tethered_agent_ixs) > 0
        update_newly_tethered_agent_ixs_metal!(state.backend.all_data, newly_tethered_agent_ixs)
    end
end
function push_newly_tethered!(state::SimState{<:SimBackendCUDA}, newly_tethered_agent_ixs::Vector{Int})
    if length(newly_tethered_agent_ixs) > 0
        update_newly_tethered_agent_ixs_CUDA!(state.backend.all_data, newly_tethered_agent_ixs)
    end
end

"""
    apply_growth_and_nonforce!(state::SimState, params::AllParams, t::Float64, newly_tethered_agent_ixs::Vector{Int})

Rescales the domain for the area added by insertions (if growth is allowed). On a GPU this also
applies the other non-force position updates (tethering, nascent sizes), so it runs every step.
"""
function apply_growth_and_nonforce!(state::SimState{SimBackendCPU}, params::AllParams, t::Float64, newly_tethered_agent_ixs::Vector{Int})
    if state.allow_membrane_growth
        rescale_domain!(state.agents, state.flat, state.grid_size, params, state.nascent_added_area_lookup, t)
    end
end
function apply_growth_and_nonforce!(state::SimState{<:SimBackendMetal}, params::AllParams, t::Float64, newly_tethered_agent_ixs::Vector{Int})
    b = state.backend
    compute_non_force_position_changes_metal!(
        b.non_force_position_kernel,
        state.agents,
        b.all_data,
        state.grid_size,
        b.grid_size_dev,
        b.effective_rad_incs,
        b.ideal_dist_incs,
        state.nascent_added_area_lookup,
        newly_tethered_agent_ixs,
        params,
        t,
        allow_membrane_growth=state.allow_membrane_growth
    )
end
function apply_growth_and_nonforce!(state::SimState{<:SimBackendCUDA}, params::AllParams, t::Float64, newly_tethered_agent_ixs::Vector{Int})
    b = state.backend
    compute_non_force_position_changes_CUDA!(
        b.non_force_position_kernel,
        state.agents,
        b.all_data,
        state.grid_size,
        b.grid_size_dev,
        b.effective_rad_incs,
        b.ideal_dist_incs,
        state.nascent_added_area_lookup,
        newly_tethered_agent_ixs,
        params,
        t,
        allow_membrane_growth=state.allow_membrane_growth
    )
end

"""
    advance_positions!(state::SimState, params::AllParams, t::Float64)

Applies diffusion (CPU only) and resolves inter-agent forces. On a GPU, copies the data back to
the CPU when the host will need it next step (see `copy_data_to_cpu_yn`).
"""
function advance_positions!(state::SimState{SimBackendCPU}, params::AllParams, t::Float64)
    compute_diffusion!(state.agents, state.flat, state.grid_size, params; rng=state.rng)
    resolve_forces_cpu!(state.agents, state.grid_size, state.grid, state.flat, params)
end
function advance_positions!(state::SimState{<:SimBackendMetal}, params::AllParams, t::Float64)
    resolve_forces_metal!(state.backend.force_kernel, state.backend.all_data, state.backend.grid_size_dev, state.backend.params_dev)
    if copy_data_to_cpu_yn(state.agents.OMP.BamA, params, t)
        copy_to_host!(state)
    end
end
function advance_positions!(state::SimState{<:SimBackendCUDA}, params::AllParams, t::Float64)
    resolve_forces_CUDA!(state.backend.force_kernel, state.backend.all_data, state.backend.grid_size_dev, state.backend.params_dev)
    if copy_data_to_cpu_yn(state.agents.OMP.BamA, params, t)
        copy_to_host!(state)
    end
end
