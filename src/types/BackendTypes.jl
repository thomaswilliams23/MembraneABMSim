"""
    AbstractSimBackend

Supertype for the compute backend a simulation runs on (`SimBackendCPU`, `SimBackendMetal`, `SimBackendCUDA`).
"""
abstract type AbstractSimBackend end

"""
    SimBackendCPU <: AbstractSimBackend

CPU backend. Holds no extra data: the CPU uses `SimState.flat` directly.
"""
struct SimBackendCPU <: AbstractSimBackend end

"""
    SimBackendMetal <: AbstractSimBackend

Metal GPU backend: the compiled kernels, the device-side data, and Float32 copies of the nascent increments.
"""
struct SimBackendMetal{FK, NK} <: AbstractSimBackend
    force_kernel::FK
    non_force_position_kernel::NK
    all_data::AllDataMetal
    grid_size_dev::GridSizeMetal
    params_dev::ParamsMetal
    effective_rad_incs::SVector{7, Float32}
    ideal_dist_incs::SVector{7, Float32}
end

"""
    SimBackendCUDA <: AbstractSimBackend

CUDA GPU backend: the compiled kernels, the device-side data, and Float32 copies of the nascent increments.
"""
struct SimBackendCUDA{FK, NK} <: AbstractSimBackend
    force_kernel::FK
    non_force_position_kernel::NK
    all_data::AllDataCUDA
    grid_size_dev::GridSizeCUDA
    params_dev::ParamsCUDA
    effective_rad_incs::SVector{7, Float32}
    ideal_dist_incs::SVector{7, Float32}
end
