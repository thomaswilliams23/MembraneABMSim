# Builders: turn a test scenario into parameters and a ready-to-step `SimState`.
#
# Tests describe *what* they want (agent positions, which rates are on) and never call agent,
# grid or flat-data constructors directly. States are built through the package's own
# initialisation (`initialise_sim`), so if a struct or constructor changes, only this file
# needs updating.

const DEMO_CONFIG = joinpath(@__DIR__, "..", "..", "config", "simulation", "demo.json")

# replace fields of an immutable struct by keyword
with(obj; kw...) = Accessors.setproperties(obj, (; kw...))

"""
    params_with(base=demo params; section=(field=value, ...), ...)

Parameters with selected fields replaced, by section, e.g.
`params_with(force=(temperature=0.0,), system=(dt=1e-3,))`. Output goes to a fresh temporary
directory unless `system.output_dir` is given (`joinpath("out", abs_path) == abs_path`, so
nothing is written into the repo's `out/`).
"""
function params_with(base::M.AllParams=M.parse_config(DEMO_CONFIG); sections...)
    p = with(base; system=with(base.system; output_dir=mktempdir()))
    for (section, fields) in pairs(sections)
        p = with(p; section => with(getfield(p, section); fields...))
    end
    return p
end

"""
    quiet_params(; sections...)

Demo parameters with every biological process switched off (no polypeptides, LPS arrivals,
tethering or assembly) and no diffusion, so only deterministic forces act. Overrides as for
`params_with`.
"""
quiet_params(; sections...) = params_with(params_with(
    OmpA=(tether_rate=0.0,),
    LptD=(complex_assembly_rate=0.0,),
    BamA=(complex_assembly_rate=0.0,),
    LPS=(arrival_rate=0.0,),
    insertion=(PP_arrival_rate=0.0, PP_bind_rate=0.0),
    init=(num_PP=0,),
    force=(temperature=0.0,),
); sections...)

"""
    busy_params(; sections...)

Demo parameters with insertion and tethering rates raised so that every biological process
(polypeptide binding, OMP and LPS insertion, promotion, tethering) happens within a few
thousand steps. No diffusion. Overrides as for `params_with`.

The rates are high enough that "something happened" checks are effectively certain: a free,
assembled LptD waits on average 0.01 time units (100 steps) for an LPS, so the chance that it
sees no arrival in 0.1 time units is e^-10.
"""
busy_params(; sections...) = params_with(params_with(
    OmpA=(tether_rate=5.0,),
    LPS=(arrival_rate=100.0, insertion_time=0.05),
    insertion=(type="guaranteed", PP_arrival_rate=50.0, PP_bind_rate=50.0, OMP_assembly_rate=2.0, attempt_dt=0.01),
    force=(temperature=0.0,),
); sections...)


const OMP_TYPES = ("OmpA", "OmpCF", "BamA", "LptD")

"""
    single_insertion_params(type; sections...)

Quiet parameters in which an assembled BamA given one polypeptide (`num_PP=1`) binds it at once
and inserts exactly one OMP of `type` at its first attempt, with a short insertion time.
Overrides as for `params_with`.
"""
function single_insertion_params(type::String; sections...)
    probs = Dict(t => Float64(t == type) for t in OMP_TYPES)
    p = quiet_params(
        OmpA=(insertion_prob=probs["OmpA"],), OmpCF=(insertion_prob=probs["OmpCF"],),
        BamA=(insertion_prob=probs["BamA"],), LptD=(insertion_prob=probs["LptD"],),
        insertion=(type="guaranteed", PP_bind_rate=1e9, attempt_dt=1e-4, OMP_assembly_rate=4.0),
    )
    return params_with(p; sections...)
end


_positions(v) = isempty(v) ? nothing : [SVector{2, Float64}(p...) for p in v]

"""
    build_state(params; OmpA=[], OmpCF=[], LptD=[], BamA=[], LPS=[], L=params.init.dim_x,
                assembled=true, num_PP=0, rng=Xoshiro(1), device="cpu") -> (state, params)

Builds a `SimState` with agents at the given positions (any iterables of 2-tuples or vectors)
on an L x L domain, with no equilibration. CPU only: the GPU backends support only the random
initialisation (use `random_state`). If `assembled`, OmpA and LptD start tethered at their
positions and BamA/LptD start assembled. Returns the state and the parameters actually used.
"""
function build_state(params::M.AllParams;
                     OmpA=[], OmpCF=[], LptD=[], BamA=[], LPS=[],
                     L::Real=params.init.dim_x, assembled::Bool=true, num_PP::Int=0,
                     rng::AbstractRNG=Xoshiro(1), device::String="cpu")
    p = params_with(params;
        init=(
            method="specify_positions",
            dim_x=Float64(L), dim_y=Float64(L),
            num_OmpA=length(OmpA), num_OmpCF=length(OmpCF), num_LptD=length(LptD),
            num_BamA=length(BamA), num_LPS=length(LPS), num_PP=num_PP,
            positions=M.InitPositions(_positions(OmpA), _positions(OmpCF), _positions(LptD),
                                      _positions(BamA), _positions(LPS)),
            equilibration_time=0.0,
            complexes_assembled=assembled,
            shrink_to_size=false,
        ),
        system=(device=device,),
    )
    state = M.initialise_sim(p; rng=rng, suppress_prints=true)
    return state, p
end

"""
    random_state(params; OmpA=0, OmpCF=0, LptD=0, BamA=0, LPS=0, L=params.init.dim_x,
                 assembled=true, num_PP=0, rng=Xoshiro(1), device="cpu") -> (state, params)

Builds a `SimState` with the given numbers of agents placed uniformly at random (overlaps
allowed, no equilibration), through the package's "random" initialisation. This is the one
initialisation every backend supports, and with the same `rng` it places agents identically on
every backend.
"""
function random_state(params::M.AllParams;
                      OmpA::Int=0, OmpCF::Int=0, LptD::Int=0, BamA::Int=0, LPS::Int=0,
                      L::Real=params.init.dim_x, assembled::Bool=true, num_PP::Int=0,
                      rng::AbstractRNG=Xoshiro(1), device::String="cpu")
    p = params_with(params;
        init=(
            method="random",
            dim_x=Float64(L), dim_y=Float64(L),
            num_OmpA=OmpA, num_OmpCF=OmpCF, num_LptD=LptD, num_BamA=BamA, num_LPS=LPS, num_PP=num_PP,
            positions=nothing,
            equilibration_time=0.0,
            complexes_assembled=assembled,
            shrink_to_size=false,
        ),
        system=(device=device,),
    )
    state = M.initialise_sim(p; rng=rng, suppress_prints=true)
    return state, p
end


# ---- stepping ----

"""
    run_steps!(state, params, n; t0=0.0) -> t

Advances `state` by `n` steps starting from time `t0` and returns the final time. Times are
whole multiples of `dt`, as in `run_sim`.
"""
function run_steps!(state::M.SimState, params::M.AllParams, n::Int; t0::Float64=0.0)
    dt = params.system.dt
    ix0 = round(Int, t0 / dt)
    for ix in (ix0 + 1):(ix0 + n)
        M.step!(state, params, ix * dt)
    end
    return (ix0 + n) * dt
end

"""
    step_until!(cond, state, params; t0=0.0, max_steps=100_000) -> t

Steps until `cond(state)` is true (checked after each step) and returns the time. Errors if
`max_steps` is reached first, so a scenario that never happens fails loudly.
"""
function step_until!(cond, state::M.SimState, params::M.AllParams; t0::Float64=0.0, max_steps::Int=100_000)
    t = t0
    for _ in 1:max_steps
        t = run_steps!(state, params, 1; t0=t)
        cond(state) && return t
    end
    error("condition not reached within $max_steps steps")
end


# ---- reading the agent structs ----

"""
    all_agents(state_or_agents)

Every agent (including nascent ones) as one list.
"""
all_agents(agents::M.AllAgents) = collect(Iterators.flatten((agents.OMP.OmpA, agents.OMP.OmpCF,
    agents.OMP.LptD, agents.OMP.BamA, agents.LPS, agents.nascent.nascent_OMP, agents.nascent.nascent_LPS)))
all_agents(state::M.SimState) = all_agents(state.agents)

"agent type as a string (\"OmpA\", \"OmpCF\", \"LptD\", \"BamA\", \"LPS\"); nascent agents report the type they will become"
agent_type(a::M.OmpAAgent) = "OmpA"
agent_type(a::M.OmpCFAgent) = "OmpCF"
agent_type(a::M.LptDAgent) = "LptD"
agent_type(a::M.BamAAgent) = "BamA"
agent_type(a::Union{M.LPSAgent, M.NascentLPSAgent}) = "LPS"
agent_type(a::M.NascentOMPAgent) = a.OMP_type

is_nascent(a) = a isa M.AbstractNascentAgent
is_OMP(a) = agent_type(a) != "LPS"
is_tethered(a) = hasproperty(a, :is_tethered) && a.is_tethered

"full radius of an agent type from the parameters"
type_radius(params::M.AllParams, type::String) = getfield(params, Symbol(type)).radius

"current interaction radius: a nascent OMP's growing radius, otherwise its type's radius"
current_radius(params::M.AllParams, a) = a isa M.NascentOMPAgent ? a.effective_radius : type_radius(params, agent_type(a))

"tether radius for a tethered agent type"
tether_radius(params::M.AllParams, a) = getfield(params, Symbol(agent_type(a))).tether_radius

"number of agents of each type, nascent agents counted separately"
agent_counts(state::M.SimState) = (
    OmpA=length(state.agents.OMP.OmpA), OmpCF=length(state.agents.OMP.OmpCF),
    LptD=length(state.agents.OMP.LptD), BamA=length(state.agents.OMP.BamA),
    LPS=length(state.agents.LPS),
    nascent_OMP=length(state.agents.nascent.nascent_OMP), nascent_LPS=length(state.agents.nascent.nascent_LPS))

"number of nascent agents (OMP and LPS)"
num_nascent(state::M.SimState) = length(state.agents.nascent.nascent_OMP) + length(state.agents.nascent.nascent_LPS)

"""
    insertion_states_of(state)

Biological state of every full OMP, keyed by agent index: (insertion state, tethered, assembled),
with `nothing` for properties the type does not have.
"""
insertion_states_of(state::M.SimState) = Dict(
    a.index => (hasproperty(a, :insertion_state) ? a.insertion_state : nothing,
                hasproperty(a, :is_tethered) ? a.is_tethered : nothing,
                hasproperty(a, :is_part_of_assembled_complex) ? a.is_part_of_assembled_complex : nothing)
    for a in all_agents(state) if is_OMP(a) && !is_nascent(a))

"positions of every agent, keyed by agent index"
positions_by_index(state::M.SimState) = Dict(a.index => a.position for a in all_agents(state))

"domain area"
area(state::M.SimState) = prod(state.grid_size.dims)

"rescale the domain (and every position) by `s`, as domain growth does"
scale_domain!(state::M.SimState, s::Real) = M.apply_scale_factor!(state.agents, state.flat, state.grid_size, Float64(s))

# compute_nascent_incs returns growth increments indexed by agent type number (OmpA 1, LPS 2,
# OmpCF 3, BamA 5, LptD 7). Slots 4 and 6 are unused and not initialised, so never read them.
const INCREMENT_SLOTS = (1, 2, 3, 5, 7)

"largest per-step growth increments of any nascent type: (effective radius, ideal distance)"
function max_growth_increments(params::M.AllParams)
    rad_incs, dist_incs = M.compute_nascent_incs(params)
    return maximum(rad_incs[i] for i in INCREMENT_SLOTS), maximum(dist_incs[i] for i in INCREMENT_SLOTS)
end


# ---- comparing states ----

"""
    deep_equal(a, b)

Field-by-field equality for nested structs, arrays and dicts (mutable structs compare by
content, not identity).
"""
deep_equal(a::T, b::T) where {T<:Union{Number, AbstractString, Symbol, Nothing, Bool}} = isequal(a, b)
deep_equal(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(deep_equal(x, y) for (x, y) in zip(a, b))
deep_equal(a::AbstractDict, b::AbstractDict) = keys(a) == keys(b) && all(deep_equal(a[k], b[k]) for k in keys(a))
deep_equal(a::T, b::T) where {T} = all(deep_equal(getfield(a, f), getfield(b, f)) for f in fieldnames(T))
deep_equal(a, b) = false

"""
    load_frame(output_dir, ix) -> (agents, dims)

Loads output frame `ix` written by `run_sim` to `output_dir`.
"""
function load_frame(output_dir::String, ix::Int)
    d = JLD2.load(joinpath("out", output_dir, "raw_data", "sys_data_$(ix).jld2"))
    return d["agents"], d["dims"]
end

"number of output frames in `output_dir`"
num_frames(output_dir::String) = count(f -> endswith(f, ".jld2"), readdir(joinpath("out", output_dir, "raw_data")))

"""
    write_config(params) -> path

Writes `params` as a JSON config file (in a fresh temporary directory) and returns its path.
"""
function write_config(params::M.AllParams)
    path = joinpath(mktempdir(), "config.json")
    write(path, JSON3.write(params))
    return path
end

"""
    state_from_frame(params, frame_path; t) -> (state, params)

Rebuilds a `SimState` from a saved output frame through the checkpoint initialisation, so a
saved frame can be checked like a live state.
"""
function state_from_frame(params::M.AllParams, frame_path::String; t::Float64=0.0)
    p = params_with(params; init=(method="from_checkpoint", checkpoint_file=frame_path, checkpoint_time=t))
    return M.initialise_sim(p; suppress_prints=true), p
end
