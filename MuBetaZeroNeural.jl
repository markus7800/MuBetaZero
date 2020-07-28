using Flux
using StatsBase

mutable struct MuBetaZeroNeural <: MuBetaZero

    policy_networks
    value_networks

    function MuBetaZeroNeural(policy_network_layout::Chain, value_network_layout::Chain)
        this = new()
        this.policy_networks = [copy(policy_network_layout), copy(policy_network_layout)]
        this.value_networks = [copy(value_network_layout), copy(value_network_layout)]
        return this
    end
end

function action(μβ0::MuBetaZeroNeural, env::Environment, state::Array{Float32}, player::Int)::Int
    ps = μβ0.policy_networks[player](state)
    a = sample(1:env.n_actions, Weights(ps))
    return a
end

function value(μβ0::MuBetaZeroNeural, env::Environment, state::Array{Float32}, player::Int)::Float32
    return μβ0.value_networks[player](state)
end


function play!(μβ0::MuBetaZeroNeural, env::Environment, adversary::MuBetaZeroNeural,
               player::Int; ϵ_greedy=false)::Tuple{Vector{Int}, Int, Float32, Bool, Vector{Int}}
    s = copy(env.current)
    rand_decision = ϵ_greedy && rand() ≤ μβ0.ϵ
    if rand_decision
        a = rand(collect(valid_actions(env, s)))
    else
        a = action(μβ0, env, s, player) # TODO: MCTS
    end

    r1, done, = step!(env, a, player, false)
    r2 = 0f0
    if !done
        player_adv = player == 1 ? 2 : 1
        a_adv, = action(adversary, env, env.current, player_adv)
        r2, done = step!(env, a_adv, player_adv, true)
    end
    r = r1 + r2
    ns = copy(env.current)

    return s, a, r, done, ns
end
