include("MuBetaZero.jl")
include("ConnectFour.jl")

using Random
using Flux
using StatsBase


mutable struct MuBetaZeroNeural <: MuBetaZero

    policy_networks
    value_networks

    transition_buffer::Array{Vector{Transition}}
    γ::Float32
    ϵ::Float64
    c::Float32 # regularisation parameter
    opt

    function MuBetaZeroNeural(policy_network_layout::Chain, value_network_layout::Chain;
                              γ=1.0f0, opt=RMSProp(), ϵ=0.1, c=0.001f0)
        this = new()
        this.policy_networks = [deepcopy(policy_network_layout), deepcopy(policy_network_layout)]
        this.value_networks = [deepcopy(value_network_layout), deepcopy(value_network_layout)]
        this.transition_buffer = [[], []]
        this.γ = γ
        this.opt = opt
        this.ϵ = ϵ
        this.c = c
        return this
    end
end

function flush_transition_buffer(μβ0::MuBetaZeroNeural)
    μβ0.transition_buffer = [[], []]
end

function greedy_action(μβ0::MuBetaZeroNeural, env::Environment, state::Array{Float32}, player::Int)::Int
    x = reshape(state, size(state)..., 1)
    ps = μβ0.policy_networks[player](x)[:,1]
    a = argmax(ps)
    return a
end

function action(μβ0::MuBetaZeroNeural, env::Environment, state::Array{Float32}, player::Int)::Int
    x = reshape(state, size(state)..., 1)
    ps = μβ0.policy_networks[player](x)[:,1]
    a = sample(1:env.n_actions, Weights(ps))
    return a
end

function value(μβ0::MuBetaZeroNeural, env::Environment, state::Array{Float32}, player::Int)::Float32
    x = reshape(state, size(state)..., 1)
    return μβ0.value_networks[player](x)[1]
end

function L2(x)
    return sum(abs2, x)
end

function policy_loss(μβ0::MuBetaZeroNeural, player::Int, c::Float32=0.001f0)
    m = μβ0.policy_networks[player]
    function loss(x,y)
        return Flux.crossentropy(m(x), y) + c * sum(L2, params(m))
    end
end

function value_loss(μβ0::MuBetaZeroNeural, player::Int, c::Float32=0.001f0)
    m = μβ0.value_networks[player]
    function loss(x,y)
        return Flux.mse(m(x), y) + c * sum(L2, params(m))
    end
end

function learn_transitions!(μβ0::MuBetaZeroNeural, env::Environment)
    ls = []
    for player in [1,2]
        X = Array{Float32}(undef, size(env.current)..., length(μβ0.transition_buffer[player]))
        Y_ps = Array{Float32}(undef, env.n_actions, length(μβ0.transition_buffer[player]))
        Y_vs = Array{Float32}(undef, 1, length(μβ0.transition_buffer[player]))
        for (i,t) in enumerate(μβ0.transition_buffer[player])
            X[:,:,:,i] = t.s
            Y_ps[:,i] = t.ps
            Y_vs[1,i] = t.Q_est
        end

        policy_loss_f = policy_loss(μβ0, env, μβ0.c)
        value_loss_f = value_loss(μβ0, env, μβ0.c)

        p_loss = policy_loss(X, Y_ps)
        v_loss = value_loss_f(X, Y_vs)
        push!(ls, (p_loss, v_loss))

        Flux.Optimise.train!(policy_loss_f, params(μβ0.policy_network[player]), [(X,Y_ps)], μβ0.opt)
        Flux.Optimise.train!(value_loss_f, params(μβ0.value_network[player]), [(X, Y_vs)], μβ0.opt)
    end

    return ls
end


function play_game!(μβ0::MuBetaZeroNeural, env::Environment;
                    verbose=false, train=false, MCTS=false, N_MCTS=1000, MCTS_type=:rollout)
    reset!(env)
    if MCTS # reset tree
        reset_tree!(μβ0)
    end

    winner = 0
    done = false
    player = 1

    println_current(env)
    println()
    while !done
        t, winner, done, nextplayer = play!(μβ0, env, player, train=train, MCTS=MCTS, N_MCTS=N_MCTS, MCTS_type=MCTS_type)

        if train
            push!(μβ0.transition_buffer[t.player], t)
        end

        if verbose
            println("Decision Stats: player: $player, Q_est: $(t.Q_est) vs Q: $(value(μβ0, env, t.s, player))")
            print_current(env)
            !done && println("State Stats: player: $nextplayer, Q = $(value(μβ0, env, env.current, nextplayer))")
            println()
        end

        player = nextplayer
    end

    return winner
end

env = ConnectFour()

Random.seed!(1)
policy_model = Chain(
    Conv((4,4), 2 => 8, sigmoid), # (3,4,8,:)
    Flux.flatten, # (96,:)
    Dense(96, 48, sigmoid),
    Dense(48, 7),
    Flux.softmax
)

value_model = Chain(
    Conv((4,4), 2 => 8, sigmoid), # (3,4,8,:)
    Flux.flatten, # (96,:)
    Dense(96, 48, relu),
    Dense(48, 1)
)

agent = MuBetaZeroNeural(policy_model, value_model)

play_game!(agent, env, verbose=true, train=false)

learn_transitions!(agent, env)
