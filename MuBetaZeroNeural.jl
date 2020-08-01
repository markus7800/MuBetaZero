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
    opt

    function MuBetaZeroNeural(policy_network_layout::Chain, value_network_layout::Chain;
                              γ=1.0f0, opt=RMSProp(), ϵ=0.1)
        this = new()
        this.policy_networks = [deepcopy(policy_network_layout), deepcopy(policy_network_layout)]
        this.value_networks = [deepcopy(value_network_layout), deepcopy(value_network_layout)]
        this.transition_buffer = [[], []]
        this.γ = γ
        this.opt = opt
        this.ϵ = ϵ
        return this
    end
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

function policy_loss(μβ0::MuBetaZeroNeural, player::Int, c::Float32=0.001)
    m = μβ0.policy_networks[player]
    function loss(x,y)
        return Flux.crossentropy(m(x), y) + c * sum(L2, params(m))
    end
end

function value_loss(μβ0::MuBetaZeroNeural, player::Int, c::Float32=0.001)
    m = μβ0.value_networks[player]
    function loss(x,y)
        return Flux.mse(m(x), y) + c * sum(L2, params(m))
    end
end

function learn_transitions!(μβ0::MuBetaZeroNeural, env::Environment)
    for player in [1,2]
        X = Array{Float32}(undef, size(env.current)..., length(μβ0.transition_buffer[player]))
        Y_ps = Array{Float32}(undef, env.n_actions, length(μβ0.transition_buffer[player]))
        Y_vs = Array{Float32}(undef, 1, length(μβ0.transition_buffer[player]))
        for (i,t) in enumerate(μβ0.transition_buffer[player])
            X[:,:,:,i] = t.s
            Y_ps[:,i] = t.ps
            Y_vs = t.Q_est
        end

        Flux.Optimise
    end

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

            # if MCTS
            #     update!(μβ0, env, t.player, t.s, t.as, t.Q_ests)
            # else
            #     update!(μβ0, env, t.player, t.s, t.a, t.Q_est)
            # end
        end

        if verbose
            # i = s_a_to_index(env, t.s, 0, player)
            # println("Decision Stats: player: $player, Q_est: $(t.Q_est) vs Q: $(value(μβ0, env, t.s, player)), visits: $(μβ0.visits[i+t.a])/$(sum(μβ0.visits[i+1:i+10]))")
            # println("Q: ", μβ0.Q[i+1:i+10])
            println("player: $(t.player), action: $(t.a), winner: $winner, done: $done")
            print_current(env)
            # i = s_a_to_index(env, env.current, 0, nextplayer)
            # !done && println("State Stats: player: $nextplayer, Q = $(value(μβ0, env, env.current, nextplayer))", ", visits: ", sum(μβ0.visits[i+1:i+10]))
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
