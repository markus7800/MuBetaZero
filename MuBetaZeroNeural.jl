include("MuBetaZero.jl")
include("ConnectFour.jl")

using Random
using Flux
using StatsBase
using Random

using CUDA
CUDA.allowscalar(false)

mutable struct MuBetaZeroNeural <: MuBetaZero

    policy_networks
    value_networks

    transition_buffer::Array{Vector{Transition}}
    γ::Float32
    ϵ::Float64
    c::Float32 # regularisation parameter
    opt

    tree::MCTSTree
    current_node::MCTSNode

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

function flush_transition_buffer!(μβ0::MuBetaZeroNeural)
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

function action_ps(μβ0::MuBetaZeroNeural, env::Environment, state::Array{Float32}, player::Int)::Vector{Float32}
    x = reshape(state, size(state)..., 1)
    ps = μβ0.policy_networks[player](x)[:,1]
    return ps
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

function learn_transitions!(μβ0::MuBetaZeroNeural, env::Environment, n_trans::Int, batchsize::Int; gpu_enabled=true)
    ls = (0f0, 0f0)

    if gpu_enabled
        μβ0.value_networks = gpu.(μβ0.value_networks)
    end

    for player in [1,2]
        @assert n_trans > batchsize

        X = Array{Float32}(undef, size(env.current)..., n_trans)
        Y_vs = Array{Float32}(undef, 1, n_trans)

        L = length(μβ0.transition_buffer[player])
        is = sample(1:L, min(n_trans,L), replace=false)
        for (i,t) in enumerate(μβ0.transition_buffer[player][is])
            X[:,:,:,i] = t.s
            Y_vs[1,i] = t.Q_est
        end

        if gpu_enabled
            X = gpu(X)
            Y_vs = gpu(Y_vs)
        end

        data = Flux.Data.DataLoader((X, Y_vs))

        value_loss_f = value_loss(μβ0, player, μβ0.c)
        ls[player] = value_loss_f

        Flux.Optimise.train!(value_loss_f, params(μβ0.value_networks[player]), data, μβ0.opt)

        v_loss = value_loss_f(X, Y_vs)
        push!(ls[player], v_loss)

        println("player $player, loss: $v_loss")
    end

    if gpu_enabled
        μβ0.value_networks = cpu.(μβ0.value_networks)
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

    while !done
        t, winner, done, nextplayer = play!(μβ0, env, player, train=train, MCTS=MCTS, N_MCTS=N_MCTS, MCTS_type=MCTS_type)

        if train
            push!(μβ0.transition_buffer[t.player], t)
        end

        if verbose
            l = value_loss(μβ0, t.player)(reshape(t.s, size(t.s)... ,1), reshape([t.Q_est],1,1))
            println("Decision Stats: player: $(t.player), action: $(t.a), Q_est: $(t.Q_est) vs Q: $(value(μβ0, env, t.s, player)), loss: $l")
            println(t.ps)
            println(action_ps(μβ0, env, t.s, t.player))
            print_current(env)
            !done && println("State Stats: player: $nextplayer, Q = $(value(μβ0, env, env.current, nextplayer))")
            println()
        end

        player = nextplayer
    end

    return winner
end

function train!(μβ0::MuBetaZeroNeural, env::Environment;
                n_games::Int=10^5, batchsize=128, buffersize=10^5, n_trans=10^4, learn_interval=10^4,
                MCTS=true, N_MCTS=100, MCTS_type=:value, gpu_enabled=true)

    println("Begin training:")
    println("Number of games: $n_games")
    println("batchsize: $batchsize, number of transitions per epoch: $n_trans")
    println("Learning every $learn_interval games")
    println("MCTS: $MCTS with $N_MCTS evaluations of $MCTS_type type")
    println("GPU ", gpu_enabled ? "enabled" : "disabled")

    winners = zeros(Int, n_games)
    v_losses = [[], []]

    ProgressMeter.@showprogress for n in 1:n_games
        winners[n] = play_game!(μβ0, env, train=true, MCTS=MCTS, N_MCTS=N_MCTS, MCTS_type=MCTS_type)

        if n % learn_interval == 0
           ls = learn_transitions!(μβ0, env, n_trans, batchsize, gpu_enabled=gpu_enabled)
           push!(v_losses[1], ls[1])
           push!(v_losses[2], ls[2])
        end

        L = minimum(length.(μβ0.transition_buffer))
        if L > buffersize
            println("Buffer full: $L")
            is = sample(1:L, L - L÷5, replace=false) # remove 25%
            μβ0.transition_buffer[1] = μβ0.transition_buffer[1][is]
            μβ0.transition_buffer[2] = μβ0.transition_buffer[2][is]
            L = minimum(length.(μβ0.transition_buffer))
            println("Buffer now: $L")
        end
    end

    return winners, v_losses
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
Random.seed!(1)
winners, v_losses = train!(agent, env, n_games=10^4, n_trans=10^4, learn_interval=10^3)

for (i, t) in enumerate(agent.transition_buffer[2])
    if isnan(t.Q_est)
        println("$i is nan, ")
    end
end

using Plots
plot(v_losses[1])




s = reshape(reset!(env), size(env.current)..., 1)
ps = agent.policy_networks[1](s)
bar(ps)

m = agent.value_networks[2]
BSON.@save "value_nn_2.bson" m

Random.seed!(1)
play_game!(agent, env, verbose=true, train=false, MCTS=true, N_MCTS=100, MCTS_type=:value)

import BenchmarkTools

BenchmarkTools.@btime play_game!(agent, env, verbose=false, train=false, MCTS=false, N_MCTS=100, MCTS_type=:value)

play_against(agent, env, MCTS=true, N_MCTS=1000, MCTS_type=:value, thinktime=0.5)

learn_transitions!(agent, env)

t = rand(agent.transition_buffer[1])

value_loss(agent, t.player)(reshape(t.s, size(t.s)..., 1), reshape([t.Q_est], 1, 1))


reset!(env)
reset_tree!(agent)
@time MCTreeSearch(agent, env, 10^7, 1)



reset!(env)
step!(env, 3, 1)
println_current(env)
step!(env, 3, 2)
println_current(env)

s = copy(env.current)

sum(s[:,:,2])

agent.current_node = MCTSNode(0)
expand!(agent.current_node, env, 2)
MCTreeSearch(agent, env, 1000, 2)

s = gpu(reshape(env.current, size(env.current)..., 1))

agent.value_networks[1](s)

agent.value_networks[1](reshape(env.current, size(env.current)..., 1))

value(agent, env, env.current, 1)



reset_tree!(agent)
flush_transition_buffer!(agent)

ProgressMeter.@showprogress for n in 1:10^4
    play_game!(agent, env, train=true, MCTS=true, N_MCTS=100, MCTS_type=:value)
end

length(agent.transition_buffer[1])

@time learn_transitions!(agent, env, 25)

agent.opt = ADAM()

Random.seed!(1)
value_model2 = Chain(
    Conv((3,3), 2 => 32, relu), # (4,5,32,:)
    Conv((2,2), 32 => 32, relu), # (3,4,32,:)
    Flux.flatten, # (384,:)
    Dense(384, 162, relu),
    Dense(162, 81, relu),
    Dense(81, 1)
)

agent2 = MuBetaZeroNeural(policy_model, value_model2)

train!(agent2, env)


@time learn_transitions!(agent2, env, 10^4, 128, gpu=false)

value_model2(rand(Float32, 6, 7, 2, 1))

X = rand(Float32, 6,7,2, 128)

@btime value_model(X)

m = gpu(value_model)

@btime m( X |> gpu)

Y = X |> gpu

@btime m(Y)


cpu_times = []
gpu_times = []
m = gpu(value_model2)

@progress for b in [2^i for i in 1:10]
    X = rand(Float32, 6,7,2, b)
    v,t, = @timed value_model2(X)
    push!(cpu_times,t)
    v,t, = @timed m(X |> gpu)
    push!(gpu_times, t)
end

using Plots

plot(cpu_times, label="cpu", yaxis=:log)
plot!(gpu_times, label="gpu")
