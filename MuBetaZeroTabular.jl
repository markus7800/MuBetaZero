
include("MuBetaZero.jl")
include("TikTakToe.jl")

mutable struct MuBetaZeroTabular <: MuBetaZero
    Q::Vector{Float32}
    visits::Vector{Int}
    γ::Float32
    α::Float32
    ϵ::Float64
    tree::MCTSTree
    current_node::MCTSNode

    function MuBetaZeroTabular(n_states, n_actions; γ=0.99, α=0.1, ϵ=0.1, init=:zero)
        this = new()
        if init == :zero
            this.Q = zeros(Float32, n_states * n_actions * 2)
        elseif init == :random
            this.Q = rand(Float32, n_states * n_actions * 2) * 2 .- 1
        end
        this.visits= zeros(Int, n_states * n_actions * 2)
        this.γ = γ
        this.α = α
        this.ϵ = ϵ
        return this
    end
end

function greedy_action(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, player::Int)::Int
    return action(μβ0, env, state, player)
end

function action(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, player::Int)::Int
    i = s_a_to_index(env, state, 0, player)
    is = [i + j for j in 1:env.n_actions] # TODO: assertion
    a, Q = maximise(a -> μβ0.Q[is[a]], valid_actions(env, state))
    return a
end

function greedy_action(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, player::Int)::Int
    return action(μβ0, env, state, player)
end

function value(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, action::Int, player::Int)::Float32
    i = s_a_to_index(env, state, action, player)
    return μβ0.Q[i]
end

function value(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, player::Int)::Float32
    a = action(μβ0, env, state, player)
    i = s_a_to_index(env, state, a, player)
    return μβ0.Q[i]
end


function update!(μβ0::MuBetaZeroTabular, env::Environment, player::Int,
                 s::Vector{Int}, a::Int, Q_est::Float32)
    α = μβ0.α
    i = s_a_to_index(env, s, a, player)
    μβ0.visits[i] += 1
    μβ0.Q[i] = (1 - α) * μβ0.Q[i] + α * Q_est
end

function update!(μβ0::MuBetaZeroTabular, env::Environment, player::Int,
                 s::Vector{Int}, as::Vector{Int}, Q_ests::Vector{Float32})
    for (a, Q_est) in zip(as, Q_ests)
        update!(μβ0, env, player, s, a, Q_est)
    end
end


function play_game!(μβ0::MuBetaZeroTabular, env::Environment;
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
            if MCTS
                update!(μβ0, env, t.player, t.s, t.as, t.Q_ests)
            else
                update!(μβ0, env, t.player, t.s, t.a, t.Q_est)
            end
        end
        if verbose
            i = s_a_to_index(env, t.s, 0, player)
            println("Decision Stats: player: $player, Q_est: $(t.Q_est) vs Q: $(value(μβ0, env, t.s, player)), visits: $(μβ0.visits[i+t.a])/$(sum(μβ0.visits[i+1:i+10]))")
            println("Q: ", μβ0.Q[i+1:i+10])
            print_current(env)
            i = s_a_to_index(env, env.current, 0, nextplayer)
            !done && println("State Stats: player: $nextplayer, Q = $(value(μβ0, env, env.current, nextplayer))", ", visits: ", sum(μβ0.visits[i+1:i+10]))
            println()
        end

        player = nextplayer
    end

    return winner
end

function train!(μβ0::MuBetaZeroTabular, env::Environment,
                n_games::Int=10^6, success_threshold::Float64=0.55;
                MCTS=false, N_MCTS=1000, MCTS_type=:rollout)

    winners = zeros(Int, n_games)
    ProgressMeter.@showprogress for n in 1:n_games
        winners[n] = play_game!(μβ0, env, train=true, MCTS=MCTS, N_MCTS=N_MCTS, MCTS_type=MCTS_type)
    end

    return winners
end


# working stuff

env = TikTakToe()
agent = MuBetaZeroTabular(env.n_states, env.n_actions)
winners = train!(agent, env, 10^6)
play_game!(agent, env, train=false, verbose=true)

play_against(agent, env)


agentMCTS = MuBetaZeroTabular(env.n_states, env.n_actions)
@time winners = train!(agentMCTS, env, 10^5, MCTS=true, N_MCTS=100, MCTS_type=:rollout)
play_game!(agentMCTS, env, train=false, verbose=true)
play_game!(agentMCTS, env, train=false, verbose=true, MCTS=true)

play_against(agentMCTS, env, MCTS=false)



agentMCTS = MuBetaZeroTabular(env.n_states, env.n_actions)
@time winners = train!(agentMCTS, env, 10^5, MCTS=true, N_MCTS=100, MCTS_type=:value)
play_game!(agentMCTS, env, train=false, verbose=true)
play_game!(agentMCTS, env, train=false, verbose=true, MCTS=true)

play_against(agentMCTS, env, MCTS=true)
