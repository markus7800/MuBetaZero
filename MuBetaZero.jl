using Random
using Plots
import ProgressMeter
using StatsBase
include("Tree.jl")
include("utils.jl")
include("TikTakToe.jl")

abstract type MuBetaZero end

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

function action(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, player::Int)::Int
    i = s_a_to_index(env, state, 0, player)
    is = [i + j for j in 1:env.n_actions] # TODO: assertion
    a, Q = maximise(a -> μβ0.Q[is[a]], valid_actions(env, state))
    return a
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


function play!(μβ0::MuBetaZero, env::Environment,
               player::Int; train=false, MCTS=false, N_MCTS=1000)
    # s .= env.current
    s = copy(env.current)
    Q_est = 0f0
    if MCTS
        best_node, Q_est, ps, ps_trunc, Q_ests = MCTreeSearch(μβ0, env, N_MCTS, player) # nextstates already observed here
        as = collect(valid_actions(env, s))
        if train
            chosen_node = sample(μβ0.current_node.children, Weights(ps_trunc))
        else
            chosen_node = best_node
        end
        a = chosen_node.action
        @assert chosen_node.player == player
        winner, done, nextplayer = step!(env, a, player)
        μβ0.current_node = chosen_node
        remove_children!(chosen_node.parent, except=chosen_node)

        return s, a, Q_est, winner, done, as, Q_ests, player, nextplayer
    else
        if train && rand() ≤ μβ0.ϵ
            a = rand(collect(valid_actions(env, s)))
        else
            a = action(μβ0, env, s, player) # greedy action
            i = s_a_to_index(env, s, 0, player)
        end
        winner, done, nextplayer = step!(env, a, player)

        # MCTS delivers estimate Q_est, for !MCTS we have to look at nextstate
        if done
            # here winner == player possible or draw possible
            @assert winner == 0 || winner == player
            Q_est = Float32(winner == player) # ∈ {0, 1}
        else
            # kind of one interation MCTS
            ns = copy(env.current)
            a_adv = action(μβ0, env, ns, nextplayer)
            winner, nextdone,  = step!(env, a_adv, nextplayer)
            if nextdone
                # here winner == nextplayer or draw possible
                @assert winner == 0 || winner == nextplayer
                Q_est = μβ0.γ * -Float32(winner == nextplayer) # ∈ {-1, 0}
            else
                Q_est = μβ0.γ * value(μβ0, env, env.current, player) # nextstate
            end
            env.current = ns
        end

        return s, a, Q_est, winner, done, [], [], player, nextplayer
    end
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

function reset_tree!(μβ0::MuBetaZero)
    root = MCTSNode(0)
    expand!(root, env, 1)
    μβ0.tree = MCTSTree(root)
    μβ0.current_node = root
end


include("MCTS.jl")

function play_game!(μβ0::MuBetaZeroTabular, env::Environment;
                    verbose=false, train=false, MCTS=false, N_MCTS=1000)
    reset!(env)
    if MCTS # reset tree
        reset_tree!(μβ0)
    end

    winner = 0
    done = false
    player = 1

    while !done
        s, a, Q_est, winner, done, as, Q_ests, player, nextplayer = play!(μβ0, env, player, train=train, MCTS=MCTS, N_MCTS=N_MCTS)
        if train
            if MCTS
                update!(μβ0, env, player, s, as, Q_ests)
            else
                update!(μβ0, env, player, s, a, Q_est)
            end
        end
        if verbose
            i = s_a_to_index(env, s, 0, player)
            println("Decision Stats: player: $player, Q_est: $Q_est vs Q: $(value(μβ0, env, s, player)), visits: $(μβ0.visits[i+a])/$(sum(μβ0.visits[i+1:i+10]))")
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
                MCTS=false, N_MCTS=1000)

    winners = zeros(Int, n_games)
    ProgressMeter.@showprogress for n in 1:n_games
        winners[n] = play_game!(μβ0, env, train=true, MCTS=MCTS, N_MCTS=N_MCTS)
    end

    rs = winners .* 2 .- 3
    cumsum(rs)
    return winners
end




function play_against(agent::MuBetaZero, env::Environment;
    start=true, MCTS=false, N_MCTS=1000, thinktime=0.5)
    reset!(env)
    winner = 0
    done = false
    player = start ? 1 : 2
    player_adv = start ? 2 : 1

    if MCTS
        reset_tree!(agent)
    end

    if !start
        if MCTS
            tik = time()
            best, = MCTreeSearch(agent, env, N_MCTS, player_adv)
            tak = time()
            while tak - tik < thinktime
                best, = MCTreeSearch(agent, env, N_MCTS, player_adv)
                tak = time()
            end
            a = best.action
            agent.current_node = best
            remove_children!(best.parent, except=best)
        else
            a = action(agent, env, env.current, player_adv)
        end
        step!(env, a, player_adv)
    end

    while !done
        println()
        println_current(env)
        print("Input action: ")
        a = parse(Int, readline())
        while !(a in collect(valid_actions(env)))
            print("\nInput valid action:")
            a = parse(Int, readline())
        end
        winner, done, = step!(env, a, player)
        if done
            break
        end

        if MCTS
            chosen_node = nothing
            for c in agent.current_node.children
                if c.action == a
                    chosen_node = c
                    break
                end
            end
            agent.current_node = chosen_node
            remove_children!(chosen_node.parent, except=chosen_node)
            expand!(chosen_node, env, player_adv)

            tik = time()
            best, = MCTreeSearch(agent, env, N_MCTS, player_adv)
            tak = time()
            n = 1
            while tak - tik < thinktime
                best, = MCTreeSearch(agent, env, N_MCTS, player_adv)
                n += 1
                tak = time()
            end
            a = best.action
            agent.current_node = best
            remove_children!(best.parent, except=best)
            t = round((tak - tik) * 1000) / 1000
            println("Did $(n * N_MCTS) simulations in $t s.)")
        else
            a = action(agent, env, env.current, player_adv)
        end

        winner, done = step!(env, a, player_adv)
    end

    println()
    println_current(env)

    if winner == 0
        println("Its a Draw!")
    else
        println("Its a ", ["Loss", "Win"][Int(winner == player) + 1], "!")
    end
    println()

    print("Rematch?\n(Y or enter):")
    ans = readline()
    if ans in ["y", "Y", "yes", "Yes", ""]
        play_against(agent, env, !start)
    end
end



env = TikTakToe()
agent = MuBetaZeroTabular(env.n_states, env.n_actions)
winners = train!(agent, env, 10^6)
play_game!(agent, env, train=false, verbose=true)

scatter(winners[end-1000: end])

play_against(agent, env, true)

import Profile
Profile.clear()
Profile.init()
@profiler train!(agent, env, 10^3, MCTS=true)
@profiler play_game!(agent, env, MCTS=true)

using BenchmarkTools
@btime train!(agent, env, 10^3)

reset!(env)

env.current = reshape([
    0 0 2;
    1 1 2;
    0 0 0
],:)
# env.current = reshape([
#     2 0 0;
#     0 1 1;
#     0 2 0
# ],:)
root = MCTSNode(0)
expand!(root, env, 1)
agent.tree = MCTSTree(root)
agent.current_node = root


Random.seed!(1)
best, v, ps, ps_trunc, scores = MCTreeSearch(agent, env, 1000, 1)
# remove_children!(best.parent, except=best)
agent.current_node = best
step!(env, best.action, best.player)
println_current(env)

print_tree(best)
print_children(best.parent)

play_game!(agent, env, train=false, verbose=true, MCTS=true)


agentMCTS = MuBetaZeroTabular(env.n_states, env.n_actions)
@time winners = train!(agentMCTS, env, 10^5, MCTS=true, N_MCTS=100)
play_game!(agentMCTS, env, train=false, verbose=true)
play_game!(agentMCTS, env, train=false, verbose=true, MCTS=true)

play_against(agent, env, MCTS=false)

using BenchmarkTools

@btime play_game!(agent, env, train=false, verbose=false, MCTS=true)
@btime play_game!(agent, env, train=false, verbose=false, MCTS=false)

@time play_game!(agent, env, train=false, verbose=false, MCTS=true)

# 8 μs ... 15s
# 20ms ...
