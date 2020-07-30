using Random
using Plots
import ProgressMeter
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

function play!(μβ0::MuBetaZeroTabular, env::Environment,
               player::Int; ϵ_greedy=false, MCTS=false, N_MCTS=1000)
    # s .= env.current
    s = copy(env.current)
    Q_est = 0f0
    if MCTS
        best_node, Q_est, ps = MCTreeSearch(μβ0, env, N_MCTS, player) # nextstates already observed here
        a = best_node.action
        @assert best_node.player == player
        winner, done, nextplayer = step!(env, a, player)
        # @assert !isnan(Q_est) && !isinf(Q_est)
        μβ0.current_node = best_node
        remove_children!(best_node.parent, except=best_node)
    else
        if ϵ_greedy && rand() ≤ μβ0.ϵ
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
    end

    return s, a, winner, done, Q_est, player, nextplayer
end

function update!(μβ0::MuBetaZeroTabular, env::Environment, player::Int,
                 s::Vector{Int}, a::Int, Q_est::Float32)
    α = μβ0.α
    i = s_a_to_index(env, s, a, player)
    μβ0.visits[i] += 1
    μβ0.Q[i] = (1 - α) * μβ0.Q[i] + α * Q_est
end

function play_game!(μβ0::MuBetaZeroTabular, env::Environment;
                    verbose=false, train=false, MCTS=false, N_MCTS=1000)
    reset!(env)
    if MCTS # reset tree
        root = MCTSNode(0)
        expand!(root, env, 1)
        μβ0.tree = MCTSTree(root)
        μβ0.current_node = root
    end

    winner = 0
    done = false
    player = 1

    while !done
        s, a, winner, done, Q_est, player, nextplayer = play!(μβ0, env, player, ϵ_greedy=train, MCTS=MCTS, N_MCTS=N_MCTS)
        if train
            update!(μβ0, env, player, s, a, Q_est)
        end
        if verbose
            print_current(env)
            !done && println("player: $nextplayer, value = $(value(μβ0, env, env.current, nextplayer))", ", visits: ", μβ0.visits[s_a_to_index(env, s, a, player)])
            println()
        end

        player = nextplayer
    end

    return winner
end

function train!(μβ0::MuBetaZeroTabular, env::Environment,
                n_games::Int=10^6, success_threshold::Float64=0.55;
                MCTS=false)

    winners = zeros(Int, n_games)
    ProgressMeter.@showprogress for n in 1:n_games
        winners[n] = play_game!(μβ0, env, train=true, MCTS=MCTS)
    end

    rs = winners .* 2 .- 3
    cumsum(rs)
    return winners
end


# player for rewards independent of node.player
# node.n = sum(node.children.n) + expand_at
global DEBUG_MCTS = false
function MCTreeSearch(μβ0::MuBetaZero, env::Environment, N::Int, player::Int; expand_at=1)
    state = copy(env.current)
    # root = MCTSNode()
    # tree = MCTSTree(root)
    # expand!(root, env)
    root = μβ0.current_node
    # if isempty(root.children)
    #     expand!(root, env)
    # end
    @assert !isempty(root.children)
    @assert player == root.children[1].player

    n = 1
    while n ≤ N
        if DEBUG_MCTS && n % 10 == 0
            print_tree(root)
        end
        DEBUG_MCTS && println("=== n = $n ===")
        n += 1
        env.current .= state
        winner = 0
        DEBUG_MCTS && println("SELECT")
        best, nextplayer, winner, done = select!(env, root) # best action already applied, foe is next player
        if done
            DEBUG_MCTS && println("end node selected")
        end
        if !done
            if best.n ≥ expand_at
                DEBUG_MCTS && println("EXPAND")
                expand!(best, env, nextplayer)
                N += 1 # allow one more rollout
                continue
            else
                DEBUG_MCTS && println("ROLLOUT")
                winner = rollout!(μβ0, env, nextplayer)
            end
        end
        DEBUG_MCTS && println("BACKPROPAGATE winner: $winner")
        backpropagate!(best, winner)
    end

    env.current .= state

    scores = map(c -> v_mean(c), root.children)
    best = root.children[argmax(scores)]

    # calc action probabilities
    N = sum(c.n for c in root.children)
    ps = zeros(Float32, env.n_actions)
    i = 1
    for a in 1:env.n_actions
        node = root.children[i]
        if node.action == a
            ps[a] = node.n / N
            i += 1
            if i > length(root.children)
                break
            end
        end
    end

    return best, v_mean(best), ps
end

function select!(env::Environment, root::MCTSNode)
    # current = tree.root
    # foe = false
    winner = 0
    done = false
    nextplayer = 0
    current = root
    while length(current.children) != 0
        current = best_child(current)
        DEBUG_MCTS && print(current.action, " -> ")
        winner, done, nextplayer = step!(env, current.action, current.player)
        if done
            break
        end
    end
    DEBUG_MCTS && done && println("(winner: $winner)")
    return current, nextplayer, winner, done
end

function expand!(node::MCTSNode, env::Environment, nextplayer::Int)
    for a in valid_actions(env, env.current)
        push!(node, MCTSNode(nextplayer, a, node))
    end
end

function rollout!(μβ0::MuBetaZero, env::Environment, nextplayer::Int)
    winner = 0
    done = false

    while !done
        a = action(μβ0, env, env.current, nextplayer) # self play
        DEBUG_MCTS && print(" -> ", a)
        winner, done, nextplayer = step!(env, a, nextplayer)
    end
    DEBUG_MCTS && println(" -> winner: ", winner)
    return winner
end

function backpropagate!(node::MCTSNode, winner::Int)
    current = node
    while current != nothing
        current.n += 1
        if winner != 0
            current.w += Int(winner == current.player) * 2 - 1 # ∈ {-1, 1}
        end
        current = current.parent
    end
end

function play_against(agent::MuBetaZero, env::Environment, start=true)
    reset!(env)
    winner = 0
    done = false
    player = start ? 1 : 2
    player_adv = start ? 2 : 1
    if !start
        a = action(agent, env, env.current, player_adv)
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

        a, = action(agent, env, env.current, player_adv)
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
@profiler train!(agent, env, 10^3)

using BenchmarkTools
@btime train!(agent, env, 10^3)

reset!(env)

env.current = reshape([
    0 0 2;
    1 1 2;
    0 0 0
],:)
root = MCTSNode(0)
expand!(root, env, 1)
agent.tree = MCTSTree(root)
agent.current_node = root


Random.seed!(1)
best, v, ps = MCTreeSearch(agent, env, 1000, 1)
# remove_children!(best.parent, except=best)
agent.current_node = best
step!(env, best.action, best.player)
println_current(env)

print_tree(best)

play_game!(agent, env, train=false, verbose=true, MCTS=true)


agentMCTS = MuBetaZeroTabular(env.n_states, env.n_actions)
@time winners = train!(agentMCTS, env, 10^4, MCTS=true)
play_game!(agentMCTS, env, train=false, verbose=true)
play_game!(agentMCTS, env, train=false, verbose=true, MCTS=true)

using BenchmarkTools

@btime play_game!(agent, env, train=false, verbose=false, MCTS=true)
@btime play_game!(agent, env, train=false, verbose=false, MCTS=false)

# 8 μs ... 15s
# 20ms ...
