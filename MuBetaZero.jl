
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
               player::Int; ϵ_greedy=false, MCTS=false)
    # s .= env.current
    s = copy(env.current)
    Q_est = 0f0
    if MCTS
        best_node = MCTreeSearch(μβ0, env, 1000, player) # nextstates already observed here
        a = best_node.action
        Q_est = v_mean(best)
        @assert !isnan(V) && !isinf(V)
        μβ0.current_node = best_node
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
                    verbose=false, train=false, MCTS=false)
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
        s, a, winner, done, Q_est, player, nextplayer = play!(μβ0, env, player, ϵ_greedy=train, MCTS=MCTS)
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
                n_games::Int=10^6, success_threshold::Float64=0.55,
                MCTS=false)

    winners = zeros(Int, n_games)
    for n in 1:n_games
        winners[n] = play_game!(μβ0, env, train=true, MCTS=MCTS)
    end

    rs = winners .* 2 .- 3
    cumsum(rs)
    return winners
end


# player for rewards independent of node.player
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

    n = 1
    while n ≤ N
        if n % 10 == 0
            print_tree(tree)
        end
        # println("=== n = $n ===")
        n += 1
        env.current .= state
        # println("SELECT")
        current, nextplayer, winner, done = select!(env, tree, player) # best action already applied, foe is next player
        if done
            # println("leaf node selected")
        end
        if !done
            if best.n ≥ expand_at
                # println("EXPAND")
                expand!(best, env, nextplayer)
                N += 1 # allow one more rollout
                continue
            else
                # println("ROLLOUT")
                r = rollout!(μβ0, env, nextplayer, foe)
            end
        end
        # println("BACKPROPAGATE $r")
        backpropagate!(best, r)
    end

    env.current .= state

    print_tree(root)

    scores = map(c -> c.v/c.n, root.children)
    best = root.children[argmax(scores)]
    return best
end

function select!(env::Environment, root::MCTSNode)
    # current = tree.root
    # foe = false
    winner = 0
    done = false
    next_player = 0
    while length(current.children) != 0
        current = best_child(current)
        # print(current.action, " -> ")
        winner, done, nextplayer = step!(env, current.action, current.player)
        if done
            break
        end
    end
    # println("(", done, " - ", r, ")")
    return current, nextplayer, winner, done
end

function expand!(node::MCTSNode, env::Environment, nextplayer::Int)
    for a in valid_actions(env, env.current)
        push!(node, MCTSNode(nextplayer, node, a))
    end
end

function rollout!(μβ0::MuBetaZero, env::Environment, nextplayer::Int)
    winner = 0
    done = false

    while !done
        a = action(μβ0, env, env.current, nextplayer) # self play
        # print(" -> ", a)
        winner, done, nextplayer = step!(env, a, nextplayer)
    end
    # println(" -> ", r)
    return winner
end

function backpropagate!(node::MCTSNode, winner::Int)
    current = node
    while current != nothing
        current.n += 1
        current.w += Int(winner == current.player)
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

reset!(env)
μβ0 = MuBetaZeroTabular(env.n_states, env.n_actions)
using Random
Random.seed!(1)
adv = MuBetaZeroTabular(env.n_states, env.n_actions, init=:random)
play_game!(μβ0, env, adv, verbose=true)


env = TikTakToe()
agent = MuBetaZeroTabular(env.n_states, env.n_actions)
winners = train!(agent, env, 10^6)
play_game!(agent, env, train=false, verbose=true)

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

println_current(env)

rollout!(agent, env, 1, false)

Random.seed!(1)

root = MCTSNode(0)
expand!(root, env, 1)
agent.tree = MCTSTree(root)
agent.current_node = root
MCTreeSearch(agent, env, 100, 1)
