import ProgressMeter

include("Tree.jl")
include("utils.jl")
include("TikTakToe.jl")

abstract type MuBetaZero end

mutable struct MuBetaZeroTabular <: MuBetaZero
    Q::Vector{Float32}
    γ::Float32
    α::Float32
    ϵ::Float64
    tree::MCTSTree

    function MuBetaZeroTabular(n_states, n_actions; γ=0.99, α=0.1, ϵ=0.1, init=:zero)
        this = new()
        if init == :zero
            this.Q = zeros(Float32, n_states * n_actions * 2)
        elseif init == :random
            this.Q = rand(Float32, n_states * n_actions * 2) * 2 .- 1
        end
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

function value(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, player::Int)::Float32
    i = s_a_to_index(env, state, 0, player)
    is = [i + j for j in 1:env.n_actions] # TODO: assertion, make range
    a, Q = maximise(a -> μβ0.Q[is[a]], valid_actions(env, state))
    return Q
end


function play!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular,
               player::Int; ϵ_greedy=false, MCTS=false)::Tuple{Vector{Int}, Int, Float32, Bool, Vector{Int}}
    # s .= env.current
    s = copy(env.current)
    if ϵ_greedy && rand() ≤ μβ0.ϵ
        a = rand(collect(valid_actions(env, s)))
    else
        if MCTS
            a = MCTreeSearch(μβ0, env, 1000)
        else
            a = action(μβ0, env, s, player) # greedy action
        end
    end

    r1, done, = step!(env, a, player, false)
    r2 = 0f0
    if !done
        player_adv = player == 1 ? 2 : 1
        a_adv = action(adversary, env, env.current, player_adv)
        r2, done = step!(env, a_adv, player_adv, true)
    end
    r = r1 + r2
    # ns .= env.current
    ns = copy(env.current)

    return s, a, r, done, ns
end

function update!(μβ0::MuBetaZeroTabular, env::Environment, player::Int,
                 s::Vector{Int}, a::Int, r::Float32, done::Bool, ns::Vector{Int})
    α = μβ0.α; γ = μβ0.γ

    i = s_a_to_index(env, s, a, player)
    if !done
        a_star = action(μβ0, env, ns, player)
        Q_star = value(μβ0, env, ns, player)
    else
        Q_star = 0f0
    end
    μβ0.Q[i] = (1 - α) * μβ0.Q[i] + α * (r + γ * Q_star)
end

function play_game!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular;
                    verbose=false, train=false, start=true, MCTS=false)
    reset!(env)
    # if MCTS
    #     μβ0.tree = MCTSTree(MCTSNode())
    # end
    #
    r = 0
    done = false
    player = start ? 1 : 2
    player_adv = start ? 2 : 1
    if !start
        a_adv = action(adversary, env, env.current, player_adv)
        step!(env, a_adv, player_adv, true)
    end
    if verbose
        print_current(env)
        println("v_agent = $(value(μβ0, env, env.current, player))")
        println("v_adversary = $(value(adversary, env, env.current, player_adv))\n")
    end
    while !done
        s, a, r, done, ns = play!(μβ0, env, adversary, player, ϵ_greedy=train, MCTS=MCTS)
        if train
            update!(μβ0, env, player, s, a, r, done, ns)
        end
        if verbose
            println("r = $r\n")
            print_current(env)
            !done && println("v_agent = $(value(μβ0, env, ns, player))")
            !done && println("v_adversary = $(value(adversary, env, ns, player_adv))\n")
        end
    end

    return r
end

function train!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular,
                min_n_games::Int=10^4, max_n_games::Int=10^7, success_threshold::Float64=0.55,
                MCTS=false)

    losses = 0
    n_games = 0
    while (n_games < min_n_games) ||
        (n_games < max_n_games && (n_games - losses) / n_games < success_threshold)

        n_games += 1
        start = n_games % 2 == 0
        r = play_game!(μβ0, env, adversary, train=true, start=start, MCTS=MCTS)
        if r < 0
            losses += 1
        end
    end
    @info "Total number of iterations: $(n_games), success_ratio = $(1 - losses/n_games)"
end

function iterate_tabular_agents(env::Environment; γ=0.99f0, ϵ=0.1f0, α=0.01f0, n_iter=100)
    adversary = MuBetaZeroTabular(env.n_states, env.n_actions)
    agent = MuBetaZeroTabular(env.n_states, env.n_actions, γ=γ, ϵ=ϵ, α=α)

    @progress for n in 1:n_iter
        agent = MuBetaZeroTabular(env.n_states, env.n_actions, γ=γ, ϵ=ϵ, α=α)
        train!(agent, env, adversary)
        adversary = agent
    end

    return agent
end

function MCTreeSearch(μβ0::MuBetaZero, env::Environment, N::Int, player::Int; expand_at=1)
    state = copy(env.current)
    root = MCTSNode()
    tree = MCTSTree(root)
    expand!(root, env)

    n = 1
    while n ≤ N
        if n % 10 == 0
            print_tree(tree)
        end
        # println("=== n = $n ===")
        n += 1
        env.current .= state
        println("SELECT")
        best, next_player, foe, r, done = select!(env, tree, player) # best action already applied, foe is next player
        if done
            # println("leaf node selected")
        end
        if !done
            if best.n ≥ expand_at
                println("EXPAND")
                expand!(best, env)
                N += 1 # allow one more rollout
                continue
            else
                println("ROLLOUT")
                r = rollout!(μβ0, env, next_player, foe)
            end
        end
        println("BACKPROPAGATE $r")
        backpropagate!(best, r)
    end

    env.current .= state

    print_tree(tree)

    scores = map(c -> c.v/c.n, root.children)
    return root.children[argmax(scores)].action
end

function select!(env::Environment, tree::MCTSTree, player::Int)
    current = tree.root
    foe = false
    r = 0.0f0
    done = false
    next_player = player
    while length(current.children) != 0
        current = best_child(current, foe)
        print(current.action, " -> ")
        r, done, next_player, foe = step!(env, current.action, next_player, foe)
        if done
            break
        end
    end
    println("(", done, " - ", r, ")")
    return current, next_player, foe, r, done
end

function expand!(node::MCTSNode, env::Environment)
    for a in valid_actions(env, env.current)
        push!(node.children, MCTSNode(node, a))
    end
end

function rollout!(μβ0::MuBetaZero, env::Environment, nextplayer::Int, foe::Bool)
    r = 0.0f0
    done = false

    while !done
        a = action(μβ0, env, env.current, nextplayer) # self play
        print(" -> ", a)
        r, done, nextplayer, foe = step!(env, a, nextplayer, foe)
    end
    println(" -> ", r)
    return r
end

function backpropagate!(node::MCTSNode, r::Float32)
    current = node
    while current != nothing
        current.n += 1
        current.v += r
        current = current.parent
    end
end

function play_against(agent::MuBetaZero, env::Environment, start=true)
    reset!(env)
    r = 0f0
    done = false
    player = start ? 1 : 2
    player_adv = start ? 2 : 1
    if !start
        a = action(agent, env, env.current, player_adv)
        step!(env, a, player_adv, true)
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
        r, done, = step!(env, a, player, false)
        if done
            break
        end

        a, = action(agent, env, env.current, player_adv)
        r, done = step!(env, a, player_adv, true)
    end

    println()
    println_current(env)

    println("Its a ", ["Loss", "Draw", "Win"][Int(r + 2)], "!")
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
train!(agent, env, agent, 10^6)
play_game!(agent, env, agent, train=false, verbose=true)
play_game!(agent, env, agent, train=false, verbose=true, start=false)

play_against(agent, env, false)




reset!(env)

env.current = reshape([
    0 0 2;
    1 1 2;
    0 0 0
],:)

println_current(env)

rollout!(agent, env, 1, false)

Random.seed!(1)
MCTreeSearch(agent, env, 100, 1)
