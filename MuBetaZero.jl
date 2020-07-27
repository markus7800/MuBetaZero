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

function greedy_action(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, player::Int)::Tuple{Int, Float32}
    i = s_a_to_index(env, state, 0, player)
    is = [i + j for j in 1:env.n_actions] # TODO: assertion
    a, Q = maximise(a -> μβ0.Q[is[a]], valid_actions(env, state))
    return a, Q
end

function V(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, player::Int)::Float32
    a, Q = greedy_action(μβ0, env, state, player)
    return Q
end

function Q(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int}, player::Int)::Vector{Float32}
    i = s_a_to_index(env, state, 0, player)
    is = [i + j for j in 1:env.n_actions]
    return μβ0.Q[is]
end

function play!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular,
               player::Int; ϵ_greedy=false)::Tuple{Vector{Int}, Int, Float32, Bool, Vector{Int}}
    # s .= env.current
    s = copy(env.current)
    if ϵ_greedy && rand() ≤ μβ0.ϵ
        a = rand(collect(valid_actions(env, s)))
    else
        a, = greedy_action(μβ0, env, s, player) # TODO: MCTS
    end

    r1, done, = step!(env, a, player, false)
    r2 = 0f0
    if !done
        player_adv = player == 1 ? 2 : 1
        a_adv, = greedy_action(adversary, env, env.current, player_adv)
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
        a_star, Q_star = greedy_action(μβ0, env, ns, player)
    else
        Q_star = 0f0
    end
    μβ0.Q[i] = (1 - α) * μβ0.Q[i] + α * (r + γ * Q_star)
end

function play_game!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular;
                    verbose=false, train=false, start=true)
    reset!(env)
    r = 0
    done = false
    player = start ? 1 : 2
    player_adv = start ? 2 : 1
    if !start
        a_adv, = greedy_action(adversary, env, env.current, player_adv)
        step!(env, a_adv, player_adv, true)
    end
    if verbose
        print_current(env)
        println("v_agent = $(Q(μβ0, env, env.current, player))")
        println("v_adversary = $(Q(adversary, env, env.current, player_adv))\n")
    end
    while !done
        s, a, r, done, ns = play!(μβ0, env, adversary, player, ϵ_greedy=train)
        if train
            update!(μβ0, env, player, s, a, r, done, ns)
        end
        if verbose
            println("r = $r\n")
            print_current(env)
            !done && println("v_agent = $(Q(μβ0, env, ns, player))")
            !done && println("v_adversary = $(Q(adversary, env, ns, player_adv))\n")
        end
    end

    return r
end

function train!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular,
                min_n_games::Int=10^4, max_n_games::Int=10^7, success_threshold::Float64=0.55)

    losses = 0
    n_games = 0
    while (n_games < min_n_games) ||
        (n_games < max_n_games && (n_games - losses) / n_games < success_threshold)

        n_games += 1
        start = n_games % 2 == 0
        r = play_game!(μβ0, env, adversary, train=true, start=start)
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

# TODO test MCTS without rollout
function MCTS(μβ0::MuBetaZero, env::Environment, N::Int; expand_at=1)
    state = copy(env.current)
    root = MCTSNode()
    tree = MCTSTree(root)
    expand!(root, env)

    n = 1
    while n ≤ N
        # println("=== n = $n ===")
        n += 1
        env.current .= state
        # println("SELECT")
        best, foe, r, done = select!(env, tree) # best action already applied, foe is next player
        if done
            # println("leaf node selected")
        end
        if !done
            if best.n ≥ expand_at
                # println("EXPAND")
                expand!(best, env)
                N += 1 # allow one more rollout
                continue
            else
                # println("ROLLOUT")
                r = rollout!(μβ0, env, foe)
            end
        end
        # println("BACKPROPAGATE")
        backpropagate!(best, r)
    end

    env.current .= state

    scores = map(c -> c.v/c.n, root.children)
    return root.children[argmax(scores)].action
end

function select!(env::Environment, tree::MCTSTree)
    current = tree.root
    foe = false
    r = 0.0f0
    done = false
    # print_env(env)
    while length(current.children) != 0
        current = best_child(current, foe)
        r, done, foe = step!(env, current.action, foe)
        # println(current.action, ", ", current.v, ", ", current.n)
        # print_env(env)
        if done
            break
        end
    end
    return current, foe, r, done
end

function expand!(node::MCTSNode, env::Environment)
    for a in valid_actions(env.current)
        push!(node.children, MCTSNode(node, a))
    end
end

function rollout!(μβ0::MuBetaZero, env::Environment, foe::Bool)
    r = 0.0f0
    done = false
    while !done
        action = greedy_action(μβ0, env, env.current) # self play
        r, done = step!(env, action, foe)
    end
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
        a, = greedy_action(agent, env, env.current, player_adv)
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

        a, = greedy_action(agent, env, env.current, player_adv)
        r, done = step!(env, a, player_adv, true)
    end

    println()
    println_current(env)

    println("Its a ", ["Loss", "Draw", "Win"][Int(r + 2)], "!")
end

reset!(env)
μβ0 = MuBetaZeroTabular(env.n_states, env.n_actions)
using Random
Random.seed!(1)
adv = MuBetaZeroTabular(env.n_states, env.n_actions, init=:random)
play_game!(μβ0, env, adv, verbose=true)

s_a_to_index(env, [2,2,2], 9, 2)
s_a_to_index(env, [0,0,0], 1, 2)

greedy_action(μβ0, env, env.current)

using BenchmarkTools

play_game!(μβ0, env, adv, verbose=true)

A = [
 0 0 2;
 0 1 0;
 0 0 2
]

env.current = reshape(A, :)
print_current(env)

MCTS(μβ0, env, 1_000_000)

env = TikTakToe()
μβ0 = MuBetaZeroTabular(env.n_states, env.n_actions)
using Random
Random.seed!(1)
adv = MuBetaZeroTabular(env.n_states, env.n_actions, init=:random)
train!(μβ0, env, adv, 10^4)


play_game!(μβ0, env, adv, verbose=true, train=false, start=true)

using Profile

@time train!(μβ0, env, adv, 1_000_000, 1.1)

println_current(env)

step!(env, 5, true)

println_current(env)

agent = iterate_tabular_agents(env)

play_against(agent, env, true)


env = TikTakToe()
μβ0 = MuBetaZeroTabular(env.n_states, env.n_actions)
adv = MuBetaZeroTabular(env.n_states, env.n_actions)
train!(μβ0, env, adv, 10^4)
play_game!(μβ0, env, adv, train=false, verbose=true)

adv = μβ0
μβ0 = MuBetaZeroTabular(env.n_states, env.n_actions)
train!(μβ0, env, adv, 10^4)
play_game!(μβ0, env, adv, train=false, verbose=true)

env = TikTakToe()
agent = MuBetaZeroTabular(env.n_states, env.n_actions)
train!(agent, env, agent, 10^6)
play_game!(agent, env, agent, train=false, verbose=true)
play_game!(agent, env, agent, train=false, verbose=true, start=false)

state = [
 1 1 0;
 2 0 0;
 0 0 0
]
state = reshape(state, :)

Q(agent, env, state, 2)

won(env.current)
