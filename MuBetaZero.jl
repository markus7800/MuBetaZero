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
            this.Q = zeros(Float32, n_states * n_actions)
        elseif init == :random
            this.Q = rand(Float32, n_states * n_actions) * 2 .- 1
        end
        this.γ = γ
        this.α = α
        this.ϵ = ϵ
        return this
    end
end

function greedy_action(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int})
    i = s_a_to_index(env, state, 0)
    is = [i + j for j in 1:env.n_actions] # TODO: assertion
    a, Q = maximise(a -> μβ0.Q[is[a]], valid_actions(env, state))
    return a, Q
end

# TODO: split from update
function play!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular)
    s = env.current

    if rand() ≤ μβ0.ϵ
        a = rand(valid_actions(env, s))
    else
        a, = greedy_action(μβ0, env, s) # TODO: MCTS
    end

    i = s_a_to_index(env, s, a)

    print_current(env)
    r1, done, = step!(env, a, false)
    print_current(env)
    r2 = 0f0
    if !done
        a_adv, = greedy_action(adversary, env, env.current)
        r2, done = step!(env, a_adv, true)
        print_current(env)
    end
    r = r1 + r2

    α = μβ0.α; γ = μβ0.γ
    println(r, ", ", done)

    if !done
        a_star, Q_star = greedy_action(μβ0, env, env.current)
    else
        Q_star = 0f0
    end

    # μβ0.Q[i] = (1 - α) * μβ0.Q[i] + α * (r + γ * Q_star)

    return done
end

function play_game!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular)
    reset!(env)
    done = false
    while !done
        done = play!(μβ0, env, adversary)
    end
end

# function update!(μβ0::MuBetaZeroTabular, env::Environment, s::Int, a::Int, r::Float32)

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

env = TikTakToe()
reset!(env)
μβ0 = MuBetaZeroTabular(env.n_states, env.n_actions)
using Random
Random.seed!(1)
adv = MuBetaZeroTabular(env.n_states, env.n_actions, init=:random)

greedy_action(μβ0, env, env.current)

play_game!(μβ0, env, adv)

A = [
 0 0 2;
 0 1 0;
 0 0 2
]

env.current = reshape(A, :)
print_current(env)

MCTS(μβ0, env, 1_000_000)
