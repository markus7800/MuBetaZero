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

function greedy_action(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int})::Tuple{Int, Float32}
    i = s_a_to_index(env, state, 0)
    is = [i + j for j in 1:env.n_actions] # TODO: assertion
    a, Q = maximise(a -> μβ0.Q[is[a]], valid_actions(env, state))
    return a, Q
end

function V(μβ0::MuBetaZeroTabular, env::Environment, state::Vector{Int})
    a, Q = greedy_action(μβ0, env, state)::Float32
    return Q
end

function play!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular;
               ϵ_greedy=false)::Tuple{Vector{Int}, Vector{Int}, Float32, Bool}
    # s .= env.current
    s = copy(env.current)
    if ϵ_greedy && rand() ≤ μβ0.ϵ
        a = rand(collect(valid_actions(env, s)))
    else
        a, = greedy_action(μβ0, env, s) # TODO: MCTS
    end

    r1, done, = step!(env, a, false)
    r2 = 0f0
    if !done
        a_adv, = greedy_action(adversary, env, env.current)
        r2, done = step!(env, a_adv, true)
    end
    r = r1 + r2
    # ns .= env.current
    ns = copy(env.current)

    return s, a, r, done, ns
end

function update!(μβ0::MuBetaZeroTabular, env::Environment,
                 s::Vector{Int}, a::Int, r::Float32, done::Bool, ns::Vector{Int})
    α = μβ0.α; γ = μβ0.γ

    i = s_a_to_index(env, s, a)
    if !done
        a_star, Q_star = greedy_action(μβ0, env, ns)
    else
        Q_star = 0f0
    end
    μβ0.Q[i] = (1 - α) * μβ0.Q[i] + α * (r + γ * Q_star)
end

function play_game!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular;
                    verbose=false, train=false)
    reset!(env)
    done = false
    verbose && print_current(env)
    while !done
        s, a, r, done, ns = play!(μβ0, env, adversary, ϵ_greedy=train)
        if train
            update!(μβ0, env, s, a, r, done, ns)
        end
        if verbose
            println("r = $r\n")
            print_current(env)
            println("v = $(V(μβ0, env, ns))\n")
        end
    end
end

function train!(μβ0::MuBetaZeroTabular, env::Environment, adversary::MuBetaZeroTabular,
                n_games::Int)
    for n in 1:n_games
        play_game!(μβ0, env, adversary, train=true)
    end
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

reset!(env)
μβ0 = MuBetaZeroTabular(env.n_states, env.n_actions)
using Random
Random.seed!(1)
adv = MuBetaZeroTabular(env.n_states, env.n_actions, init=:random)
play_game!(μβ0, env, adv, verbose=true)

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
@btime train!(μβ0, env, adv, 1_000)

println(μβ0.Q[1:9])

play_game!(μβ0, env, adv, verbose=true, train=false)

using Profile

@profiler train!(μβ0, env, adv, 1_000)
