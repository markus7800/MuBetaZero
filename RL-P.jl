include("Tree.jl")

struct MuBetaZero
    Q::Vector{Float32}
    γ::Float32
end

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
        action = rand(valid_actions(env.current)) #greedy(μβ0, env.current)
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

μβ0 = MuBetaZero([], 1.0)
env = TikTakToe()

A = [
 0 0 2;
 0 1 0;
 0 0 2
]

env.current = reshape(A, :)
print_current(env)

MCTS(μβ0, env, 1_000_000)

tree = MCTSTree(MCTSNode())
push!(tree.root.children, MCTSNode(tree.root, 1))

println(tree)

leaf = tree.root.children[1]

print_tree(tree)
