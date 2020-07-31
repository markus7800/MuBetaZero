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
    ps_trunc = zeros(Float32, length(root.children))
    i = 1
    for a in 1:env.n_actions
        node = root.children[i]
        if node.action == a
            ps[a] = node.n / N
            ps_trunc[i] = node.n / N
            i += 1
            if i > length(root.children)
                break
            end
        end
    end

    return best, v_mean(best), ps, ps_trunc, scores
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
