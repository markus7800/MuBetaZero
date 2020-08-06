# player for rewards independent of node.player
# node.n = sum(node.children.n) + expand_at
global DEBUG_MCTS = false
function MCTreeSearch(μβ0::MuBetaZero, env::Environment, N::Int, player::Int; expand_at=1, type=:rollout)
    state = copy(env.current)
    root = μβ0.current_node
    @assert !isempty(root.children)
    @assert player == root.children[1].player

    n = 1
    while n ≤ N
        if DEBUG_MCTS && n % 10 == 0
            print_tree(root)
        end
        DEBUG_MCTS && println("========== n = $n ==========")
        n += 1
        env.current .= state
        winner = 0
        DEBUG_MCTS && println("SELECT")
        best, nextplayer, winner, done = select!(env, root) # actions up to best action already applied
        if done
            DEBUG_MCTS && println("end node selected")
            DEBUG_MCTS && println("BACKPROPAGATE winner: $winner")
            backpropagate!(best, winner)
        else
            if best.n ≥ expand_at
                DEBUG_MCTS && println("EXPAND")
                expand!(best, env, nextplayer)
                N += 1 # allow one more rollout/update for child of best
                continue
            else
                if type == :rollout
                    DEBUG_MCTS && println("ROLLOUT")
                    winner = rollout!(μβ0, env, nextplayer)
                    DEBUG_MCTS && println("BACKPROPAGATE winner: $winner")
                    backpropagate!(best, winner)
                elseif type == :value
                    # env.current is state after best.action was applied
                    val = value(μβ0, env, env.current, nextplayer)
                    @assert !isnan(val)
                    DEBUG_MCTS && println("BACKPROPAGATE value: $val")
                    backpropagate!(best, -val)
                end
            end
        end
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
    DEBUG_MCTS && !done && println()
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

# value from perspective of node.player
function backpropagate!(node::MCTSNode, value::Float32)
    current = node
    while current != nothing
        current.n += 1
        current.w += (Float32(node.player == current.player) * 2 - 1) * value # ∈ {-1, 1} * value
        current = current.parent
    end
end
