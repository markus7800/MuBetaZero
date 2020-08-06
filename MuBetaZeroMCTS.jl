using StatsBase

include("MuBetaZero.jl")
include("TikTakToe.jl")

mutable struct MuBetaZeroMCTS <: MuBetaZero
    N::Int
    tree::MCTSTree
    current_node::MCTSNode

    function MuBetaZeroMCTS(N::Int)
        this = new()
        this.N = N
        return this
    end
end

function action(μβ0::MuBetaZeroMCTS, env::Environment, state::Array, player::Int)::Int
    return sample(collect(valid_actions(env, state)))
end

function greedy_action(μβ0::MuBetaZeroMCTS, env::Environment, state::Array, player::Int)::Int
    reset_tree!(μβ0, player)
    if any(env.current .!== state)
        @warn "Current environment state not the same as given state"
    end

    best_node, Q_est, ps, ps_trunc, Q_ests = MCTreeSearch(μβ0, env, μβ0.N, player, type=:rollout)
    return best_node.action
end

function value(μβ0::MuBetaZeroMCTS, env::Environment, state::Array, action::Int, player::Int)::Float32
    return 0f0
end

function value(μβ0::MuBetaZeroMCTS, env::Environment, state::Array, player::Int)::Float32
    return 0f0
end

# adversary makes first move from state
function play_game!(μβ0::MuBetaZero, env::Environment, adversary::MuBetaZeroMCTS, state::Array, adv_player=1;
                    verbose=false, MCTS=false, N_MCTS=1000, MCTS_type=:rollout)

    env.current .= state
    player = [2,1][adv_player]

    if MCTS # reset tree
        reset_tree!(μβ0, adv_player)
    end

    winner = 0
    done = false

    while !done
        a = greedy_action(adversary, env, env.current, adv_player)
        winner, done, = step!(env, a, adv_player)
        if done
            println_current(env)
            break
        end

        t, winner, done, nextplayer = play!(μβ0, env, player, train=false, MCTS=MCTS, N_MCTS=N_MCTS, MCTS_type=MCTS_type)

        if verbose
            # l = value_loss(μβ0, t.player)(reshape(t.s, size(t.s)... ,1), reshape([t.Q_est],1,1))
            # println("Decision Stats: player: $(t.player), action: $(t.a), Q_est: $(t.Q_est) vs Q: $(value(μβ0, env, t.s, player)), loss: $l")
            # println(t.ps)
            # println(action_ps(μβ0, env, t.s, t.player))
            print_current(env)
            # !done && println("State Stats: player: $nextplayer, Q = $(value(μβ0, env, env.current, nextplayer))")
            println()
        end
    end

    return winner
end

# working stuff

env = ConnectFour()
agentMCTS = MuBetaZeroMCTS(10000)
reset!(env)
play_game!(agent, env, agentMCTS, env.current, 1, verbose=true)

play_against(agentMCTS, env)
