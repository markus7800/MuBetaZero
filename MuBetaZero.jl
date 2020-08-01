using Random
import ProgressMeter
using StatsBase
include("Tree.jl")
include("utils.jl")

abstract type Environment end
abstract type MuBetaZero end

include("MCTS.jl")

struct Transition{T}
    s::Array{T}                 # state
    a::Int                      # chosen action
    Q_est::Float32              # estimated value of selected action
    Q_ests::Vector{Float32}     # estimated values of valid actions
    ps::Vector{Float32}         # estimated probabilities for all actions
    player::Int                 # player selecting action a

    function Transition(s::Array{T}, a::Int, Q_est::Float32, player::Int,
                        Q_ests::Vector{Float32}=Float32[], ps::Vector{Float32}=Float32[]) where T <: Real
        return new{T}(s, a, Q_est, Q_ests, ps, player)
    end
end

function play!(μβ0::MuBetaZero, env::Environment,
               player::Int; train=false, MCTS=false, N_MCTS=1000, MCTS_type=:rollout)::Tuple{Transition, Int, Bool, Int}
    # s .= env.current
    s = copy(env.current)
    Q_est = 0f0
    if MCTS
        best_node, Q_est, ps, ps_trunc, Q_ests = MCTreeSearch(μβ0, env, N_MCTS, player, type=MCTS_type) # nextstates already observed here
        # as = collect(valid_actions(env, s))
        if train
            chosen_node = sample(μβ0.current_node.children, Weights(ps_trunc))
        else
            chosen_node = best_node
        end
        a = chosen_node.action
        @assert chosen_node.player == player
        winner, done, nextplayer = step!(env, a, player)
        μβ0.current_node = chosen_node
        remove_children!(chosen_node.parent, except=chosen_node)

        return Transition(s, a, Q_est, player, Q_ests, ps), winner, done, nextplayer
    else
        if train && rand() ≤ μβ0.ϵ
            a = rand(collect(valid_actions(env, s)))
        else
            a = action(μβ0, env, s, player) # greedy action
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

        ps = zeros(Float32, env.n_actions)
        ps[a] = 1 # one hot actions

        return Transition(s, a, Q_est, player, Float32[], ps), winner, done, nextplayer
    end
end


function reset_tree!(μβ0::MuBetaZero)
    root = MCTSNode(0)
    expand!(root, env, 1)
    μβ0.tree = MCTSTree(root)
    μβ0.current_node = root
end


function opponent_move(agent::MuBetaZero, env::Environment, player_adv::Int, N_MCTS::Int, thinktime::Float64)
    tik = time()
    best, = MCTreeSearch(agent, env, N_MCTS, player_adv)
    tak = time()
    n = 1
    while tak - tik < thinktime
        best, = MCTreeSearch(agent, env, N_MCTS, player_adv)
        n += 1
        tak = time()
    end
    a = best.action
    agent.current_node = best
    remove_children!(best.parent, except=best)
    t = round((tak - tik) * 1000) / 1000
    println("(Did $(n * N_MCTS) simulations in $t s.)")

    return a
end

function play_against(agent::MuBetaZero, env::Environment;
    start=true, MCTS=false, N_MCTS=1000, thinktime=0.5)
    reset!(env)
    winner = 0
    done = false
    player = start ? 1 : 2
    player_adv = start ? 2 : 1

    if MCTS
        reset_tree!(agent)
    end

    if !start
        if MCTS
            a = opponent_move(agent, env, player_adv, N_MCTS, thinktime)
        else
            a = greedy_action(agent, env, env.current, player_adv)
        end
        step!(env, a, player_adv)
    end

    while !done
        println()
        println_current(env)
        print("Input action: ")
        a = parse(Int, readline())
        while !(a in collect(valid_actions(env)))
            print("\nInput valid action: ")
            a = parse(Int, readline())
        end
        winner, done, = step!(env, a, player)
        if done
            break
        end

        if MCTS
            # track my move in mcts tree
            chosen_node = nothing
            for c in agent.current_node.children
                if c.action == a
                    chosen_node = c
                    break
                end
            end
            agent.current_node = chosen_node
            remove_children!(chosen_node.parent, except=chosen_node)
            expand!(chosen_node, env, player_adv)

            a = opponent_move(agent, env, player_adv, N_MCTS, thinktime)
        else
            a = greedy_action(agent, env, env.current, player_adv)
        end

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
        play_against(agent, env, start=!start, N_MCTS=N_MCTS, MCTS=MCTS, thinktime=thinktime)
    end
end

#=
import Profile
Profile.clear()
Profile.init()
@profiler train!(agent, env, 10^3, MCTS=true)
@profiler play_game!(agent, env, MCTS=true)

using BenchmarkTools
@btime train!(agent, env, 10^3)

reset!(env)

env.current = reshape([
    0 0 2;
    1 1 2;
    0 0 0
],:)
# env.current = reshape([
#     2 0 0;
#     0 1 1;
#     0 2 0
# ],:)
root = MCTSNode(0)
expand!(root, env, 1)
agent.tree = MCTSTree(root)
agent.current_node = root


Random.seed!(1)
best, v, ps, ps_trunc, scores = MCTreeSearch(agent, env, 100, 1, type=:value)
# remove_children!(best.parent, except=best)
agent.current_node = best
step!(env, best.action, best.player)
println_current(env)

print_tree(best)
print_children(best.parent)

play_game!(agent, env, train=false, verbose=true, MCTS=true)


agentMCTS = MuBetaZeroTabular(env.n_states, env.n_actions)
@time winners = train!(agentMCTS, env, 10^5, MCTS=true, N_MCTS=100, MCTS_type=:rollout)
play_game!(agentMCTS, env, train=false, verbose=true)
play_game!(agentMCTS, env, train=false, verbose=true, MCTS=true)

play_against(agentMCTS, env, MCTS=false)



using BenchmarkTools

@btime play_game!(agent, env, train=false, verbose=false, MCTS=true)
@btime play_game!(agent, env, train=false, verbose=false, MCTS=false)

@time play_game!(agent, env, train=false, verbose=false, MCTS=true)

# 8 μs ... 15s
# 20ms ...
=#
