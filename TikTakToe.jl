
import Base.display

abstract type Environment end

mutable struct TikTakToe <: Environment
    current::Vector{Int}
    n_states::Int
    n_actions::Int
    function TikTakToe()
        this = new()
        this.current = fill(0, 9)
        this.n_states = 3^9
        this.n_actions = 9
        return this
    end
end

# TODO:  makro
function s_a_to_index(env::Environment, state::Vector{Int}, action::Int)::Int
    branching_factor = 3
    scalar = 9
    index = action
    for s in state
        index += s * scalar
        scalar *= branching_factor
    end
    return index
end

function print_current(env::TikTakToe)
    symbols = ["0", "X", "O"]
    board = map(i -> symbols[i+1], env.current)
    for i in 1:3 # col
        for j in 1:3 # row
            print(board[i + 3 * (j - 1)], " ")
        end
        print("\n")
    end
    println()
end

function valid_actions(env::TikTakToe, state::Vector{Int})::Vector{Int}
    return findall(state .== 0)
end

function valid_actions(env::TikTakToe)::Vector{Int}
    return valid_actions(env.current)
end

function reset!(env::TikTakToe)
    env.current .= 0
end

function allsame(A::SubArray{Int})
    a_comp = A[1]
    for a in A
        if a != a_comp
            return false
        end
    end
    return true
end

function won(state::Vector{Int})::Int
    col = [1,2,3]
    row = [1,4,7]
    diag1 = [1,5,9]
    diag2 = [3,5,7]
    for i in 0:2
        p = state[col[1] + 3*i]
        if p != 0 && allsame(view(state, col .+ 3*i))
            return p
        end
        p = state[row[1] + i]
        if p != 0 && allsame(view(state, row .+ i))
            return p
        end
    end
    p = state[diag1[1]]
    if p != 0 && allsame(view(state, diag1))
        return p
    end
    p = state[diag2[1]]
    if p != 0 && allsame(view(state, diag2))
        return p
    end
    return 0
end


function step!(env::TikTakToe, action::Int, foe::Bool)::Tuple{Float32, Bool, Bool}
    player = foe+1 # me ... 1, foe ... 2
    env.current[action] = player
    p = won(env.current)

    if p == 1
        return 1f0, true, !foe # won
    elseif p == 2
        return -1f0, true, !foe # lost
    end
    if length(valid_actions(env)) == 0
        return 0f0, true, !foe # draw
    else
        return 0f0, false, !foe
    end
end


env = TikTakToe()

print_current(env)
step!(env, 5, false)
print_current(env)
step!(env, 9, true)
print_current(env)
step!(env, 4, false)
print_current(env)
step!(env, 8, true)
print_current(env)
step!(env, 6, false)
print_current(env)
