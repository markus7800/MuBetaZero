
mutable struct ConnectFour <: Environment
    current::Array{Float32,3}
    internal_state::Array{Int,2}
    n_actions::Int
    n_rows::Int
    n_cols::Int
    function ConnectFour()
        this = new()
        this.n_rows = 6
        this.n_cols = 7
        this.current = fill(0f0, 6, 7, 2)
        this.internal_state = fill(0, 6, 7)
        this.n_actions = 7
        return this
    end
end


function print_current(env::ConnectFour)
    symbols = ["0", "X", "O"]
    board = map(i -> symbols[i+1], env.internal_state)
    for i in env.n_rows:-1:1
        for j in 1:env.n_cols
            print(board[i,j], " ")
        end
        print("\n")
    end
end

function println_current(env::ConnectFour)
    print_current(env)
    println()
end

function valid_actions(env::ConnectFour, state::Array{Int,2})::Base.Iterators.Filter
    return Base.Iterators.filter(a -> state[env.n_rows,a] == 0, 1:env.n_actions)
end

function valid_actions(env::ConnectFour, state::Array{Float32,3})::Base.Iterators.Filter
    return Base.Iterators.filter(a -> state[env.n_rows,a,1] == 0 && state[env.n_rows,a,2] == 0, 1:env.n_actions)
end

function valid_actions(env::ConnectFour)::Base.Iterators.Filter
    return valid_actions(env, env.internal_state)
end

function reset!(env::ConnectFour)
    env.current .= 0f0
    env.internal_state .= 0
end

function step!(env::ConnectFour, action::Int, player::Int)::Tuple{Int, Bool, Int}
    i = 0; j = action
    for l in 1:env.n_rows
        if env.internal_state[l,action] == 0
            env.internal_state[l,action] = player
            env.current[l,action,player] = 1
            i = l
            break
        end
    end

    next_player = player == 1 ? 2 : 1

    w = won(env, i, j, player)
    if w
        return player, true, next_player
    else
        return 0, isempty(valid_actions(env)), next_player
    end

end

function index(env::ConnectFour, i::Int, j::Int)
    return i + (j-1) * env.n_rows
end

# for i in 1:env.n_rows, j in 1:env.n_cols
#     @assert env.current[i,j] == env.current[index(env, i, j)]
# end

function won(env::ConnectFour, i::Int, j::Int, player::Int)
    L = max(j-3,1)
    R = min(j+3, env.n_cols)

    T = max(i-3, 1)
    B = min(i+3, env.n_rows)

    tl = min(j-L,i-T)
    TL = index(env, i-tl, j-tl)
    br = min(R-j,B-i)
    BR = index(env, i+br, j+br)

    bl = min(j-L,B-i)
    BL = index(env, i+bl, j-bl)
    tr = min(R-j,i-T)
    TR = index(env, i-tr, j+tr)

    col = index(env,T,j):1:index(env,B,j)
    row = index(env,i,L):env.n_rows:index(env,i,R)
    diag1 = TL:env.n_rows+1:BR
    diag2 = BL:env.n_rows-1:TR

    if has_four(env.internal_state, col, player)
        return true
    elseif has_four(env.internal_state, row, player)
        return true
    elseif has_four(env.internal_state, diag1, player)
        return true
    elseif has_four(env.internal_state, diag2, player)
        return true
    else
        return false
    end
    # if has_four(view(env.current, col), player)
    #     return true
    # elseif has_four(view(env.current, row), player)
    #     return true
    # elseif has_four(view(env.current, diag1), player)
    #     return true
    # elseif has_four(view(env.current, diag2), player)
    #     return true
    # else
    #     return false
    # end
end

# function has_four(A::Array{Int,2}, a::Int)
#     l = length(A)
#     if l < 4
#         return false
#     end
#     count = 0
#     for (i,b) in enumerate(A)
#         if b == a
#             count += 1
#             if count == 4
#                 return true
#             end
#         else
#             count = 0
#             if l - i < 4
#                 return false
#             end
#         end
#     end
#     return false
# end

function has_four(A::Array{Int,2}, indeces::StepRange, a::Int)
    l = length(indeces)
    if l < 4
        return false
    end
    count = 0
    for (i, index) in  enumerate(indeces)
        if A[index] == a
            count += 1
            if count == 4
                return true
            end
        else
            count = 0
            if l - i < 4
                return false
            end
        end
    end
    return false
end


# env = ConnectFour()
# reset!(env)
# println_current(env)
# step!(env, 4, 1)
# println_current(env)
# step!(env, 4, 2)
# println_current(env)
# step!(env, 4, 1)
# println_current(env)
# step!(env, 4, 2)
# println_current(env)
# step!(env, 4, 1)
# println_current(env)
# step!(env, 4, 2)
# println_current(env)
# collect(valid_actions(env))
#
# reset!(env)
# println_current(env)
# step!(env, 4, 1)
# println_current(env)
# step!(env, 5, 2)
# println_current(env)
# step!(env, 4, 1)
# println_current(env)
# step!(env, 5, 2)
# println_current(env)
# step!(env, 4, 1)
# println_current(env)
# step!(env, 5, 2)
# println_current(env)
# step!(env, 4, 1)
# println_current(env)
#
#
# reset!(env)
# println_current(env)
# step!(env, 4, 1)
# println_current(env)
# step!(env, 5, 2)
# println_current(env)
# step!(env, 3, 1)
# println_current(env)
# step!(env, 5, 2)
# println_current(env)
# step!(env, 2, 1)
# println_current(env)
# step!(env, 5, 2)
# println_current(env)
# step!(env, 1, 1)
# println_current(env)
#
#
# reset!(env)
# println_current(env)
# step!(env, 4, 1)
# println_current(env)
# step!(env, 5, 2)
# println_current(env)
# step!(env, 5, 1)
# println_current(env)
# step!(env, 6, 2)
# println_current(env)
# step!(env, 6, 1)
# println_current(env)
# step!(env, 7, 2)
# println_current(env)
# step!(env, 6, 1)
# println_current(env)
# step!(env, 7, 2)
# println_current(env)
# step!(env, 7, 1)
# println_current(env)
# step!(env, 1, 2)
# println_current(env)
# step!(env, 7, 1)
# println_current(env)
