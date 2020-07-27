import Base.argmax
import Base.argmin

function maximise(f::Function, A)
    a = first(A)
    fa = f(a)
    for b in A
        fb = f(b)
        if fb > fa
            a = b
            fa = fb
        end
    end
    return a, fa
end

function minimise(f::Function, A)
    a = first(A)
    fa = f(a)
    for b in A
        fb = f(b)
        if fb < fa
            a = b
            fa = fb
        end
    end
    return a, fa
end
