import Base.push!

mutable struct MCTSNode
    parent::Union{MCTSNode, Nothing}
    children::Vector{MCTSNode}
    w::Int # number of wins for player
    n::Int # number of sims
    action::Int
    player::Int

    function MCTSNode(player, action=0, parent=nothing)
        this = new()
        this.parent = parent
        this.children = []
        this.v = 0
        this.n = 0
        this.action = action
        this.player = player
        return this
    end
end

struct MCTSTree
    root::MCTSNode
end

function push!(node::MCTSNode, child::MCTSNode)
    push!(node.children, child)
    child.parent = node
end

# dont know if necessary for garbage collector
function remove_children!(node::MCTSNode)
    for child in node.children
        child.parent = nothing
    end
    node.children = []
end

function UCB1(node::MCTSNode)::Float64
    if node.n == 0
        return Inf
    else
        return node.v / node.n + √(2 * node.parent.n / node.n)
    end
end

function best_child(node::MCTSNode)
    scores = map(UCB1, node.children)
    # if foe
    #     # best action for foe is our worst action (least wins)
    #     return node.children[argmax(-scores)]
    # else
    return node.children[argmax(scores)]
    # end
end

function str_node(node::MCTSNode)
    return "p:$(node.player) a:$(node.action) v:$(node.v)/$(node.n)"
end

function print_node(node::MCTSNode, tab=""; frt::Function=str_node)
    c = println(tab * frt(node))
    for c in node.children
        print_node(c, tab * "\t", frt=frt)
    end
end

function print_tree(tree::MCTSTree)
    print_node(tree.root)
end

function print_tree(node::MCTSNode)
    root = node
    while root.parent != nothing
        root = root.parent
    end
    str_node_2 = function(n)
        c = n == node ? " * " : ""
        return c * str_node(n) * c
    end
    print_node(root, frt=str_node_2)
end

function v_mean(node::MCTSNode)
    return node.v / node.n
end
