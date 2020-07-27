mutable struct MCTSNode
    parent::Union{MCTSNode, Nothing}
    children::Vector{MCTSNode}
    v::Float64 # total return
    n::Int # number of sims
    action::Int

    function MCTSNode(parent=nothing, action=0)
        this = new()
        this.parent = parent
        this.children = []
        this.v = 0
        this.n = 0
        this.action = action
        return this
    end
end

struct MCTSTree
    root::MCTSNode
end

function USB1(node::MCTSNode)::Float64
    return node.v / node.n + √(2 * node.parent.n / node.n)
end

function best_child(node::MCTSNode, foe=false)
    scores = map(USB1, node.children)
    if foe
        # best action for foe is our worst action (least wins)
        return node.children[argmin(scores)]
    else
        return node.children[argmax(scores)]
    end
end


function print_node(node::MCTSNode, tab="")
    println(tab * "$(node.action) $(node.v)/$(node.n)")

    for c in node.children
        print_node(c, tab * "\t")
    end
end

function print_tree(tree::MCTSTree)
    print_node(tree.root)
end
