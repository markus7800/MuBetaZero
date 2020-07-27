
abstract type Environment end

function step!(env::Environment, action::Int, player::Int)::Tuple{Float32, Bool} end

function
