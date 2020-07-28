using Flux

Flux.sum(abs2, [1,2,3])


c = Conv((4,4), 2 => 8, Flux.sigmoid)

m = MaxPool((4,4))

c.weight

Flux.outdims(c, (6,7))

input = zeros(6,7,2,1)

c(input)
Flux.flatten(c(input))


4*4*2*8*2

model = Chain(
    Conv((4,4), 2 => 8, sigmoid), # (3,4,8,:)
    Flux.flatten, # (96,:)
    Dense(96, 48, sigmoid),
    Dense(48, 7),
    Flux.softmax
)

model2 = Chain(
    Conv((4,4), 2 => 8, sigmoid), # (3,4,8,:)
    Flux.flatten, # (96,:)
    Dense(96, 48, relu),
    Dense(48, 1)
)

using BenchmarkTools
@btime
@btime model(rand(Float32, 6, 7, 2, 1))
1000 * 42 * 24 * 10^-6
mapreduce(length, +, params(model))

X = rand(Float32, 6, 7, 2, 32)
Y = Flux.softmax(rand(Float32, 7, 32))
V = rand(1, 32)
@btime gradient(params(model)) do
    Flux.crossentropy(model(X), Y)
end

@btime gradient(params(model2)) do
    Flux.mse(model2(X), V)
end
