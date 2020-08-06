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

X = rand(Float32, 6, 7, 2, 1024)
Y = Flux.softmax(rand(Float32, 7, 1024))
V = rand(1, 32)
@btime

@btime gradient(params(model2)) do
    Flux.mse(model2(X), V)
end

using Logging
# debug logging
logger=Logging.SimpleLogger(stderr,Logging.Debug);
ENV["JULIA_DEBUG"] = "all";

# ENV["JULIA_DEBUG"] = Flux

using Random
Random.seed!(1)
model = Chain(
    Conv((4,4), 2 => 8, sigmoid), # (3,4,8,:)
    Flux.flatten, # (96,:)
    Dense(96, 48, sigmoid),
    Dense(48, 7),
    Flux.softmax
)
X = rand(Float32, 6, 7, 2, 1024)
Y = Flux.softmax(rand(Float32, 7, 1024))

Xs = [reshape(X[:,:,:,i], 6, 7, 2, 1) for i in 1:size(X, 4)]
Ys = [reshape(Y[:,i], 7, 1) for i in 1:size(Y, 4)]

gradient(params(model)) do
    Flux.crossentropy(model(X), Y)
end

m = deepcopy(model)

loss(x,y) = Flux.crossentropy(m(x), y)

loss(X, Y)
l = 0f0
for (x, y) in zip(Xs, Ys)
    global l += loss(x, y)
end
display(l)
-sum(Y .* log.(m(X)) .* 1) / size(Y, 2)

Flux.Optimise.train!(loss, params(m), [(X, Y)], RMSProp())

loss(X, Y)

dl = Flux.Data.DataLoader(X, Y, batchsize=4)

for d in dl
    println(typeof(d))
end

sum(L2, params(m))

m = gpu(model)

@btime cpu(m(gpu(X))) # 372.400 μs 1024

@btime for x in Xs
    m(gpu(x))
end # 316.811 ms 1024 LOL

maximum(m(gpu(Xs[1]))[1, :])


m2 = deepcopy(model)

@btime m2(X) # 2.337 ms 1024

@btime for x in Xs
    m2(x)
end # 28.415 ms
m2()
