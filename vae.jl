# Variational Autoencoder(VAE)
#
# Auto-Encoding Variational Bayes
# Diederik P Kingma, Max Welling
# https://arxiv.org/abs/1312.6114

using BSON
using CUDA
using DrWatson: struct2dict
using Flux
using Flux: @functor, chunk
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Images
using Logging: with_logger
using MLDatasets
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random

# load MNIST images and return loader
function get_data(batch_size, nclasses)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtrain = reshape(xtrain, 28^2, :)
    ytrain = float.(Flux.onehotbatch(ytrain, 0:nclasses-1))
    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
end

struct Encoder
    linear
    μ
    logσ
end
@functor Encoder
    
Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int, nclasses::Int) = Encoder(
    Dense(input_dim + nclasses, hidden_dim, tanh),   # linear
    Dense(hidden_dim, latent_dim),        	     # μ
    Dense(hidden_dim, latent_dim),        	     # logσ
)

#function (encoder::Encoder)(x, y) # concatenate x and y here
function (encoder::Encoder)(x) # concatenate x and y here
    #input = vcat(x, y)
    h = encoder.linear(x)
    #h = encoder.linear(input)
    encoder.μ(h), encoder.logσ(h)
end

Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int, nclasses::Int) = Chain(
    Dense(latent_dim + nclasses, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
)

function reconstuct(encoder, decoder, x, y, device) # y input here
    #μ, logσ = encoder(x, y)
    μ, logσ = encoder(vcat(x, y))
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    μ, logσ, decoder(vcat(z, y))
end

function model_loss(encoder, decoder, λ, x, y, device) # y input here
    μ, logσ, decoder_z = reconstuct(encoder, decoder, x, y, device) # y input here
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len

    logp_x_z = -logitbinarycrossentropy(decoder_z, x, agg=sum) / len
    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(decoder))
    
    -logp_x_z + kl_q_p + reg
end

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

# arguments for the `train` function 
@with_kw mutable struct Args
    η = 1e-3                # learning rate
    λ = 0.01f0              # regularization paramater
    nclasses = 10	    # number of classes for conditioning
    batch_size = 128        # batch size
    sample_size = 10        # sampling size for output    
    epochs = 20             # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    input_dim = 28^2        # image size
    latent_dim = 2          # latent dimension
    hidden_dim = 500        # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    tblogger = false        # log training with tensorboard
    save_path = "output"    # results path
end

function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load MNIST images
    loader = get_data(args.batch_size, args.nclasses)
    
    # initialize encoder and decoder, add class dim here
    encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim, args.nclasses) |> device
    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim, args.nclasses) |> device

    # ADAM optimizer
    opt = ADAM(args.η)
    
    # parameters
    ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)

    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogger
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    # fixed input
    original, y_original = first(get_data(args.sample_size^2, args.nclasses))
    original = original |> device
    y_original = y_original |> device
    image = convert_to_image(original, args.sample_size)
    image_path = joinpath(args.save_path, "original.png")
    save(image_path, image)

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for (x, y) in loader 
            loss, back = Flux.pullback(ps) do
                model_loss(encoder, decoder, args.λ, x |> device, y |> device, device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)]) 

            # logging with TensorBoard
            if args.tblogger && train_steps % args.verbose_freq == 0
                with_logger(tblogger) do
                    @info "train" loss=loss
                end
            end

            train_steps += 1
        end

        # save image
        _, _, rec_original = reconstuct(encoder, decoder, original, y_original, device)
        rec_original = sigmoid.(rec_original)
        image = convert_to_image(rec_original, args.sample_size)
        image_path = joinpath(args.save_path, "epoch_$(epoch).png")
        save(image_path, image)
        @info "Image saved: $(image_path)"
    end

    # save model
    model_path = joinpath(args.save_path, "model.bson") 
    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
        BSON.@save model_path encoder decoder args
        @info "Model saved: $(model_path)"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__ 
    train()
end
