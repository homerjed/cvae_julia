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
    yints = ytrain # copy for getting images for plots
    xtrain = reshape(xtrain, 28^2, :)
    ytrain = float.(Flux.onehotbatch(ytrain, 0:nclasses-1))
    
    datadims = size(xtrain)
    @info datadims

    # reserve some imgs to plot
    nperclass = datadims[2] ÷ nclasses # @info nperclass
    xplot = zeros(Float32, 28^2, nclasses ^2)
    yplot = zeros(Float32, nclasses, nclasses * nclasses)
    
    for i in 0:nclasses-1
        # take the first 10 labels == i
        idx = findall(x -> x == i, yints)[1:nclasses]
        xplot[:,(i * nclasses) + 1 : (i + 1) * nclasses] = xtrain[:,idx]
        yplot[:,(i * nclasses) + 1 : (i + 1) * nclasses] = ytrain[:,idx]
    end

    xtrain = reshape(xtrain, 28, 28, 1, :)
    loader = DataLoader(
        (xtrain, ytrain), batchsize=batch_size, shuffle=true
    )
    return loader, xplot, yplot
end

struct Encoder
    conv_net
    μ
    logσ
end
@functor Encoder
    
Encoder(
    input_dim::Int, 
    latent_dim::Int, 
    hidden_dim::Int, 
    nclasses::Int
    ) = Encoder(
    Chain(
        Conv((3, 3), 1=>16,  stride=(2, 2), pad=(1, 1), relu), # 28x28
        Conv((3, 3), 16=>32, stride=(2, 2), pad=(1, 1), relu), # 14x14
        Conv((3, 3), 32=>32, stride=(2, 2), pad=(1, 1), relu), # 7x7
        x -> reshape(x, :, size(x, 4)), # Reshape 3d -> 2d
    ),
    Dense(512 + nclasses, latent_dim),        	         # μ
    Dense(512 + nclasses, latent_dim),        	         # logσ
)

function (encoder::Encoder)(x, y) 
    x = reshape(x, 28, 28, 1, :)
    x = encoder.conv_net(x) 
    x = reshape(x, 512, :)
    x = vcat(x, y) # conv output, onehot
    x = reshape(x, 512 + 10, :)
    # gaussian variables 
    encoder.μ(h), encoder.logσ(h)
end

Decoder(
    input_dim::Int, 
    latent_dim::Int, 
    hidden_dim::Int, 
    nclasses::Int
    ) = Chain(
    Dense(
        latent_dim + nclasses, 
        7 * 7 * (hidden_dim + nclasses)
    ), 
    x -> reshape(x, 7, 7, hidden_dim + nclasses, :),
    ConvTranspose(
        (3,3), 
        hidden_dim + nclasses => 16, 
        stride=2, 
        pad=SamePad(), 
        relu
    ),
    ConvTranspose((3,3), 16 => 1, stride=2, pad=SamePad(), relu),
)

function reconstruct(encoder, decoder, x, y, device) # y input here
    μ, logσ = encoder(x, y) # concatenate onehot label + latent vector
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    μ, logσ, reshape(decoder(vcat(z, y)), 28, 28, 1, :)
end

function model_loss(encoder, decoder, λ, x, y, device) # y input here
    μ, logσ, decoder_z = reconstruct(encoder, decoder, x, y, device) # y input here
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len

    decoder_z = reshape(decoder_z, len, :)
    x = reshape(x, len, :)
    logp_x_z = -logitbinarycrossentropy(decoder_z, x, agg=sum) / len
    # regularization
    reg = λ * sum(x -> sum(x.^2), Flux.params(decoder))
    
    -logp_x_z + kl_q_p + reg
end

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 28, :)...), (2, 1)))
end

# arguments for the `train` function 
@with_kw mutable struct Args
    η = 1e-3                # learning rate
    λ = 0.01f0              # regularization paramater
    nclasses = 10	        # number of classes for conditioning
    batch_size = 32         # batch size
    sample_size = 10        # sampling size for output    
    epochs = 100            # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    input_dim = 28^2        # image size
    latent_dim = 2          # latent dimension
    hidden_dim = 32         # hidden dimension
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
    loader, xplot, yplot = get_data(args.batch_size, args.nclasses)
    
    # initialize encoder and decoder, add class dim here
    encoder = Encoder(
        args.input_dim, 
        args.latent_dim, 
        args.hidden_dim, 
        args.nclasses
    ) |> device
    decoder = Decoder(
        args.input_dim, 
        args.latent_dim, 
        args.hidden_dim, 
        args.nclasses
    ) |> device

    # ADAM optimizer
    opt = ADAM(args.η)
    
    # parameters
    ps = Flux.params(
        encoder.conv_net, 
        encoder.μ, 
        encoder.logσ, 
        decoder
    )

    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogger
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    # fixed input
    xplot = xplot |> device
    yplot = yplot |> device
    image = convert_to_image(xplot, args.sample_size)
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
                model_loss(
                    encoder, 
                    decoder, 
                    args.λ, 
                    x |> device, 
                    y |> device, 
                    device
                )
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
        _, _, rec_xplot = reconstruct(
            encoder, decoder, xplot, yplot, device
        )
        rec_xplot = sigmoid.(rec_xplot)
        image = convert_to_image(rec_xplot, args.sample_size)
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
