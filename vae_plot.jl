include("./vae.jl")
# using vae

using Plots

function plot_result()
    BSON.@load "output/model.bson" encoder decoder args
    args = Args(; args...)
    device = args.cuda && CUDA.has_cuda() ? gpu : cpu
    encoder, decoder = encoder |> device, decoder |> device

    # load MNIST images
    loader, _, _ = get_data(args.batch_size, args.nclasses)

    # clustering in the latent space
    # visualize first two dims
    plt = scatter(palette=:rainbow)
    for (i, (x, y)) in enumerate(loader)
        i < 20 || break
        
        μ, logσ = encoder(vcat(x, y) |> device)

        y = Flux.onecold(y, 0:args.nclasses-1)
        scatter!(
            μ[1, :], μ[2, :], 
            markerstrokewidth=0, 
            markeralpha=0.8,
            aspect_ratio=1,
            markercolor=y, 
            label=""
        )
    end
    savefig(plt, "output/clustering.png")

    z = range(-2.0, stop=2.0, length=11)
    len = Base.length(z)
    z1 = repeat(z, len)
    z2 = sort(z1)
    x = zeros(Float32, args.latent_dim, len^2) |> device
    y = rand(0:9, len^2)
    y = float.(Flux.onehotbatch(y, 0:args.nclasses-1))
    x[1, :] = z1
    x[2, :] = z2
    samples = decoder(vcat(x,y))
    samples = sigmoid.(samples)
    image = convert_to_image(samples, len)
    save("output/manifold.png", image)
end

if abspath(PROGRAM_FILE) == @__FILE__ 
    plot_result()
end
