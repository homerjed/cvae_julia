include("./vae.jl")

using Plots

function plot_result()
    BSON.@load "output/model.bson" encoder decoder args
    args = Args(; args...)
    device = args.cuda && CUDA.has_cuda() ? gpu : cpu
    encoder, decoder = encoder |> device, decoder |> device

    int_a, int_b = 3, 6 # integers to interpolate between
    n = 100 # must be a square number
	y = zeros(args.nclasses, n)
	for i in 1:n
		step = i * 1.0 / n
	    y[int_a, i] = step
		y[int_b, i] = 1.0 - step
	end
    
    n_ = convert(Int32, sqrt(n))
    z = repeat(randn(Float32, args.latent_dim, n_), 1, n_)
    x = decoder(vcat(z, y))
    x = sigmoid.(x)
    image = convert_to_image(x, n_)   
    save("output/interpolation.png", image)
end

if abspath(PROGRAM_FILE) == @__FILE__ 
    plot_result()
end
