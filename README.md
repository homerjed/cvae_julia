# cvae_julia
Conditional Variational Auto-Encoder in Julia.

* `vae.jl` uses `Dense` layers
* `vae_conv.jl` uses `Conv` and `ConvTranspose` layers

Mostly borrowed from [FluxML](https://github.com/FluxML/model-zoo/tree/master/vision/vae_mnist).

![alt text](https://github.com/homerjed/cvae_julia/blob/main/original.png?raw=true)
![alt text](https://github.com/homerjed/cvae_julia/blob/main/epoch_20.png?raw=true)

To Do:
* interpolation test

---------------------------------------------------------------
Bonus: Add julia to the path in Mac OS

```
sudo mkdir -p /usr/local/bin
sudo rm -f /usr/local/bin/julia
sudo ln -s /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
```
