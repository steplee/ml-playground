# VAE in TensorFlow
Right now it is set up to reconstruct Kanji (see `./text_data.py`)

`./run_interpolation.py` will interpolate in z-space and reconstruct.

Here is a picture of interpolating between two encoded z-points (top half) and getting further and further from an encoded point in z-space (bottom half).
![alt text](./vae-interpolation.png)

## Usage
` python3 run_train.py run &&  python3 run_interpolate.py run && google-chrome vae-interpolation.png `
