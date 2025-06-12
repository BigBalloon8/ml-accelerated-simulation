import torch
import safetensors
import matplotlib.pyplot as plt

FILENAME = "/home/crae/projects/ml-accelerated-simulation/data/data.safetensors"

def main():
    opener = safetensors.safe_open(FILENAME, framework="pt").__enter__()
    
    sample = opener.get_tensor("2_f")
    samplec = opener.get_tensor("2_c")

    plt.subplot(1,2,1)
    plt.imshow(sample[0])
    plt.subplot(1,2,2)
    plt.imshow(sample[1])
    plt.show()

    norm_sample = samplec/torch.norm(samplec)
    Ux = norm_sample[0].numpy()
    Uy = norm_sample[1].numpy()
    import numpy as np
    X, Y = np.meshgrid(np.arange(Ux.shape[1]), np.arange(Ux.shape[0]))
    fig, ax = plt.subplots(figsize=(6, 6))

    mag = np.sqrt(Ux**2 + Uy**2)
    im = ax.imshow(mag, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax, label='Magnitude')

    # Add vector field
    ax.quiver(X, Y, Ux, Uy, color='w',)  # adjust scale for visual clarity
    ax.set_aspect('equal')
    ax.set_title('Vector Field (quiver)')
    plt.show()


    opener.__exit__(None, None, None)

if __name__ == "__main__":
    main()
    #st.save_file({}, "/home/crae/projects/ml-accelerated-simulation/data/data.safetensors")