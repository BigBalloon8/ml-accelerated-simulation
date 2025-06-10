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
    plt.imshow(samplec[0])
    plt.show()

    opener.__exit__(None, None, None)

if __name__ == "__main__":
    main()
    #st.save_file({}, "/home/crae/projects/ml-accelerated-simulation/data/data.safetensors")