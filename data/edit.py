import torch
import safetensors

FILENAME = "/home/crae/projects/ml-accelerated-simulation/data/data.safetensors"

def main():
    opener = safetensors.safe_open(FILENAME, framework="pt").__enter__()
    print(opener.keys())
    opener.__exit__(None, None, None)

if __name__ == "__main__":
    main()
    #st.save_file({}, "/home/crae/projects/ml-accelerated-simulation/data/data.safetensors")