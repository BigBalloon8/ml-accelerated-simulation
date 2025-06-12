from src.data.dataloader import get_kolomogrov_flow_data_loader

def main():
    tl, vl = get_kolomogrov_flow_data_loader("/home/crae/projects/ml-accelerated-simulation/data/data.safetensors")
    for x, y in tl:
        print(x.shape)
        print(y.shape)

if __name__ == "__main__":
    main()
    