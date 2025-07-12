

def main(path):
    with open(path, "r") as f:
        logs = f.read().split("\n\n\n")