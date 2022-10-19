import argparse
import torch

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='To state dict')
    parser.add_argument('--checkpoint_path', type=str, default="")
    parser.add_argument('--save_path', type=str, default="")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path)
    model = checkpoint["model"]
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()
