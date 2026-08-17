import argparse
from utils import set_seed
from evaluation.load_model import load_model_from_checkpoint
from evaluation.demo_utils import predict_all, select_examples, log_demo_table

def demo_pipeline(model_path, n=10, mode="random"):
    bundle = load_model_from_checkpoint(model_path)
    
    results = predict_all(bundle['model'], bundle['test_loader'], bundle['device'],
                        bundle['mean'], bundle['std'], bundle['config']['data']['target_idx'])

    examples = select_examples(results, n=n, mode=mode)

    log_demo_table(examples, bundle['test_data'], bundle['config'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path of the model checkpoint")
    parser.add_argument("--n", type=int, default=10, help="Number of examples showed")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "worst", "best"])

    args = parser.parse_args()

    demo_pipeline(args.model_path, n=args.n, mode=args.mode)