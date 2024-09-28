import torch
import argparse
import numpy as np
from lm_eval import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.modify_llama import QH2OLlamaForCausalLM, QH2OLlamaAttention

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")

    # KV Cache Policy
    parser.add_argument("--hh_size", type=float, default=0.15)
    parser.add_argument("--recent_size", type=float, default=0.05)
    # For quantization
    parser.add_argument("--kbits", type=int, default=4)
    parser.add_argument("--vbits", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument('--enable_qh2o_cache', action='store_true')

    parser.add_argument("--seed", type=int, default=2, help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if args.enable_qh2o_cache:
        if args.alpha == 1:
            print('Enabling Quantization H2O KV cache')
        else:
            print('Enabling Q-Hitter Cache')
        config.hh_size = args.hh_size
        config.recent_size = args.recent_size
        config.kbits = args.kbits
        config.vbits = args.vbits
        config.alpha = args.alpha
        # model = ENABLE_Heavy_Hitter_FUNCTIONS['llama_qh2o'].from_pretrained(model_name, config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    from utils.lmeval_warp import LMEvalLM

    model_warp = LMEvalLM(model_name, config=config)

    results = evaluator.simple_evaluate(
        model=model_warp,
        tasks=["coqa"],
        log_samples=True,
        # limit=8
    )

    print(results["results"])