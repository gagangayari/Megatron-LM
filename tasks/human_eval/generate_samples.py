
"""Sample Generate GPT"""
import os
import sys
import re
sys.path.append(os.path.abspath(os.path.join(
    os.getcwd(),
    "Megatron-LM",
)))
sys.path.remove("/app")
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm


GENERATE_NUM = 0
BEAM_NUM = 1

# End on unindented code
EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "<EOS>", r"<\|endoftext\|>"]

def send_do_generate():
    choice = torch.cuda.LongTensor([GENERATE_NUM])
    torch.distributed.broadcast(choice, 0)

def send_do_beam_search():
    choice = torch.cuda.LongTensor([BEAM_NUM])
    torch.distributed.broadcast(choice, 0)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    args = get_args()
    model = GPTModel(num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process, prefix_lm=args.is_prefix_lm)

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--strip_prompt", action="store_true",
                       help='Strip whitespaces from prompt.')
    # group.add_argument("--out-seq-length", type=int, default=1024,
    #                    help='Size of the output generated text.')
    group.add_argument("--tokens_to_generate", type=int, default=64,
                       help='Max number of tokens to generate')
    group.add_argument("--add_bos", default=False, action='store_true',
                       help='Add <|endoftext|> token at the beginning of the prompt')
    group.add_argument("--is-prefix-lm", action="store_true")
    group.add_argument("--prefix", default="", type=str,
                       help="prefix to add at the beginning of each (stripped) prompt. For example, '<BOS>[S]' for a model trained with UL2")
    group.add_argument("--suffix", default="", type=str,
                       help="suffix to add at the end of each (stripped) prompt. For example, '<SEP>' for a model trained with UL2")
    group.add_argument("--inference_batch_size", type=int, default=128,
                       help='Batch-size for inference.')
    group.add_argument("--load_iteration", type=int, default=None,
        help='Iteration to load. If not specified, will load the latest checkpoint'
    )

    group = parser.add_argument_group(title='text generation output')
    group.add_argument(
        "--num_samples_per_task", type=int, default=200
    )
    group.add_argument(
        "--output-file", type=str, default="samples_{}.jsonl"
    )
    return parser


def get_batches(prompts, batch_size):
    for start_idx in tqdm(range(0, len(prompts), batch_size)):
        actual_batch_size = min(batch_size, len(prompts) - start_idx)
        yield prompts[start_idx: start_idx + actual_batch_size]


def unbatch(d: dict):
    return [dict(zip(d.keys(), t)) for t in zip(*d.values())]


def load_evaluation_data(args):
    # HumanEval data
    problems = read_problems()

    batches = get_batches(
        [
            problems[task_id]
            for task_id in problems
            for _ in range(args.num_samples_per_task)
        ],
        args.inference_batch_size
    )
    return batches


def post_process_completion(completion: str):
    """Cut generation at any of EOF_STRINGS"""
    eof_pattern = r'|'.join(EOF_STRINGS)
    match = re.search(eof_pattern, completion)
    if match is not None:
        return completion[:match.start()]
    else:
        return completion


if __name__ == "__main__":
    # Initialize Megatron
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Setup model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        iteration = load_checkpoint(model, None, None, iteration=args.load_iteration)
    else:
        iteration = None

    assert len(model) == 1
    model = model[0]

    def generate(prompts):
        prefix = args.prefix if args.prefix is not None else ""
        suffix  = args.suffix if args.suffix is not None else ""
        prompts = [prefix + p + suffix for p in prompts]
        assert mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0
        send_do_generate()  # Tell other ranks we're doing generate
        response, response_seg, response_logprobs, tokens = \
                generate_and_post_process(
                    model,
                    prompts=prompts,
                    tokens_to_generate=args.tokens_to_generate,
                    return_output_log_probs=False,
                    top_k_sampling=args.top_k,
                    top_p_sampling=args.top_p,
                    temperature=args.temperature,
                    add_BOS=args.add_bos,
                    use_eod_token_for_early_termination=True,
                    prefix_lm=args.is_prefix_lm,
                    sep_in_bidirectional_context=False)
        if args.add_bos:
            # assert all([p in r for r, p in zip(response, prompts)])
            result = {
                "response": response, 
                "response_seg": response_seg,
                "raw_completion": [r[r.find(p)+len(p):] for r, p in zip(response, prompts)]
            }
        else:
            # assert all([r.startswith(p) for r, p in zip(response, prompts)])  # TODO: add assert again once tokenizer issue is fixed
            result = {
                "response": response, 
                "response_seg": response_seg,
                "raw_completion": [r[len(p):] for r, p in zip(response, prompts)]
            }
        # The "completion" field contains the string that is actually going to be evaluated by the HumanEval script
        result["completion"] = [post_process_completion(c) for c in result["raw_completion"]]
        # Return a list of dicts
        return unbatch(result)

    # if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
    #     server = MegatronServer(model)
    #     server.run("0.0.0.0")

    # while True:
    #     choice = torch.cuda.LongTensor(1)
    #     torch.distributed.broadcast(choice, 0)
    #     if choice[0].item() == 0:
    #         generate_and_post_process(model)

    # Evaluation data iterator
    batches = load_evaluation_data(args)
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:

        # Generate
        samples = [
            dict(task_id=task_id,**generate_dict)
            for batch in batches
            for task_id, generate_dict in zip(
                [e["task_id"] for e in batch],
                generate([e["prompt"].strip() if args.strip_prompt else e["prompt"] for e in batch])
            )
        ]

        # Write results to file
        write_jsonl(args.output_file.format(iteration), samples)
        
    else:
        # Other ranks: Call generate once for each batch
        for _ in batches:
            choice = torch.cuda.LongTensor(1)
            torch.distributed.broadcast(choice, 0)
            if choice[0].item() == 0:
                generate_and_post_process(model)
            elif choice[0].item() == 1:
                beam_search_and_post_process(model)

