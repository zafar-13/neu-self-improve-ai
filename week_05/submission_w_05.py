import os
import re
import glob
import json
import math
import argparse
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class Config:
    # Model
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    trust_remote_code: bool = True
    use_4bit: bool = True
    device_map: str = "auto"

    # Data sources
    use_hf_math: bool = True
    math_repo_dir: str = "./math"
    use_hf_math500: bool = True

    # Data sizes
    train_samples: int = 7500
    test_samples: int = 500

    # A*-PO: Stage 1 (V*)
    n_samples_vstar: int = 8
    beta_1: float = 0.5
    vstar_cache_path: str = "v_star_cache.json"

    # A*-PO: Stage 2 (online)
    beta_2: float = 1e-3
    learning_rate: float = 1e-6
    num_epochs: int = 1
    batch_size: int = 4
    grad_clip: float = 1.0

    # Generation
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0

    # Eval
    greedy_eval: bool = True
    use_pag_eval: bool = False
    pag_temperature: float = 0.6
    pag_top_p: float = 1.0
    pag_max_turns: int = 2

    # Misc
    seed: int = 42
    out_dir: str = "./outputs_astar_po"

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--use_4bit", type=int, default=1)
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--trust_remote_code", type=int, default=1)

    # Data sources
    p.add_argument("--use_hf_math", type=int, default=1)
    p.add_argument("--math_repo_dir", type=str, default="")
    p.add_argument("--use_hf_math500", type=int, default=1)

    # Data sizes
    p.add_argument("--train_samples", type=int, default=7500)
    p.add_argument("--test_samples", type=int, default=500)

    # Stage 1
    p.add_argument("--n_samples_vstar", type=int, default=8)
    p.add_argument("--beta_1", type=float, default=0.5)
    p.add_argument("--vstar_cache_path", type=str, default="v_star_cache.json")

    # Stage 2
    p.add_argument("--beta_2", type=float, default=1e-3)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)

    # Eval
    p.add_argument("--greedy_eval", type=int, default=1)
    p.add_argument("--use_pag_eval", type=int, default=0)
    p.add_argument("--pag_temperature", type=float, default=0.6)
    p.add_argument("--pag_top_p", type=float, default=1.0)
    p.add_argument("--pag_max_turns", type=int, default=2)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="./outputs_astar_po")

    args = p.parse_args()
    cfg = Config(
        model_id=args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
        use_4bit=bool(args.use_4bit),
        device_map=args.device_map,
        use_hf_math=bool(args.use_hf_math),
        math_repo_dir=args.math_repo_dir,
        use_hf_math500=bool(args.use_hf_math500),
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        n_samples_vstar=args.n_samples_vstar,
        beta_1=args.beta_1,
        vstar_cache_path=args.vstar_cache_path,
        beta_2=args.beta_2,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        greedy_eval=bool(args.greedy_eval),
        use_pag_eval=bool(args.use_pag_eval),
        pag_temperature=args.pag_temperature,
        pag_top_p=args.pag_top_p,
        pag_max_turns=args.pag_max_turns,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    return cfg

# Utils

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

BOX_PAT = re.compile(r'\\boxed\{([^}]*)\}')

def extract_boxed(text: str) -> Optional[str]:
    m = BOX_PAT.search(text or "")
    return m.group(1).strip() if m else None

def normalize_ans(s: Optional[str]) -> str:
    if s is None: return ""
    s = s.strip()
    s = s.replace(r'\,', '').replace(r'\!', '').replace('~', ' ')
    s = s.replace(r'\left', '').replace(r'\right', '')
    s = re.sub(r'\s+', '', s)
    if len(s) >= 2 and s[0] == '(' and s[-1] == ')':
        inner = s[1:-1]
        if all(op not in inner for op in '+-*/'):
            s = inner
    s = s[:-1] if s.endswith('.') and s[:-1].isdigit() else s
    return s

def parse_gt(solution_text: str) -> str:
    gt = extract_boxed(solution_text)
    if gt is None:
        lines = [ln for ln in (solution_text or "").splitlines() if ln.strip()]
        gt = lines[-1].strip() if lines else ""
    return normalize_ans(gt)

def is_correct(pred_boxed: Optional[str], gt_solution: str) -> int:
    pred = normalize_ans(pred_boxed)
    truth = parse_gt(gt_solution)
    return int(pred == truth)

def format_prompt(question: str) -> str:
    return f"Question: {question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# Model Loading

def load_models_and_tokenizer(cfg: Config):
    quantization_config = None
    if cfg.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                print("CUDA not available; disabling 4-bit.")
        except Exception as e:
            print("bitsandbytes not available, proceeding without 4-bit:", e)
            quantization_config = None

    common_kwargs = dict(trust_remote_code=cfg.trust_remote_code)
    if quantization_config is not None:
        common_kwargs.update(dict(
            quantization_config=quantization_config,
            device_map=cfg.device_map
        ))
    else:
        common_kwargs.update(dict(
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=cfg.device_map
        ))

    print(f"Loading policy: {cfg.model_id}")
    policy = AutoModelForCausalLM.from_pretrained(cfg.model_id, **common_kwargs)
    print(f"Loading reference (frozen): {cfg.model_id}")
    ref = AutoModelForCausalLM.from_pretrained(cfg.model_id, **common_kwargs)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return policy, ref, tok

# Data Loading

def _read_math_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {"problem": obj.get("problem", ""), "solution": obj.get("solution", "")}

def load_math_from_repo(math_repo_dir: str, train_samples: int, test_samples: int):
    train_glob = os.path.join(math_repo_dir, "train", "**", "*.json")
    test_glob  = os.path.join(math_repo_dir, "test",  "**", "*.json")
    train_files = sorted(glob.glob(train_glob, recursive=True))
    test_files  = sorted(glob.glob(test_glob,  recursive=True))

    if len(train_files) == 0 or len(test_files) == 0:
        raise FileNotFoundError(
            f"No JSON files found under {math_repo_dir} (expected train/**.json and test/**.json). "
            "Ensure you cloned https://github.com/hendrycks/math ."
        )

    random.Random(123).shuffle(train_files)
    random.Random(456).shuffle(test_files)
    train_pick = train_files[:train_samples]
    test_pick  = test_files[:test_samples]

    train = [_read_math_json(p) for p in train_pick]
    test  = [_read_math_json(p) for p in test_pick]
    return train, test

def load_math_hf(train_samples: int, test_samples: int):
    from datasets import load_dataset
    ds_train = load_dataset("hendrycks/competition_math", split="train")
    ds_test  = load_dataset("hendrycks/competition_math", split="test")
    train = ds_train.select(range(min(train_samples, len(ds_train))))
    test  = ds_test.select(range(min(test_samples, len(ds_test))))
    train = [{"problem": ex["problem"], "solution": ex["solution"]} for ex in train]
    test  = [{"problem": ex["problem"], "solution": ex["solution"]} for ex in test]
    return train, test

def load_math500_hf():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    eval500 = [{"problem": ex["problem"], "solution": ex["solution"]} for ex in ds]
    return eval500

# Stage 1: V* estimation

@torch.no_grad()
def estimate_v_star(ref_model, tokenizer, dataset: List[dict], cfg: Config) -> Dict[int, float]:
    cache_path = cfg.vstar_cache_path
    if os.path.exists(cache_path):
        print(f"Loading V* cache from {cache_path}")
        with open(cache_path, "r") as f:
            data = json.load(f)
        return {int(k): float(v) for k, v in data.items()}

    print("--- Stage 1: V* Estimation ---")
    v_star_values: Dict[int, float] = {}
    for idx, item in enumerate(tqdm(dataset, total=len(dataset))):
        prompt = format_prompt(item["problem"])
        gt = item["solution"]
        inputs = tokenizer(prompt, return_tensors="pt").to(ref_model.device)

        rewards = []
        for _ in range(cfg.n_samples_vstar):
            out_ids = ref_model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            rewards.append(is_correct(extract_boxed(text), gt))

        r = torch.tensor(rewards, dtype=torch.float32, device=ref_model.device)
        v = cfg.beta_1 * torch.log(torch.clamp(torch.exp(r / cfg.beta_1).mean(), min=1.0))
        v_star_values[idx] = float(v.item())

    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in v_star_values.items()}, f)
    return v_star_values

# Stage 2 helpers


@torch.no_grad()
def seq_logprob(model, tokenizer, prompt: str, response_text: str) -> torch.Tensor:
    # Build full text (safe even if response doesn't include prompt prefix)
    full_text = prompt + response_text[len(prompt):] if response_text.startswith(prompt) else prompt + response_text
    enc = tokenizer(full_text, return_tensors="pt").to(model.device)
    out = model(**enc)
    logits = out.logits  # [1, T, V]

    p = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = p.input_ids.shape[1]

    labels = enc.input_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt
    logp = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    targets = labels[:, 1:]
    mask = (targets != -100)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    return token_logp.sum()  # scalar

@torch.no_grad()
def get_log_probs(model, tokenizer, prompts: List[str], responses: List[str]) -> torch.Tensor:
    vals = []
    for pr, rs in zip(prompts, responses):
        vals.append(seq_logprob(model, tokenizer, pr, rs).to(model.device))
    return torch.stack(vals)

# Stage 2 training (A*-PO)

def train_astar_po(policy_model, ref_model, tokenizer, train_dataset: List[dict], v_star_values: Dict[int, float], cfg: Config):
    print("\n--- Stage 2: On-Policy Training (A*-PO) ---")
    os.makedirs(cfg.out_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.learning_rate)

    N = len(train_dataset)
    steps_per_epoch = math.ceil(N / cfg.batch_size)
    for epoch in range(cfg.num_epochs):
        policy_model.train()
        total_loss = 0.0

        for i in tqdm(range(0, N, cfg.batch_size), total=steps_per_epoch, desc=f"Epoch {epoch+1}"):
            j0, j1 = i, min(i + cfg.batch_size, N)
            items = [train_dataset[j] for j in range(j0, j1)]
            idxs = list(range(j0, j1))

            prompts = [format_prompt(it["problem"]) for it in items]
            gts = [it["solution"] for it in items]

            # Generate responses with current policy
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(policy_model.device)
            out_ids = policy_model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
            responses = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

            # Rewards and V*
            rewards = torch.tensor(
                [is_correct(extract_boxed(r), gt) for r, gt in zip(responses, gts)],
                dtype=torch.float32, device=policy_model.device
            )
            v_star_batch = torch.tensor(
                [v_star_values[idx] for idx in idxs],
                dtype=torch.float32, device=policy_model.device
            )

            # Log-prob ratio
            policy_lp = get_log_probs(policy_model, tokenizer, prompts, responses)
            ref_lp    = get_log_probs(ref_model,    tokenizer, prompts, responses)
            log_ratio = policy_lp - ref_lp  # [B]

            # A*-PO loss
            advantages = rewards - v_star_batch
            loss = (cfg.beta_2 * log_ratio - advantages).pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, steps_per_epoch)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

    # Save
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoint_last")
    os.makedirs(ckpt_dir, exist_ok=True)
    policy_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print("Saved:", ckpt_dir)

# Evaluation


@torch.no_grad()
def evaluate_greedy(model, tokenizer, eval_dataset: List[dict], cfg: Config) -> float:
    print("\n--- Evaluation (greedy) ---")
    model.eval()
    correct = 0
    for it in tqdm(eval_dataset, total=len(eval_dataset)):
        prompt = format_prompt(it["problem"])
        gt = it["solution"]
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **enc, max_new_tokens=cfg.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        ans = extract_boxed(text)
        correct += is_correct(ans, gt)
    acc = 100.0 * correct / len(eval_dataset)
    print(f"Accuracy: {acc:.2f}%")
    return acc

@torch.no_grad()
def generate(model, tokenizer, prompt: str, temperature: float, top_p: float, max_new: int) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc, do_sample=True, temperature=temperature, top_p=top_p,
        max_new_tokens=max_new, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

@torch.no_grad()
def evaluate_pag(model, tokenizer, eval_dataset: List[dict], cfg: Config) -> Tuple[float, float]:
    print("\n--- Evaluation (PAG-style: verify-then-revise) ---")
    model.eval()
    t1 = 0
    tf = 0
    VERIFIER_INSTR = (
        "Check the math solution step-by-step. If incorrect, end with 'The answer is wrong'. "
        "If correct, end with 'The answer is correct'."
    )
    REVISE_INSTR = (
        "You indicated the previous answer was wrong. Provide the corrected solution. "
        "End with the final answer in \\boxed{}."
    )
    for it in tqdm(eval_dataset, total=len(eval_dataset)):
        q = it["problem"]; gt = it["solution"]
        p1 = format_prompt(q)
        y1 = generate(model, tokenizer, p1, cfg.pag_temperature, cfg.pag_top_p, cfg.max_new_tokens)
        a1 = extract_boxed(y1)
        t1 += is_correct(a1, gt)

        if cfg.pag_max_turns < 2:
            tf += is_correct(a1, gt); continue

        v_prompt = f"{q}\n\nPrevious solution:\n{y1}\n\nInstruction: {VERIFIER_INSTR}"
        v_out = generate(model, tokenizer, v_prompt, cfg.pag_temperature, cfg.pag_top_p, cfg.max_new_tokens)
        is_wrong = "The answer is wrong" in v_out
        if not is_wrong:
            tf += is_correct(a1, gt)
        else:
            r_prompt = f"{q}\n\nPrevious solution:\n{y1}\n\nInstruction: {REVISE_INSTR}"
            y2 = generate(model, tokenizer, r_prompt, cfg.pag_temperature, cfg.pag_top_p, cfg.max_new_tokens)
            a2 = extract_boxed(y2)
            tf += is_correct(a2, gt)

    acc_t1 = 100.0 * t1 / len(eval_dataset)
    acc_final = 100.0 * tf / len(eval_dataset)
    print(f"Acc@t1: {acc_t1:.2f}%, Acc@final: {acc_final:.2f}%")
    return acc_t1, acc_final

# Main

def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("Config:", asdict(cfg))

    # Load models
    policy, ref, tok = load_models_and_tokenizer(cfg)

    # Data
    if cfg.use_hf_math:
        train_ds, test_ds = load_math_hf(cfg.train_samples, cfg.test_samples)
    else:
        if not cfg.math_repo_dir:
            raise ValueError("--use_hf_math=0 but --math_repo_dir is empty.")
        train_ds, test_ds = load_math_from_repo(cfg.math_repo_dir, cfg.train_samples, cfg.test_samples)

    if cfg.use_hf_math500:
        eval_ds = load_math500_hf()
    else:
        eval_ds = test_ds[:min(cfg.test_samples, 500)]

    # Stage 1
    v_star = estimate_v_star(ref, tok, train_ds, cfg)

    # Stage 2
    train_astar_po(policy, ref, tok, train_ds, v_star, cfg)

    # Eval
    if cfg.greedy_eval:
        evaluate_greedy(policy, tok, eval_ds, cfg)
    if cfg.use_pag_eval:
        evaluate_pag(policy, tok, eval_ds, cfg)

if __name__ == "__main__":
    main()
