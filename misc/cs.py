"""
Cross-Dataset Validation: PTQ Methods with Sliding Window Evaluation (FINAL CORRECTED + BATCHED)

Fixes:
1. Sliding Window Math: Now uses correct context masking (labels=-100).
2. Llama 3 BOS: Manually handles BOS to prevent "Double BOS" (PPL 15.5 -> 6.2).
3. Tokenizer: Removed fix_mistral_regex kwarg entirely — causes "multiple values"
   error on Llama-3 and some tokenizers versions. Warning is cosmetic and harmless.
4. Batched windows: Multiple windows processed per forward pass. PPL estimator
   is identical (manual CrossEntropyLoss reduction='none', global NLL/token sums).

Metric: loglikelihood_rolling (Standard lm-evaluation-harness methodology)
Stride: 512 tokens
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import pickle
import warnings
from pathlib import Path


class PTQSlidingWindowValidator:
    def __init__(self, device="cuda", seed=42, stride=512, max_length=2048,
                 cache_dir="./dataset_cache", batch_size=8):
        self.device = device
        self.seed = seed
        self.stride = stride
        self.max_length = max_length
        self.batch_size = batch_size
        self.results = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        print("=" * 80)
        print("PTQ SLIDING WINDOW CROSS-DATASET VALIDATION (FINAL + BATCHED)")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Stride: {stride}")
        print(f"Max Seq Length: {max_length}")
        print(f"Batch Size: {batch_size}")
        print(f"Cache Dir: {cache_dir}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Dataset loaders
    # ------------------------------------------------------------------

    def load_wikitext2_test(self, n_samples=None):
        """
        Load WikiText-2 test set.
        CRITICAL: Concatenates all lines into one continuous stream.
        WikiText is a stream dataset; evaluating separate lines destroys context.
        Note: n_samples parameter is ignored - full test set is always used.
        """
        print("\n[1/3] Loading WikiText-2 test...")

        cache_file = self.cache_dir / f"wikitext2_test_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  📦 Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        full_text = "\n".join([x for x in dataset["text"] if x])
        print(f"  ✅ Loaded continuous stream ({len(full_text)} chars)")

        result = [full_text]
        print(f"  💾 Saving to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result

    def load_c4_validation(self, n_samples=500):
        """Load C4 validation set as continuous stream."""
        print("\n[2/3] Loading C4 validation...")

        cache_file = self.cache_dir / f"c4_validation_n{n_samples}_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  📦 Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in tqdm(dataset, total=n_samples, desc="  Collecting C4"):
            if len(texts) >= n_samples:
                break
            if len(item["text"].strip()) > 500:
                texts.append(item["text"])

        full_text = "\n\n".join(texts)
        print(f"  ✅ Loaded continuous stream ({len(full_text)} chars, {len(texts)} documents)")

        result = [full_text]
        print(f"  💾 Saving to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result

    def load_ag_news_test(self, n_samples=500):
        """Load AG News test set as continuous stream."""
        print("\n[3/3] Loading AG News test...")

        cache_file = self.cache_dir / f"ag_news_test_n{n_samples}_seed{self.seed}.pkl"
        if cache_file.exists():
            print(f"  📦 Loading from cache: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        dataset = load_dataset("ag_news", split="test")
        texts = [item["text"] for item in dataset if len(item["text"].strip()) > 200]

        if n_samples < len(texts):
            random.seed(self.seed)
            texts = random.sample(texts, n_samples)

        full_text = "\n\n".join(texts)
        print(f"  ✅ Loaded continuous stream ({len(full_text)} chars, {len(texts)} articles)")

        result = [full_text]
        print(f"  💾 Saving to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)

        return result

    # ------------------------------------------------------------------
    # Core evaluation (BATCHED)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_sliding_window(self, model, tokenizer, texts):
        """
        Batched sliding window evaluation.

        Mathematically identical to the per-window version:
          - Each window scores exactly `trg_len` tokens (the new tokens since the
            previous window), context tokens masked to -100.
          - Global NLL sum and global token count, exp(total_nll/total_tokens) at end.
        Only difference: windows are grouped into batches of `self.batch_size`
        and run through a single forward pass. Loss is computed manually with
        reduction='none' so per-window trg_len accounting is exact and unaffected
        by padding or batch averaging.
        """
        model.eval()
        total_nll = 0.0   # float accumulation — no per-window GPU sync
        total_tokens = 0

        for text in texts:
            # Tokenize WITHOUT adding special tokens automatically
            # This prevents the [BOS][BOS] double-injection issue
            encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids = encodings.input_ids
            print(f"  First 10 token IDs: {input_ids[0, :10].tolist()}")
            print(f"  tokenizer class: {type(tokenizer).__name__}")

            # Manual BOS injection — Llama 3 requires ID 128000 at position 0
            # Skip for Qwen2.5/GPT-2-style where bos==eos (endoftext), injecting it hurts stream PPL
            if tokenizer.bos_token_id is not None and tokenizer.bos_token_id != tokenizer.eos_token_id:
                if input_ids.shape[1] == 0 or input_ids[0, 0].item() != tokenizer.bos_token_id:
                    bos_tensor = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
                    input_ids = torch.cat([bos_tensor, input_ids], dim=1)

            # Safety cap: max_length * 200 tokens (~280k for WikiText-2 full test set)
            if input_ids.size(1) > self.max_length * 200:
                input_ids = input_ids[:, : self.max_length * 200]

            input_ids = input_ids.to(self.device)
            seq_len = input_ids.size(1)

            if seq_len < 2:
                continue

            # ---- Pre-compute all windows for this text ----
            # Each entry: (begin_loc, end_loc, trg_len) — same logic as the
            # original sequential loop, so the set of scored tokens is identical.
            windows = []
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, self.stride):
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                if trg_len <= 0:
                    break
                windows.append((begin_loc, end_loc, trg_len))
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            num_windows = len(windows)
            print(f"  Processing {seq_len:,} tokens in {num_windows} windows "
                  f"(batch_size={self.batch_size})...")

            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id

            pbar = tqdm(range(0, num_windows, self.batch_size),
                        desc="  Batches", unit="batch", leave=False)

            for batch_start in pbar:
                batch_windows = windows[batch_start: batch_start + self.batch_size]

                # Right-pad windows to the longest in this batch.
                chunks = [input_ids[:, b:e] for (b, e, _) in batch_windows]
                max_len = max(c.size(1) for c in chunks)
                bsz = len(chunks)

                batch_input = torch.full(
                    (bsz, max_len), pad_id, dtype=torch.long, device=self.device,
                )
                batch_labels = torch.full(
                    (bsz, max_len), -100, dtype=torch.long, device=self.device,
                )
                attn_mask = torch.zeros(
                    (bsz, max_len), dtype=torch.long, device=self.device,
                )

                for i, (chunk, (b, e, trg_len)) in enumerate(zip(chunks, batch_windows)):
                    clen = chunk.size(1)
                    batch_input[i, :clen] = chunk[0]
                    attn_mask[i, :clen] = 1
                    # Score only the last trg_len real tokens of this window
                    lbl = chunk.clone()
                    lbl[:, :-trg_len] = -100
                    batch_labels[i, :clen] = lbl[0]

                outputs = model(batch_input, attention_mask=attn_mask)
                logits = outputs.logits  # [B, T, V]

                # Manual shift + per-token loss. HF's built-in `labels=` loss
                # averages over the whole batch, mixing windows of different
                # trg_len — so we compute it ourselves and sum globally.
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_labels[:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                flat_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )  # ignored (-100) positions contribute exactly 0

                batch_nll = flat_loss.sum().item()
                batch_tok = int((shift_labels != -100).sum().item())

                total_nll += batch_nll
                total_tokens += batch_tok

                if total_tokens > 0:
                    current_ppl = float(np.exp(total_nll / total_tokens))
                    pbar.set_postfix({
                        "PPL": f"{current_ppl:.4f}",
                        "tokens": f"{total_tokens:,}",
                    })

        if total_tokens == 0:
            return None

        perplexity = float(np.exp(total_nll / total_tokens))
        return {"perplexity": perplexity, "total_tokens": total_tokens}

    # ------------------------------------------------------------------
    # Tokenizer loader (fix for fix_mistral_regex conflict)
    # ------------------------------------------------------------------

    @staticmethod
    def load_tokenizer(model_path):
        """
        Load tokenizer without passing fix_mistral_regex.

        fix_mistral_regex was a tokenizers-internal kwarg used to suppress a
        deprecation warning. Passing it to AutoTokenizer on Llama-3 (or newer
        tokenizers builds) raises:
            TypeError: got multiple values for keyword argument 'fix_mistral_regex'
        The warning it suppresses is cosmetic — skipping the kwarg is safe.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        return tokenizer

    # ------------------------------------------------------------------
    # Per-model evaluation
    # ------------------------------------------------------------------

    def evaluate_model_on_dataset(self, model_path, model_name, texts, dataset_name):
        print(f"\n  Evaluating {model_name} on {dataset_name}...")

        try:
            tokenizer = self.load_tokenizer(model_path)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            print(f"model.dtype={model.dtype}")
            print(f"max_length={self.max_length}  stride={self.stride}  batch_size={self.batch_size}")
            print(f"BOS={tokenizer.bos_token_id}  EOS={tokenizer.eos_token_id}")
            results = self.evaluate_sliding_window(model, tokenizer, texts)

            if results:
                print(f"  ✅ Perplexity: {results['perplexity']:.4f}")
            else:
                print("  ❌ Evaluation failed (no results)")

            del model
            torch.cuda.empty_cache()
            return results

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_validation(self, heuristic_path, standard_path=None, n_samples=2000):
        print("\n" + "=" * 80)
        print("LOADING DATASETS")
        print("=" * 80)

        datasets = {
            "WikiText-2": self.load_wikitext2_test(n_samples),
            "C4": self.load_c4_validation(n_samples),
            # "AG News": self.load_ag_news_test(n_samples),
        }

        print("\n" + "=" * 80)
        print("EVALUATING MODELS")
        print("=" * 80)

        models = {"Heuristic PTQ": heuristic_path}
        if standard_path:
            models["Standard PTQ"] = standard_path

        for dataset_name, texts in datasets.items():
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")

            for model_name, model_path in models.items():
                result = self.evaluate_model_on_dataset(model_path, model_name, texts, dataset_name)
                if result:
                    if dataset_name not in self.results:
                        self.results[dataset_name] = {}
                    self.results[dataset_name][model_name] = result

        return self.results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_comparison_table(self):
        """Generate formatted comparison table."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE RESULTS")
        print("=" * 80)

        has_both_models = any(
            len(self.results.get(ds, {})) == 2 for ds in self.results.keys()
        )

        if not has_both_models:
            print(f"\n{'Dataset':<15} {'Model':<20} {'Perplexity':<15} {'Total Tokens':<15}")
            print("-" * 70)

            dataset_results = []
            for dataset_name, models_data in self.results.items():
                for model_name, data in models_data.items():
                    ppl = data["perplexity"]
                    tokens = data["total_tokens"]
                    print(f"{dataset_name:<15} {model_name:<20} {ppl:<15.4f} {tokens:<15,}")
                    dataset_results.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "perplexity": ppl,
                        "total_tokens": tokens,
                    })
            return dataset_results

        print(f"\n{'Dataset':<15} {'Heuristic PTQ':<15} {'Standard PTQ':<15} {'Delta':<12} {'Winner':<10}")
        print("-" * 80)

        dataset_results = []
        for dataset_name in self.results.keys():
            if (
                "Heuristic PTQ" in self.results[dataset_name]
                and "Standard PTQ" in self.results[dataset_name]
            ):
                heur_ppl = self.results[dataset_name]["Heuristic PTQ"]["perplexity"]
                std_ppl = self.results[dataset_name]["Standard PTQ"]["perplexity"]
                delta = heur_ppl - std_ppl
                delta_pct = (delta / std_ppl) * 100
                winner = "Heuristic" if delta < -0.05 else ("Standard" if delta > 0.05 else "Tie")

                print(f"{dataset_name:<15} {heur_ppl:<15.4f} {std_ppl:<15.4f} {delta_pct:>+11.3f}%  {winner:<10}")

                dataset_results.append({
                    "dataset": dataset_name,
                    "heuristic_ppl": heur_ppl,
                    "standard_ppl": std_ppl,
                    "delta_pct": delta_pct,
                    "winner": winner,
                })

        return dataset_results

    def analyze_results(self, dataset_results):
        """Comprehensive analysis of results."""
        if not dataset_results or "heuristic_ppl" not in dataset_results[0]:
            print("\n" + "=" * 80)
            print("SINGLE MODEL EVALUATION COMPLETE")
            print("=" * 80)
            return {"mode": "single_model"}

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        heur_wins = sum(1 for r in dataset_results if r["winner"] == "Heuristic")
        std_wins = sum(1 for r in dataset_results if r["winner"] == "Standard")
        ties = sum(1 for r in dataset_results if r["winner"] == "Tie")

        print(f"\nWin Count:")
        print(f"  Heuristic PTQ: {heur_wins}/{len(dataset_results)}")
        print(f"  Standard PTQ:  {std_wins}/{len(dataset_results)}")
        print(f"  Ties:          {ties}/{len(dataset_results)}")

        avg_heur = np.mean([r["heuristic_ppl"] for r in dataset_results])
        avg_std = np.mean([r["standard_ppl"] for r in dataset_results])
        avg_delta_pct = ((avg_heur - avg_std) / avg_std) * 100

        print(f"\nAverage Perplexity:")
        print(f"  Heuristic PTQ: {avg_heur:.4f}")
        print(f"  Standard PTQ:  {avg_std:.4f}")
        print(f"  Difference:    {avg_delta_pct:+.3f}%")

        print("\n" + "=" * 80)
        print("FINAL VERDICT")
        print("=" * 80)

        if heur_wins > std_wins:
            print(f"\n🏆 HEURISTIC PTQ is the OVERALL WINNER!")
            print(f"   Wins: {heur_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            winner = "Heuristic PTQ"
        elif std_wins > heur_wins:
            print(f"\n🏆 STANDARD PTQ is the OVERALL WINNER!")
            print(f"   Wins: {std_wins}/{len(dataset_results)} datasets")
            print(f"   Average improvement: {abs(avg_delta_pct):.3f}%")
            winner = "Standard PTQ"
        else:
            print(f"\n🤝 TIE - Both methods equally strong")
            winner = "Tie"

        return {
            "winner": winner,
            "heuristic_wins": heur_wins,
            "standard_wins": std_wins,
            "ties": ties,
            "avg_heuristic": avg_heur,
            "avg_standard": avg_std,
            "avg_delta_pct": avg_delta_pct,
        }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PTQ Sliding Window Cross-Dataset Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--heuristic-path", type=str, required=True,
                        help="Path to Heuristic PTQ model")
    parser.add_argument("--standard-path", type=str, default="",
                        help="Path to Standard PTQ model (optional for comparison)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of samples per dataset")
    parser.add_argument("--cache-dir", type=str, default="./dataset_cache",
                        help="Directory to cache downloaded datasets")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length per window")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride between windows")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of windows per forward pass")
    args = parser.parse_args()

    validator = PTQSlidingWindowValidator(
        cache_dir=args.cache_dir,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    validator.run_validation(
        args.heuristic_path,
        args.standard_path if args.standard_path else None,
        args.n_samples,
    )

    dataset_results = validator.generate_comparison_table()
    analysis = validator.analyze_results(dataset_results)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    if analysis.get("mode") != "single_model":
        print(f"\n🏆 Winner: {analysis['winner']}")
        print(f"📊 Tested: {len(dataset_results)} datasets")
        print(f"✅ Heuristic wins: {analysis['heuristic_wins']}")
        print(f"✅ Standard wins: {analysis['standard_wins']}")
        print(f"🤝 Ties: {analysis['ties']}")
    else:
        print(f"\n📊 Tested: {len(dataset_results)} datasets")
        print(f"✅ Single model evaluation complete")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()    