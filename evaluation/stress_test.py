import os
import sys
import torch

# Clean path appending
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import GPT
from configs.experiment_configs import CONFIGS
from tokenizer.bpe_tokenizer import SimpleBPETokenizer
from evaluation.generate import generate_text, load_model_and_tokenizer

print("=========================================")
print("Starting Automated Model Stress Testing")
print("=========================================")

CONFIG_NAME = 'task_4_1_baseline'
CHECKPOINT_PATH = f'experiments/{CONFIG_NAME}_checkpoint.pt'

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Baseline checkpoint not found at {CHECKPOINT_PATH}. Run training first!")
    sys.exit(1)

model, tokenizer, device = load_model_and_tokenizer(CONFIG_NAME, CHECKPOINT_PATH)

# Setup a results container to document findings
stress_test_log = []

def log_test(test_name, input_used, output_generated, hazard_rating):
    print(f"\n[!] Test Executed: {test_name}")
    print(f"Input: {input_used.strip()}")
    print(f"Output: {output_generated.strip()}")
    print(f"Observed Severity: {hazard_rating}")
    print("-" * 50)
    stress_test_log.append({
        "test_name": test_name,
        "input": input_used,
        "output": output_generated,
        "severity": hazard_rating
    })

# --- Test Case 1: Triggering Repetition Loops ---
# Strategy: Feed a highly generic, flat phrase combined with strict greedy decoding.
prompt_1 = "The rain falls. The rain falls. "
out_1 = generate_text(model, tokenizer, prompt_1, max_new_tokens=40, strategy='greedy')
log_test("Repetition Loop Vulnerability", prompt_1, out_1, "HIGH")

# --- Test Case 2: Out-Of-Bounds Context Forgetting ---
# Strategy: Feed a massive prompt that matches or exceeds the block size configuration (128 tokens)
# to see how the model manages its context truncation limit.
prompt_2 = "word " * 140 
try:
    out_2 = generate_text(model, tokenizer, prompt_2, max_new_tokens=20, strategy='top_p', top_p=0.9, temperature=0.8)
except Exception as e:
    out_2 = f"CRASHED: {str(e)}"
log_test("Context Memory Overflow Limit", "140 word repetitions", out_2, "CRITICAL")

# --- Test Case 3: Hallucinations and Subword Disintegration ---
# Strategy: Crank temperature up to extreme values to force selection from the low-probability tail.
prompt_3 = "Deep in the shadows of the concrete jungle,"
out_3 = generate_text(model, tokenizer, prompt_3, max_new_tokens=40, strategy='temperature', temperature=4.5)
log_test("High-Temperature Entropy Hallucination", prompt_3, out_3, "HIGH")

# --- Test Case 4: Logical Contradictions ---
# Strategy: Provide a multi-clause prompt stating contradictory facts to analyze consistency.
prompt_4 = "I hate hip-hop music, it makes me angry. So I wrote a rap song because I love it: "
out_4 = generate_text(model, tokenizer, prompt_4, max_new_tokens=40, strategy='top_k', top_k=10, temperature=0.8)
log_test("Logical Coherence & Contradiction Tracking", prompt_4, out_4, "MEDIUM")

# Write results summary to reports directory
report_path = "output/stress_test_py/raw_stress_test_outputs.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== RAW ADVERSARIAL STRESS TEST LOGS ===\n\n")
    for test in stress_test_log:
        f.write(f"Test: {test['test_name']}\nSeverity: {test['severity']}\nInput: {test['input']}\nOutput: {test['output']}\n")
        f.write("="*40 + "\n")

print(f"\nStress testing complete. Raw findings exported directly to: {report_path}")