_# =============================================================================
# NLP Project: FLAN-T5 Large + LoRA — VS CODE VERSION
# Local Environment Version — Local paths, no Google Drive
# =============================================================================

# =============================================================================
# STEP 1 — Setup Instructions (Run these in your VS Code Terminal first)
# =============================================================================
"""
Before running, open your VS Code terminal and run:
pip install transformers datasets evaluate[meteor] sacrebleu rouge_score peft accelerate sentencepiece nltk pandas matplotlib torch torchvision torchaudio
"""

# =============================================================================
# STEP 2 — Imports
# =============================================================================
import os
import re
import json
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# On local machines, 'Agg' prevents windows from popping up during training
matplotlib.use('Agg') 

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import nltk
from nltk.corpus import wordnet

# Download NLTK data to your local machine
nltk.download('wordnet',    quiet=True)
nltk.download('omw-1.4',    quiet=True)
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)

print(" All imports done.")

# =============================================================================
# STEP 3 — Settings (LOCAL PATHS)
# =============================================================================
SEED       = 42
MODEL_NAME = "google/flan-t5-base"

# ❗ CHANGE THESE TO YOUR LOCAL FOLDERS
# Example: "C:/Users/YourName/Desktop/ASL_Project/dataset.csv"
# ✅ Updated paths
DATA_PATH = r"C:\Users\sondo\OneDrive - Alexandria National University\Desktop\NLP code\2\asl_dataset_english_to_gloss.csv"
OUTPUT_DIR = r"C:\Users\sondo\OneDrive - Alexandria National University\Desktop\NLP code\2"

MAX_INPUT  = 128
MAX_TARGET = 128
BATCH_SIZE = 4      # Lowered for local GPUs (change to 8 if you have >12GB VRAM)
GRAD_ACCUM = 4      # 4*4 = 16 (matches your previous effective batch size)
EPOCHS     = 10 
LR         = 1e-4

# LoRA config
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGET     = ["q", "v"]

PREFIX = "translate english to gloss: "

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f" All results will be saved to local folder: {os.path.abspath(OUTPUT_DIR)}")

# =============================================================================
# STEP 4 — GPU Info (Checks for NVIDIA GPU)
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device      : {device}")
if device == "cuda":
    print(f"   GPU          : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   VRAM         : {vram:.1f} GB")
else:
    print("    WARNING: No GPU found. Training on CPU will be EXTREMELY slow.")

# =============================================================================
# STEP 5 — Load Local Dataset
# =============================================================================
print(f"\n Loading dataset from {DATA_PATH}...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f" Could not find {DATA_PATH}. Please put the CSV in the same folder as this script.")

df = pd.read_csv(DATA_PATH)
df = df[['input', 'output']].dropna().reset_index(drop=True)

print(f" Dataset loaded: {len(df):,} rows")

# --- LOGIC CHECK ---
# This ensures 'input' is English and 'output' is Gloss before you start training.
print("\n Logic Check (First 2 rows):")
for i in range(2):
    print(f"   Row {i} | Input (English): {df['input'].iloc[i]}")
    print(f"   Row {i} | Output (Gloss):  {df['output'].iloc[i]}")
print("-" * 30)

# =============================================================================
# STEP 6 — Split Dataset  80 / 10 / 10
# =============================================================================
dataset  = Dataset.from_pandas(df)
split    = dataset.train_test_split(test_size=0.2, seed=SEED)
val_test = split['test'].train_test_split(test_size=0.5, seed=SEED)

dataset_dict = DatasetDict({
    'train': split['train'],
    'val':   val_test['train'],
    'test':  val_test['test'],
})

print(f"\n Splits:")
print(f"   Train : {len(dataset_dict['train']):,}")
print(f"   Val   : {len(dataset_dict['val']):,}")
print(f"   Test  : {len(dataset_dict['test']):,}")

# =============================================================================
# STEP 7 — Load Tokenizer & Model with LoRA
#    LoRA trains only ~0.5% of parameters — full model quality, fraction of RAM
# =============================================================================
print(f"\n Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_config)
model.to(device)

print(" Model ready with LoRA adapter")
print("   Trainable parameters:")
model.print_trainable_parameters()

# =============================================================================
# STEP 8 
# =============================================================================
def normalize_english_input(text):
    """Simple cleanup for the English sentence before translation."""
    if text.startswith(PREFIX):
        sentence = text[len(PREFIX):]
    else:
        sentence = text
    
    sentence = sentence.strip().lower() # Model handles lowercase English better
    return PREFIX + sentence

# =============================================================================
# STEP 9 — Output Post-Processor
#    Cleans up the model's raw output string.
# =============================================================================
def post_process(text):
    """Cleans up model output: fixes punctuation, capitalizes, trims."""
    if not text:
        return text
    # Fix double punctuation like '?.' or '.?'
    text = re.sub(r'[?.!]+([?.!])', r'\1', text)
    # Remove space before punctuation
    text = re.sub(r'\s+([?.!,;:])', r'\1', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    return text

# =============================================================================
# STEP 10 — Data Augmentation
#    Applied to INPUT (gloss) on train split only — 30% chance per sample.
#    Replaces one word in the gloss with a WordNet synonym.
#    OUTPUT (English sentence) is NEVER modified.
# =============================================================================
def synonym_replace_english(text, n=1):
    if text.startswith(PREFIX):
        gloss = text[len(PREFIX):]
    else:
        gloss = text

    words = gloss.split()
    if len(words) < 4:                 # skip very short glosses
        return PREFIX + gloss

    new_words = words.copy()
    count = 0

    for i, word in enumerate(words):
        syns = wordnet.synsets(word.lower())
        if syns and count < n:
            lemmas = [
                l.name().replace("_", " ")
                for l in syns[0].lemmas()
                if l.name().lower() != word.lower()
            ]
            if lemmas:
                new_words[i] = random.choice(lemmas).upper()
                count += 1

    return PREFIX + " ".join(new_words)

# =============================================================================
# STEP 11 — Preprocessing
#    padding='max_length' is required — do not remove it.
#    -100 for label padding tells the loss function to ignore those positions.
# =============================================================================
def preprocess(example, augment=False):
    text = str(example['input'])

    if augment and random.random() < 0.3:
        text = synonym_replace_english(text, 1)

    model_input = tokenizer(
        text,
        max_length=MAX_INPUT,
        truncation=True,
        padding='max_length',
    )

    labels = tokenizer(
        text_target=str(example['output']),
        max_length=MAX_TARGET,
        truncation=True,
        padding='max_length',
    )

    model_input['labels'] = [
        -100 if tok == tokenizer.pad_token_id else tok
        for tok in labels['input_ids']
    ]
    return model_input


print("\n Tokenizing datasets ...")
tokenized = DatasetDict({
    'train': dataset_dict['train'].map(
        lambda x: preprocess(x, augment=True),  batched=False),
    'val':   dataset_dict['val'].map(
        lambda x: preprocess(x, augment=False), batched=False),
    'test':  dataset_dict['test'].map(
        lambda x: preprocess(x, augment=False), batched=False),
})
print(" Tokenization complete")

# =============================================================================
# STEP 12 — Data Collator
# =============================================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

# =============================================================================
# STEP 13 — Metrics
# =============================================================================
bleu_metric   = evaluate.load("bleu")
rouge_metric  = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 with pad token so decoder doesn't crash
    preds   = np.where(preds   != -100, preds,   tokenizer.pad_token_id)
    labels  = np.where(labels  != -100, labels,  tokenizer.pad_token_id)

    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    bleu_score   = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels]
    )
    rouge_score  = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    try:
        meteor_score = meteor_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        meteor_val = meteor_score['meteor'] * 100
    except Exception:
        meteor_val = 0.0

    exact = sum(
        p == r for p, r in zip(decoded_preds, decoded_labels)
    ) / len(decoded_preds) * 100

    return {
        "bleu":        round(bleu_score['bleu'] * 100, 4),
        "rougeL":      round(rouge_score['rougeL'] * 100, 4),
        "meteor":      round(meteor_val, 4),
        "exact_match": round(exact, 4),
    }

# =============================================================================
# STEP 14 — Training Arguments
# =============================================================================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # Evaluate and save every epoch
    # With 79K rows this is ~10K steps per epoch — epoch-based is correct here
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,

    # Learning — 3e-4 is correct for LoRA (much higher than full fine-tune)
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_ratio=0.05,
    max_grad_norm=0.5,
    label_smoothing_factor=0.1,

    # Batch
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    # Generation — beam search with 4 beams for best quality
    predict_with_generate=True,
    generation_max_length=MAX_TARGET,
    generation_num_beams=4,

    # Logging
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=100,
    logging_strategy="steps",
    report_to="none",

    # Save only last 3 checkpoints to save Drive space
    save_total_limit=3,

    # fp16 disabled — model loaded in float16 already via torch_dtype
    fp16=False,

    seed=SEED,
)

# =============================================================================
# STEP 15 — Trainer
# =============================================================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['val'],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# =============================================================================
# STEP 16 — Checkpoint Resume + Train
# =============================================================================
def get_last_checkpoint(folder):
    """Returns the latest checkpoint path in folder, or None."""
    if not os.path.isdir(folder):
        return None
    checkpoints = [
        os.path.join(folder, d)
        for d in os.listdir(folder)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(folder, d))
    ]
    return max(checkpoints, key=os.path.getmtime) if checkpoints else None

last_ckpt = get_last_checkpoint(OUTPUT_DIR)
print(f"\n Starting training on: {device}")
print(f"   Resuming from: {last_ckpt}" if last_ckpt else "   Starting from scratch.")
print(f"   Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"   Max epochs: {EPOCHS} (early stopping patience=3)")

trainer.train(resume_from_checkpoint=last_ckpt)

# =============================================================================
# STEP 17 — Save LoRA Adapter + Tokenizer
# =============================================================================
print("\n Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   LoRA adapter saved to: {OUTPUT_DIR}")

# Also merge LoRA back into base model for easy app deployment
# (App can load merged_model without needing PEFT installed)
print("   Merging LoRA into base model for deployment...")
merged_model = model.merge_and_unload()
merged_dir = os.path.join(OUTPUT_DIR, "merged_model")
os.makedirs(merged_dir, exist_ok=True)
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print(f"   Merged model saved to: {merged_dir}")
print("    Use merged_model/ for app integration — no PEFT needed at inference.")

# =============================================================================
# STEP 18 — Training Curves
# =============================================================================
print("\n Generating training curves...")
history = trainer.state.log_history

train_steps  = [l["step"] for l in history if "loss"        in l]
train_loss   = [l["loss"] for l in history if "loss"        in l]
eval_steps   = [l["step"] for l in history if "eval_loss"   in l]
eval_loss    = [l["eval_loss"]   for l in history if "eval_loss"   in l]
bleu_hist    = [l["eval_bleu"]   for l in history if "eval_bleu"   in l]
rougel_hist  = [l["eval_rougeL"] for l in history if "eval_rougeL" in l]
meteor_hist  = [l["eval_meteor"] for l in history if "eval_meteor" in l]

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

axs[0][0].plot(train_steps, train_loss, color='blue', lw=1.5)
axs[0][0].set_title("Training Loss", fontsize=13, fontweight='bold')
axs[0][0].set_xlabel("Steps"); axs[0][0].set_ylabel("Loss")
axs[0][0].grid(True, alpha=0.3)

axs[0][1].plot(eval_steps, eval_loss, color='orange', marker='o', lw=1.5)
axs[0][1].set_title("Validation Loss", fontsize=13, fontweight='bold')
axs[0][1].set_xlabel("Epoch"); axs[0][1].set_ylabel("Loss")
axs[0][1].grid(True, alpha=0.3)

axs[0][2].plot(eval_steps[:len(bleu_hist)], bleu_hist, color='green', marker='o', lw=1.5)
axs[0][2].set_title("Validation BLEU", fontsize=13, fontweight='bold')
axs[0][2].set_xlabel("Epoch"); axs[0][2].set_ylabel("BLEU (%)")
axs[0][2].grid(True, alpha=0.3)

axs[1][0].plot(eval_steps[:len(rougel_hist)], rougel_hist, color='red', marker='o', lw=1.5)
axs[1][0].set_title("Validation ROUGE-L", fontsize=13, fontweight='bold')
axs[1][0].set_xlabel("Epoch"); axs[1][0].set_ylabel("ROUGE-L (%)")
axs[1][0].grid(True, alpha=0.3)

if meteor_hist:
    axs[1][1].plot(eval_steps[:len(meteor_hist)], meteor_hist, color='purple', marker='o', lw=1.5)
    axs[1][1].set_title("Validation METEOR", fontsize=13, fontweight='bold')
    axs[1][1].set_xlabel("Epoch"); axs[1][1].set_ylabel("METEOR (%)")
    axs[1][1].grid(True, alpha=0.3)

# Loss comparison in one chart
axs[1][2].plot(train_steps, train_loss, color='blue', lw=1.2, label='Train Loss', alpha=0.7)
if eval_steps:
    axs[1][2].plot(eval_steps, eval_loss, color='orange', marker='o', lw=1.5, label='Val Loss')
axs[1][2].set_title("Train vs Validation Loss", fontsize=13, fontweight='bold')
axs[1][2].set_xlabel("Steps"); axs[1][2].set_ylabel("Loss")
axs[1][2].legend(); axs[1][2].grid(True, alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Saved: {curves_path}")

# =============================================================================
# STEP 19 — Test Set Evaluation
# =============================================================================
print("\n Evaluating on test set...")
test_results = trainer.predict(tokenized['test'])

raw_preds = test_results.predictions
if isinstance(raw_preds, tuple):
    raw_preds = raw_preds[0]

clean_preds = np.where(raw_preds != -100, raw_preds, tokenizer.pad_token_id)
test_preds  = tokenizer.batch_decode(clean_preds, skip_special_tokens=True)

clean_labels = [[t for t in l if t != -100] for l in test_results.label_ids]
test_labels  = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)

test_preds  = [post_process(p.strip()) for p in test_preds]
test_labels = [l.strip() for l in test_labels]

bleu_test   = bleu_metric.compute(predictions=test_preds, references=[[l] for l in test_labels])
rouge_test  = rouge_metric.compute(predictions=test_preds, references=test_labels)
try:
    meteor_test = meteor_metric.compute(predictions=test_preds, references=test_labels)
    meteor_val  = meteor_test['meteor'] * 100
except Exception:
    meteor_val  = 0.0
exact_test  = sum(p == r for p, r in zip(test_preds, test_labels)) / len(test_preds) * 100

print("\n" + "=" * 55)
print("   FINAL TEST RESULTS")
print("=" * 55)
print(f"   BLEU        : {bleu_test['bleu']*100:.2f}%")
print(f"   ROUGE-L     : {rouge_test['rougeL']*100:.2f}%")
print(f"   METEOR      : {meteor_val:.2f}%")
print(f"   Exact Match : {exact_test:.2f}%")
print("=" * 55)

# Save metrics to JSON for app integration
metrics_dict = {
    "bleu":        round(bleu_test['bleu'] * 100, 4),
    "rougeL":      round(rouge_test['rougeL'] * 100, 4),
    "meteor":      round(meteor_val, 4),
    "exact_match": round(exact_test, 4),
}
metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"\n   Metrics saved: {metrics_path}")

# =============================================================================
# STEP 20 — Save Predictions + Error Log
# =============================================================================
test_inputs = [row['input'] for row in dataset_dict['test']]

results_df = pd.DataFrame({
    'input':      test_inputs,
    'reference':  test_labels,
    'prediction': test_preds,
    'correct':    [p == r for p, r in zip(test_preds, test_labels)],
})

preds_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
results_df.to_csv(preds_path, index=False, encoding='utf-8-sig')
print(f"   Predictions saved: {preds_path}")

errors_df = results_df[~results_df['correct']].drop(columns=['correct'])
errors_path = os.path.join(OUTPUT_DIR, "error_log.csv")
errors_df.to_csv(errors_path, index=False, encoding='utf-8-sig')
print(f"   Error log saved  : {errors_path}  ({len(errors_df):,} errors)")

# =============================================================================
# STEP 21 — Metrics Bar Chart
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    metrics_dict.keys(), metrics_dict.values(),
    color=['#2196F3', '#FF9800', '#4CAF50', '#F44336'],
    edgecolor='white', linewidth=1.5
)
for bar, val in zip(bars, metrics_dict.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title("Model Performance on Test Set", fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')
ax.set_facecolor('#f9f9f9')

bar_path = os.path.join(OUTPUT_DIR, "metrics_bar_chart.png")
plt.tight_layout()
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   Bar chart saved  : {bar_path}")

# =============================================================================
# STEP 22 — Sample Predictions Print
# =============================================================================
print("\n Sample Test Predictions:")
print("-" * 70)
for i in range(min(5, len(results_df))):
    row = results_df.iloc[i]
    match = "true" if row['correct'] else "false"
    print(f"  Input     : {row['input'].replace(PREFIX, '')}")
    print(f"  Reference : {row['reference']}")
    print(f"  Predicted : {row['prediction']}  {match}")
    print("-" * 70)

# =============================================================================
# STEP 23 — Generation Function (used in interactive loop)
# =============================================================================
def translate_to_gloss(english_sentence, num_beams=5, num_return=5):
    """
    Translates English to ASL Gloss and returns multiple candidates with confidence.
    Used for sending the best Gloss string to the avatar.
    """
    model.eval()
    
    # 1. Prepare English input (lowercase is better for T5)
    prepared_input = PREFIX + english_sentence.strip().lower()
    
    # Step 2: tokenize
    inputs = tokenizer(
        prepared_input,
        return_tensors="pt",
        max_length=MAX_INPUT,
        truncation=True,
    ).to(device)

    # Step 3: generate top-N candidates
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return,
            output_scores=True,
            return_dict_in_generate=True,
            max_length=MAX_TARGET,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )

    sequences = outputs.sequences
    scores    = outputs.sequences_scores
    decoded   = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    probs     = torch.softmax(scores, dim=0)

    candidates = []
    for sent, score, prob in zip(decoded, scores, probs):
        # CLEANING FOR AVATAR: Uppercase and remove punctuation
        clean_gloss = sent.upper().strip()
        clean_gloss = re.sub(r'[?.!,;:]', '', clean_gloss)
        
        candidates.append({
            "gloss":      clean_gloss,      # This goes to the avatar
            "confidence": float(prob),
            "log_score":  float(score),
        })

    # Sort by confidence descending
    candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)

    return prepared_input.replace(PREFIX, ''), candidates


# =============================================================================
# STEP 24 — Interactive Loop (Fixed for English-to-Gloss)
# =============================================================================
model.eval()
print(f"\n{'='*60}")
print(f"  INTERACTIVE TRANSLATION  |  Device: {device}")
print(f"{'='*60}")
print("  Type English sentences to translate to ASL Gloss for the Avatar.")
print("  Examples: What is your name? | I am not going to school. | I like you.")
print("  Type 'quit' to exit.")
print(f"{'='*60}\n")

while True:
    raw = input("English (or quit): ").strip()

    if not raw:
        continue
    if raw.lower() == 'quit':
        print("Bye!")
        break

    normalized_display, candidates = translate_to_gloss(raw)

    # Use the first (best) candidate
    best = candidates[0]

    print(f"\n   Input received : {raw}")
    print(f"   Normalized to  : {normalized_display}")
    # CHANGED: Use 'gloss' instead of 'sentence'
    print(f"\n   AVATAR GLOSS: {best['gloss']}") 

    print(f"\n   Top {len(candidates)} candidates (debug):")
    for i, c in enumerate(candidates, 1):
        # CHANGED: Use 'gloss' instead of 'sentence'
        print(f"     {i}. {c['gloss']}"
              f"  |  confidence: {c['confidence']:.4f}"
              f"  |  log_score: {c['log_score']:.4f}")
    print()