import torch as th
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

allowed_choices_and_meaning = [
    ("A", "finance"),
    ("B", "IT"),
    ("C", "HR"),
    ("D", "cannot answer")
]

# ===================================================================================
# Find token IDs corresponding ONLY to the allowed choices.
# Each choice must map to exactly one token (no BOS or other special tokens).
# The order of token IDs must match the order of allowed_choices.
# ===================================================================================
allowed_choices = [choice for choice, _ in allowed_choices_and_meaning]
allowed_token_ids = []
for choice in allowed_choices:
    ids = tokenizer(choice, add_special_tokens=False).input_ids
    assert len(ids) == 1, f"'{choice}' is not a single token: {ids}"
    allowed_token_ids.append(ids[0])

query_prompt = f"""Please classify the following query to be redirected to one of the company departments.
Who should respond to the customer's query?
Answer with only a single capital letter, defining the department. No other output allowed.
Query: "Can you grant me access to the database system?"
"""

for choice, meaning in allowed_choices_and_meaning:
    query_prompt += f"\n{choice} - {meaning}"

messages = [
    {"role": "system", "content": "You are a helpful assistant. Stick strictly to the instructions."},
    {"role": "user", "content": query_prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

with th.no_grad():
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    logits = model(**model_inputs).logits

# ===================================================================================
# Key step: Get the probability distribution over the allowed choices ONLY!
# Index [0, -1] targets the last prompt token (predicting the next token).
# Cast to float32 to avoid bf16 rounding in softmax.
# ===================================================================================
last_logits = logits[0, -1].float()
probability = th.softmax(last_logits[allowed_token_ids], dim=-1)

# Sanity check: how much of the FULL next-token distribution falls on the allowed tokens.
# A low value means the model wanted to output something else, so treat the
# renormalized probabilities above with caution.
allowed_mass = th.softmax(last_logits, dim=-1)[allowed_token_ids].sum().item()

for choice, prob in zip(allowed_choices, probability.tolist()):
    print(f"{choice}: {prob:.3f}")

print(f"Most likely choice: {allowed_choices[probability.argmax().item()]}")
print(f"Probability mass on allowed tokens: {allowed_mass:.3f}")

"""Expected outputs
A: 0.016
B: 0.979
C: 0.004
D: 0.001
Most likely choice: B
Probability mass on allowed tokens: 0.996
"""