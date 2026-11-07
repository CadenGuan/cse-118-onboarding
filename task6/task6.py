import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"
)

prompts = [
    # --- Category A: Open-Ended Questions ---
    {
        "category": "A - Open-Ended",
        "prompt": "Please summarize the primary causes of the 2008 global financial crisis in exactly three paragraphs. Be sure to mention the roles of subprime mortgages, collateralized debt obligations (CDOs), and the failure of major financial institutions."
    },
    {
        "category": "A - Open-Ended",
        "prompt": "Explain the concept of quantum entanglement to a 10-year-old child. You must use an everyday analogy that a child would easily understand (for example, a pair of magic gloves or a pair of socks) and avoid using complex physics jargon."
    },
    {
        "category": "A - Open-Ended",
        "prompt": "Compare the ethical implications of autonomous vehicles versus human drivers in edge-case accident scenarios. Specifically, discuss the 'Trolley Problem' variations that AI systems must be programmed to handle, and argue whether an AI should prioritize the lives of its passengers or pedestrians."
    },

    # --- Category B: Rule-Based Reasoning ---
    {
        "category": "B - Rule-Based",
        "prompt": "You are a formal logic engine. Follow the rules of syllogism.\nRule: If All X are Y, and All Y are Z, then All X are Z.\n\nExample 1: All dogs are mammals. All mammals are animals. Therefore, all dogs are animals.\nExample 2: All roses are flowers. All flowers are plants. Therefore, all roses are plants.\n\nNew Case to Solve: All robots are machines. All machines require electricity. Therefore, what can we conclude about robots? Answer with the conclusion only."
    },
    {
        "category": "B - Rule-Based",
        "prompt": "You are a math tutor. Solve word problems by breaking them down step-by-step.\nRule: First, identify the total budget. Second, subtract the fixed costs. Third, divide the remaining budget by the cost per item to find the maximum quantity.\n\nExample: A student has $50. They spend $10 on a textbook. They want to buy notebooks that cost $5 each. $50 - $10 = $40. $40 / $5 = 8 notebooks.\n\nNew Case to Solve: A club has $150. They spend $30 on venue rental. They want to buy pizzas that cost $12 each. Show your step-by-step math to find how many pizzas they can buy."
    },
    {
        "category": "B - Rule-Based",
        "prompt": "You are a data parsing assistant. Convert messy text into a strict CSV format.\nRule: Extract the Name, Age, and City. Format as: Name, Age, City. Do not include any other text.\n\nExample Input: 'Hey, my name is John, I am 25 years old, and I live in New York.'\nExample Output: John, 25, New York\n\nNew Input to Process: 'Hi there! I'm Sarah, currently 30 years of age, and I'm based out of London.' Output:"
    },

    # --- Category C: Single-Word Constrained Reasoning ---
    {
        "category": "C - Single-Word",
        "prompt": "Read the premise and hypothesis carefully. Determine the relationship. Answer strictly with exactly one word: 'Entailment', 'Contradiction', or 'Neutral'. Do not use any punctuation or extra text.\nPremise: A man is walking a black dog in the park.\nHypothesis: The man is walking a pet.\nAnswer:"
    },
    {
        "category": "C - Single-Word",
        "prompt": "Evaluate the scenario for plausibility. Answer strictly with exactly one word: 'Possible' or 'Impossible'. Do not use any punctuation or extra text.\nScenario: A person is swimming deep underwater in the ocean without any scuba gear or breathing apparatus for 45 minutes.\nAnswer:"
    },
    {
        "category": "C - Single-Word",
        "prompt": "Read the multi-step statement and evaluate its truth value. Answer strictly with exactly one word: 'True' or 'False'. Do not use any punctuation or extra text.\nStatement: If all birds can fly, and penguins are birds, then penguins can fly.\nAnswer:"
    }
]


for i, item in enumerate(prompts):
    prompt_text = item["prompt"]
    category = item["category"]

    print(f"{'='*50}")
    print(f"PROMPT {i+1} [{category}]")
    print(f"{'='*50}")
    print(f"INPUT:\n{prompt_text}\n")

    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": prompt_text}
    ]

    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256, # Limits the length to prevent rambling
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"OUTPUT:\n{response}\n")


