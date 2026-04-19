import argparse

from petals.client.biology import load_biology_model
from petals.constants import DEFAULT_BIO_MODEL_NAME


def main():
    parser = argparse.ArgumentParser(description="Run a biology-oriented model over the Petals distributed network")
    parser.add_argument("--model", default=DEFAULT_BIO_MODEL_NAME, help="Hugging Face model ID")
    parser.add_argument(
        "--prompt",
        default="Explain how CRISPR-Cas9 can introduce an insertion mutation during DNA repair.",
        help="Prompt to send to the model",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated models")
    parser.add_argument("--max_new_tokens", type=int, default=192, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Penalty for repeated tokens")
    parser.add_argument(
        "--plain_prompt",
        action="store_true",
        help="Skip chat-template formatting and send the prompt as-is",
    )
    args = parser.parse_args()

    tokenizer, model = load_biology_model(args.model, token=args.token)

    if not args.plain_prompt and hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = args.prompt

    inputs = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    outputs = model.generate(
        inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    completion = tokenizer.decode(outputs[0, inputs.shape[-1] :], skip_special_tokens=True).strip()
    print(completion)


if __name__ == "__main__":
    main()
