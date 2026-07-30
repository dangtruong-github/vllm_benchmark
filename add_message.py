import argparse
import json


def get_system_message(max_tokens: int) -> str:
    return (
        "Answer directly. Start with the answer, not with meta-commentary such as "
        "'Based on...', 'According to...', or 'The answer is...'. "
        "Explain the answer briefly when necessary, but do not repeat the question "
        "or restate the same point. "
        f"Keep the response concise and within {max_tokens} tokens."
    )


def add_system_instruction(messages: list, max_tokens: int) -> list:
    new_instruction = get_system_message(max_tokens)

    # Find an existing system message.
    for message in messages:
        if message.get("role") == "system":
            existing_content = message.get("content", "")

            # Append the new instruction to the existing system message.
            if existing_content:
                message["content"] = (
                    existing_content.rstrip()
                    + "\n\n"
                    + new_instruction
                )
            else:
                message["content"] = new_instruction

            return messages

    # No system message exists, so add one at the beginning.
    return [
        {
            "role": "system",
            "content": new_instruction,
        }
    ] + messages


def process_jsonl(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"Skipping invalid JSON at line {line_number}: {e}"
                )
                continue

            body = data.get("body", {})

            # Get max_tokens from the request body.
            # Default to 200 if it is missing.
            max_tokens = body.get("max_tokens", 200)

            messages = body.get("messages", [])

            # Preserve existing system message and append
            # the new instruction, or create a new system message.
            body["messages"] = add_system_instruction(
                messages,
                max_tokens,
            )

            data["body"] = body

            # Write one JSON object per line.
            fout.write(
                json.dumps(data, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Add a dynamic instruction to the system message "
            "in each JSONL request."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file",
    )

    args = parser.parse_args()

    process_jsonl(
        input_file=args.input,
        output_file=args.output,
    )

    print(f"Processed: {args.input}")
    print(f"Output:    {args.output}")
