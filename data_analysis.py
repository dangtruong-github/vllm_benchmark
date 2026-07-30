import json


INPUT_FILE = "data/processed-trace.jsonl"
GROUP_SIZE = 20


def count_input_tokens(messages):
    """
    Approximate input token count.

    This is only a rough estimate using whitespace splitting.
    For exact token counts, use the same tokenizer as the model.
    """
    return sum(
        len(message.get("content", "").split())
        for message in messages
        if isinstance(message.get("content"), str)
    )


groups = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    requests = [
        json.loads(line)
        for line in file
        if line.strip()
    ]


for group_start in range(0, len(requests), GROUP_SIZE):

    group = requests[
        group_start:group_start + GROUP_SIZE
    ]

    if not group:
        continue

    input_tokens = []
    decode_tokens = []

    cur_mes = None

    for data in group:
        messages = data["body"]["messages"]

        if cur_mes is not None:
            if cur_mes == messages:
                print(
                    f"Request {data['request_id']}: SAME"
                )
            else:
                print(
                    f"Request {data['request_id']}: DIFFERENT"
                )

        cur_mes = messages

        # Approximate prefill/input token count.
        input_token_count = count_input_tokens(messages)

        input_tokens.append(
            input_token_count
        )

        # Requested maximum decode tokens.
        decode_tokens.append(
            data["body"].get("max_tokens", 0)
        )

    group_id = (
        group_start // GROUP_SIZE
    ) + 1

    print(
        f"Group {group_id} "
        f"(requests {group_start}-"
        f"{group_start + len(group) - 1})"
    )

    print(
        f"  Prefill tokens : "
        f"{min(input_tokens):,}-"
        f"{max(input_tokens):,}"
    )

    print(
        f"  Decode tokens  : "
        f"{min(decode_tokens):,}-"
        f"{max(decode_tokens):,}"
    )

    print(
        f"  Requests       : "
        f"{len(group)}"
    )

    print()