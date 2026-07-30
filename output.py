import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

INPUT_FILE = "data/processed-trace.jsonl"
OUTPUT_FILE = "data/predictions-baseline.jsonl"


def generate(messages, model, max_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            req = json.loads(line)
            body = req["body"]

            answer = generate(
                messages=body["messages"],
                model="Qwen/Qwen3.5-2B",
                max_tokens=body["max_tokens"],
            )

            output = {
                "request_id": req["request_id"],
                "prediction": answer,
            }

            fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()