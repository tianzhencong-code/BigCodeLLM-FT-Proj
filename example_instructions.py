# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions = [
        [
            {
                "role": "user",
                "content": "In Bash, how do I list all text files in the current directory (excluding subdirectories) that have been modified in the last month?",
            }
        ],
        [
            {
                "role": "user",
                "content": "What is the difference between ineli and eli5?",
            }
        ],
        [
            {
                "role": "system",
                "content": "Provide answers in JavaScript",
            },
            {
                "role": "user",
                "content": "Write a function that computes the set of sums of all contiguous sublists of a given list.",
            },
        ],
        [
            {
                "role": "system",
                "content": "Provide answers in JavaScript",
            },
            {
                "role": "user",
                "content": "Write a function that computes the set of sums of all contiguous sublists of a given list.",
            },
            {
                "role": "assistant",
                "content": """Here is a function in JavaScript that computes the set of sums of all contiguous sublists of a given list:

```javascript
function contiguousSublistSums(list) {
  const sums = new Set();
  for (let i = 0; i < list.length; i++) {
    let sum = 0;
    for (let j = i; j < list.length; j++) {
      sum += list[j];
      sums.add(sum);
    }
  }
  return sums;
}
```

This function uses two nested loops to iterate over all possible contiguous sublists of the input list. For each sublist, it computes the sum and adds it to a set. The set is used to ensure that each sum is only counted once, even if there are multiple sublists with the same sum.

Here is an example usage of the function:

```javascript
const list = [1, 2, 3, 4, 5];
const sums = contiguousSublistSums(list);
console.log(sums); // Set { 1, 3, 6, 10, 15, 2, 5, 9, 14, 3, 7, 12, 4, 9, 5 }
```""",
            },
            {
                "role": "user",
                "content": "Now write it in Rust",
            },
        ],
    ]
    results = generator.chat_completion(
        instructions,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for instruction, result in zip(instructions, results):
        for msg in instruction:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
