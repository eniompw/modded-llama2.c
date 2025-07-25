# TinyStories Dataset Info

This document explains the data from the [TinyStories dataset on Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories), providing a brief overview of the structure and content of the `data00.json` file.

## Data Source

It was downloaded using `wget`:
```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz
```

And extracted with `tar`:
```bash
tar -I pigz -xvf TinyStories_all_data.tar.gz ./data{00..01}.json
```

## File Structure

The file `data00.json` contains a JSON array. We can count the number of entries using `jq`:
```bash
jq 'length' data00.json
```
This shows there are **100,000** entries. Each entry is a JSON object representing a single data record.

## Entry Structure

Each object in the array has four keys. We can see them by inspecting the first entry:
```bash
jq '.[0] | keys' data00.json
```
The keys are:

*   `story`: (String) A short story, often written with simple language.
*   `instruction`: (Object) Contains the generation parameters for the story.
    *   `prompt`: (String) The detailed prompt used to generate the story, including constraints on vocabulary, length, and features.
    *   `words`: (Array of Strings) A list of specific words that must be included in the story.
    *   `features`: (Array of Strings) A list of specific literary features the story must contain (e.g., "Dialogue").
*   `summary`: (String) A brief one-sentence summary of the story.
*   `source`: (String) The source of the data entry (e.g., "GPT-4").

### Example Entry

We can view the first full entry with:
```bash
jq '.[0]' data00.json
```

```json
{
  "story": "\n\nLily and Ben are friends. They like to play in the park. One day, they see a big tree with a swing. Lily wants to try the swing. She runs to the tree and climbs on the swing.\n\"Push me, Ben!\" she says. Ben pushes her gently. Lily feels happy. She swings higher and higher. She laughs and shouts.\nBen watches Lily. He thinks she is cute. He wants to swing too. He waits for Lily to stop. But Lily does not stop. She swings faster and faster. She is having too much fun.\n\"Can I swing too, Lily?\" Ben asks. Lily does not hear him. She is too busy swinging. Ben feels sad. He walks away.\nLily swings so high that she loses her grip. She falls off the swing. She lands on the ground. She hurts her foot. She cries.\n\"Ow, ow, ow!\" she says. She looks for Ben. She wants him to help her. But Ben is not there. He is gone.\nLily feels sorry. She wishes she had shared the swing with Ben. She wishes he was there to hug her. She limps to the tree. She sees something hanging from a branch. It is Ben's hat. He left it for her.\nLily smiles. She thinks Ben is nice. She puts on his hat. She hopes he will come back. She wants to say sorry. She wants to be friends again.",
  "instruction": {
    "prompt:": "Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would understand. The story should use the verb \"hang\", the noun \"foot\" and the adjective \"cute\". The story has the following features: the story should contain at least one dialogue. Remember to only use simple words!\n\nPossible story:",
    "words": [
      "hang",
      "foot",
      "cute"
    ],
    "features": [
      "Dialogue"
    ]
  },
  "summary": "Lily and Ben play in the park and Lily gets too caught up in swinging, causing Ben to leave. Lily falls off the swing and hurts herself, but Ben leaves his hat for her as a kind gesture.",
  "source": "GPT-4"
}
```

## Pre-tokenization

The `.json` files are processed to create binary (`.bin`) files that are used for training the model. This is done using the `pretokenize` command:

```bash
python tinystories.py pretokenize
```

This script performs the following steps:

1.  It reads each `dataXX.json` file.
2.  For each entry in the JSON array, it extracts the `story` text.
3.  The story text is then tokenized using the Llama 2 tokenizer. This converts the text into a sequence of integers.
4.  A special "beginning of sentence" (BOS) token is added to the start of each tokenized story.
5.  All the tokenized stories from a single `.json` file are concatenated into one long sequence of integers.
6.  This sequence is saved as a binary file with the same name, but with a `.bin` extension (e.g., `data00.json` is processed into `data00.bin`).

These `.bin` files are the final pre-processed data used for training the language model.
