"""
Implementation of the label generation part for Hacker News post classification
using `transformers` and DeepSeek.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import contextlib
import math
from tqdm.auto import tqdm
import json
import polars as pl
from datasets import Dataset, Value, ClassLabel


JSON_PATTERN = re.compile(r"```json\n(.*?)```", re.DOTALL)
DIRECT_JSON_PATTERN = re.compile(r"\{[^}]*\}", re.DOTALL)
BATCH_SIZE = 64
NUM_SAMPLES = 3000


@torch.no_grad()
def load_model():
    repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer


def format_text_as_prompt(title: str):
    categories = """
    1. dev: Programming languages, coding practices, software development techniques  
    2. web: Web development, browsers, frontend/backend frameworks, web standards
    3. ai_ml: Artificial intelligence, machine learning, data science
    4. security: Cybersecurity, privacy, vulnerabilities, authentication
    5. business: Startups, companies, funding, acquisitions, industry news
    6. career: Job seeking, workplace discussions, professional development
    7. science: Research, space exploration, physics, biology, academic papers
    8. tools: Development tools, utilities, software applications
    9. culture: Tech industry trends, social impact, community issues
    10. tech_news: General technology news and updates
    """
    return f"""Look at the following Hacker News post title and classify it into one of the categories listed below:

Title: {title}

{categories}

Your role is to decide which category the post belongs to based on the title. First you should think about which category is most relevant to the post. You should then return your reasoning and the label you've chosen.

Return your reasoning and the label you've chosen as a JSON object like this:
```json
{{
    "label": "dev" | "web" | "ai_ml" | "security" | "business" | "career" | "science" | "tools" | "culture" | "tech_news",
    "explanation": "The reasoning the model used to come to its conclusion"
}}
```
"""


def load_dataset():
    # Load your Hacker News dataset here
    # For example, if you have a Polars DataFrame `df` with columns 'title' and 'score'
    df = pl.scan_parquet("../data/hackernews_filtered.parquet")
    return df


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
@torch.no_grad()
def predict_label_without_structured_output(data: list[str], model: torch.nn.Module, tokenizer) -> str:
    prompts = [format_text_as_prompt(d) for d in data]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        for prompt in prompts
    ]
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,  # important so they line up in a batch
        truncation=True,  # so they don't exceed model's max length
    ).to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
    results_ids = []
    for i, output_ids in enumerate(generated_ids):
        input_len = len(model_inputs.input_ids[i])
        results_ids.append(output_ids[input_len:])
    outputs = tokenizer.batch_decode(results_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return outputs


def try_extract_json_from_text(text: str) -> tuple[str, dict | None]:
    if match := JSON_PATTERN.search(text):
        json_results = match.group(1)
        with contextlib.suppress(json.JSONDecodeError):
            return text, json.loads(json_results)
    if match := DIRECT_JSON_PATTERN.search(text):
        json_text = match.group(0)
        with contextlib.suppress(json.JSONDecodeError):
            return text, json.loads(json_text)
    return text, None


def create_and_push_ds(df):
    ds = Dataset.from_polars(
        df.select(["title", "labels", "explanations"]),
    )
    large_string_columns = [k for k, v in ds.features.items() if isinstance(v, Value) and v.dtype == "large_string"]
    for column in large_string_columns:
        ds = ds.cast_column(column, Value("string"))
    ds = ds.cast_column("labels", ClassLabel(names=["dev", "web", "ai_ml", "security", "business", "career", "science", "tools", "culture", "tech_news"]))
    ds.push_to_hub("your-username/hackernews-classification")


def chunked(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def main():
    df = load_dataset()
    model, tokenizer = load_model()

    sample_df = df.sample(NUM_SAMPLES, seed=42)
    examples = sample_df.select(pl.col(["title"])).to_lists()[0]
    total_batches = math.ceil(len(examples) / BATCH_SIZE)

    # run _sample
    raw_predictions = []
    for i, batch_examples in enumerate(tqdm(chunked(examples, BATCH_SIZE), total=total_batches)):
        preds = predict_label_without_structured_output(batch_examples, model, tokenizer)
        raw_predictions.extend(preds)

    parsed_results = [try_extract_json_from_text(result) for result in raw_predictions]
    labels_and_explanations = [
        (result[1].get("label"), result[1].get("explanation"))
        if result[1] is not None and isinstance(result[1], dict)
        else (None, None)
        for result in parsed_results
    ]

    # Unzip the list of tuples into separate lists
    labels, explanations = zip(*labels_and_explanations)
    labels = list(labels)
    explanations = list(explanations)
    sample_df = sample_df.with_columns(
        pl.Series(labels).alias("labels"),
        pl.Series(explanations).alias("explanations"),
    )

    create_and_push_ds(sample_df)


if __name__ == "__main__":
    main()