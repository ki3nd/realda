import os
import json
import random
from pathlib import Path
from typing import List

import configargparse
import torch
from diffusers import StableDiffusion3Pipeline
from dotenv import load_dotenv

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


def parse_args():
    parser = configargparse.ArgParser(
        description="Generate images by class labels using Stable Diffusion.",
        default_config_files=[],
    )
    parser.add(
        "-c",
        "--config",
        is_config_file=True,
        help="Path to a config file.",
    )
    parser.add("--labels-file", type=str, required=True, help="Path to txt file, one label per line.")
    parser.add("--target-path", type=str, required=True, help="Output root directory.")
    parser.add("--device", type=str, default="cuda:0", help='Torch device, e.g. "cuda:0" or "cpu".')
    parser.add("--num-image-each-class", type=int, default=200, help="Number of images generated for each class.")
    parser.add(
        "--domain-config",
        type=str,
        required=True,
        help="Path to domain JSON config containing prompt_templates and class_mapping.",
    )
    parser.add("--num-inference-steps", type=int, default=40, help="Diffusion inference steps.")
    parser.add("--guidance-scale", type=float, default=4.5, help="Classifier-free guidance scale.")
    parser.add(
        "--template-strategy",
        type=str,
        choices=["random", "round_robin"],
        default="random",
        help="How to select prompt templates for each generated image.",
    )
    parser.add("--seed", type=int, default=42, help="Random seed for template selection.")
    return parser.parse_args()


def read_labels(labels_file: str) -> List[str]:
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {labels_file}")
    return labels


def load_domain_config(domain_config_path: str) -> dict:
    with open(domain_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    prompt_templates = config.get("prompt_templates", [])
    if not isinstance(prompt_templates, list) or not prompt_templates:
        raise ValueError(f"Invalid domain config: 'prompt_templates' must be a non-empty list in {domain_config_path}")

    class_mapping = config.get("class_mapping", {})
    if class_mapping is None:
        class_mapping = {}
    if not isinstance(class_mapping, dict):
        raise ValueError(f"Invalid domain config: 'class_mapping' must be a dict in {domain_config_path}")

    return {
        "prompt_templates": prompt_templates,
        "class_mapping": class_mapping,
    }


def normalize_label(raw_label: str) -> str:
    return raw_label.replace("_", " ").strip().lower()


def resolve_class_name(raw_label: str, class_mapping: dict[str, str]) -> str:
    base = normalize_label(raw_label)
    if base in class_mapping:
        return class_mapping[base]
    return base


def build_prompt(name_class: str, prompt_templates: List[str], image_idx: int, strategy: str) -> str:
    if strategy == "round_robin":
        template = prompt_templates[(image_idx - 1) % len(prompt_templates)]
    else:
        template = random.choice(prompt_templates)
    return template.replace("{label}", name_class)


def build_pipeline(device: str):
    is_cuda = device.startswith("cuda")
    dtype = torch.bfloat16 if is_cuda else torch.float32
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Missing Hugging Face token. Set HF_TOKEN in .env/environment.")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=token,
    )
    return pipe.to(device)


def generate_images(args):
    random.seed(args.seed)
    labels = read_labels(args.labels_file)
    domain_cfg = load_domain_config(args.domain_config)
    prompt_templates = domain_cfg["prompt_templates"]
    class_mapping = {str(k).strip().lower(): str(v).strip() for k, v in domain_cfg["class_mapping"].items()}

    pipe = build_pipeline(args.device)
    output_root = Path(args.target_path)
    output_root.mkdir(parents=True, exist_ok=True)

    for raw_label in labels:
        label = raw_label.strip()
        label_dir = output_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        name_class = resolve_class_name(label, class_mapping)
        print(f"Generating class: {name_class}")

        for idx in range(1, args.num_image_each_class + 1):
            prompt = build_prompt(name_class, prompt_templates, idx, args.template_strategy)
            image = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            ).images[0]
            image.save(label_dir / f"{idx}.png")


def main():
    load_dotenv()
    args = parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
