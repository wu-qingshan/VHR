import os
import json
import copy
import random
import datetime
import argparse
from math import ceil
from tqdm import tqdm

import torch

from datasets import *
from generation import replace_generation

from utils import get_model, add_diffusion_noise, get_layers
from eah import replace_attention_with_eah

replace_generation()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="path to the model")
    parser.add_argument("--num_chunks", default=8, type=int, help="number of chunks for parallel inference")
    parser.add_argument("--chunk_idx", default=0, type=int, help="chunk index for parallel inference")
    parser.add_argument("--device", default="cuda:0", type=str, help="device")
    parser.add_argument("--method", default="vhr", type=str, help="method to use for hallucination mitigation")
    parser.add_argument("--vhr_aug_ratio", default=2, type=float, help="augmentation ratio for VHR (default: 2, 1 means no augmentation)")
    parser.add_argument("--vhr_layers", default=14, type=int, help="number of the last few layers to apply VHR")
    parser.add_argument("--vhr_layer1", action="store_true", default=False, help="whether to apply VHR on layer 1")
    parser.add_argument("--vhr_filter", action="store_true", default=False, help="whether use filter for VHR")
    # A-VHR (Adaptive-ratio VHR, see optim-opus.md 方法 1)
    parser.add_argument("--vhr_tau", default=1.0, type=float, help="A-VHR sigmoid 温度（越小越接近原版硬阈值）")
    parser.add_argument("--vhr_alpha", default=0.5, type=float, help="A-VHR 层深衰减强度，0 表示不衰减")
    parser.add_argument("--vhr_rho_max", default=2.0, type=float, help="A-VHR 单 head 最大加权倍率")
    parser.add_argument("--vhr_score_mode", default="magnitude", choices=["magnitude", "directional", "directional_linear", "delayed_directional_linear", "calibrated", "consensus"], help="head selection score used by VHR")
    parser.add_argument("--vhr_head_fraction", default=0.5, type=float, help="fraction of heads selected in each VHR layer")
    parser.add_argument("--vhr_positive_only", action="store_true", default=False, help="only select heads with a positive directional score")
    parser.add_argument("--vhr_binary_fallback", action="store_true", default=False, help="use original magnitude VHR for binary questions")
    parser.add_argument("--vhr_contrast_mode", default="remove", choices=["remove", "zero"], help="visual counterfactual used to score heads")
    parser.add_argument("--vhr_directional_delay", default=64, type=int, help="generated tokens before switching from magnitude to directional heads")
    parser.add_argument("--vhr_noise_step", default=500, type=int, help="diffusion noise step used by CC-VHR's noisy counterfactual")
    parser.add_argument("--cd_alpha", default=0.5, type=float, help="VCD contrast strength")
    parser.add_argument("--cd_beta", default=0.1, type=float, help="VCD plausibility cutoff ratio")
    parser.add_argument("--vhr_logit_alpha", default=0.5, type=float, help="Self-Contrast VHR logit contrast strength")
    parser.add_argument("--vhr_logit_beta", default=0.1, type=float, help="Self-Contrast VHR plausibility cutoff ratio")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max new tokens to generate")
    parser.add_argument("--no_repeat_ngram_size", default=0, type=int, help="standard generation no-repeat ngram constraint")
    parser.add_argument("--repetition_penalty", default=1.0, type=float, help="standard generation repetition penalty")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--output_dir", default="results", type=str, help="output directory")
    parser.add_argument("--datasets", default="chair,pope_popular,pope_adversarial,pope_random", type=str, help="comma-separated datasets to infer")
    parser.add_argument("--chair_seeds", default="0,1,2,3,4", type=str, help="comma-separated CHAIR seeds")
    parser.add_argument("--chair_samples", default=500, type=int, help="number of randomly sampled CHAIR images")
    parser.add_argument("--chair_ids_file", default=None, type=str, help="optional JSON file containing fixed COCO image ids")
    return parser


class HallucinationTest:

    def __init__(self, args):
        attn_impl = "eager" if "eah" in args.method else "sdpa"
        self.device = torch.device(args.device)
        self.processor, self.model, self.template, self.ans_start, self.num_img_tokens, self.image_token_id = get_model(args.model_path, self.device, attn_impl)
        # 解析为短目录名，避免绝对路径整串落到 results/ 下
        _mp = os.path.realpath(os.path.expanduser(args.model_path))
        _base = os.path.basename(_mp.rstrip(os.sep))
        self.model_name = _base.split("--")[-1] if "--" in _base else _base
        self.output_dir = args.output_dir
        self.chunk_idx = args.chunk_idx
        self.num_chunks = args.num_chunks
        self.method = args.method
        self.max_new_tokens = args.max_new_tokens
        self.no_repeat_ngram_size = args.no_repeat_ngram_size
        self.repetition_penalty = args.repetition_penalty

        self.vhr_aug_ratio = args.vhr_aug_ratio
        self.vhr_layers = args.vhr_layers
        self.vhr_layer1 = args.vhr_layer1
        self.vhr_filter = args.vhr_filter
        self.vhr_tau = args.vhr_tau
        self.vhr_alpha = args.vhr_alpha
        self.vhr_rho_max = args.vhr_rho_max
        self.vhr_score_mode = args.vhr_score_mode
        self.vhr_head_fraction = args.vhr_head_fraction
        self.vhr_positive_only = args.vhr_positive_only
        self.vhr_binary_fallback = args.vhr_binary_fallback
        self.vhr_contrast_mode = args.vhr_contrast_mode
        self.vhr_directional_delay = args.vhr_directional_delay
        self.vhr_noise_step = args.vhr_noise_step
        self.cd_alpha = args.cd_alpha
        self.cd_beta = args.cd_beta
        self.vhr_logit_alpha = args.vhr_logit_alpha
        self.vhr_logit_beta = args.vhr_logit_beta
        self.chair_samples = args.chair_samples
        self.chair_ids = None
        if args.chair_ids_file is not None:
            with open(args.chair_ids_file, "r") as f:
                self.chair_ids = json.load(f)
    
    def infer_dataset(self, dataset_name, seed):
        output_file_path = f"{self.output_dir}/{self.model_name}/{self.chunk_idx}-{dataset_name}_{seed}-{self.method}-{self.max_new_tokens}.json"
        complete_output_file_path = f"{self.output_dir}/{self.model_name}/{dataset_name}_{seed}-{self.method}-{self.max_new_tokens}.json"
        
        if (os.path.isfile(output_file_path) and len(open(output_file_path, "r").readlines()) > 0) \
            or (os.path.isfile(complete_output_file_path) and len(open(complete_output_file_path, "r").readlines()) > 0):
            return

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        random.seed(seed)
        if dataset_name == "chair":
            dataset = CHAIRBench(num_samples=self.chair_samples, image_ids=self.chair_ids)
        else:
            dataset = load_dataset(dataset_name)

        generate_kwargs = {
            "num_beams": 5 if "beam" in self.method else 1, 
            "do_sample": "sample" in self.method,
            "max_new_tokens": self.max_new_tokens,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_penalty": self.repetition_penalty,
        }
        
        if "vhr" in self.method:
            vhr_layers = list(range(len(get_layers(self.model))))[-self.vhr_layers:] if self.vhr_layers > 0 else []
            vhr_layers = [1] + vhr_layers if self.vhr_layer1 else vhr_layers

            generate_kwargs.update(
                {
                    "vhr_aug_ratio": self.vhr_aug_ratio,
                    "vhr_layers": vhr_layers,
                    "vhr_filter": self.vhr_filter,
                    "vhr_score_mode": self.vhr_score_mode,
                    "vhr_head_fraction": self.vhr_head_fraction,
                    "vhr_positive_only": self.vhr_positive_only,
                    "vhr_contrast_mode": self.vhr_contrast_mode,
                    "vhr_directional_delay": self.vhr_directional_delay,
                }
            )
            if "csvhr" in self.method:
                generate_kwargs.update(
                    {
                        "vhr_score_mode": "directional",
                        "vhr_positive_only": True,
                    }
                )
            # 若 method 含 "avhr"，启用 A-VHR：连续 sigmoid 头加权 + 层深衰减
            if "avhr" in self.method:
                generate_kwargs.update(
                    {
                        "vhr_use_avhr": True,
                        "vhr_tau": self.vhr_tau,
                        "vhr_alpha": self.vhr_alpha,
                        "vhr_rho_max": self.vhr_rho_max,
                    }
                )
            if "scvhr" in self.method:
                generate_kwargs.update(
                    {
                        "vhr_base_contrast": True,
                        "vhr_logit_alpha": self.vhr_logit_alpha,
                        "vhr_logit_beta": self.vhr_logit_beta,
                    }
                )
        elif "dola" in self.method:
            self.model.config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
            generate_kwargs.update(
                {
                    "num_beams": 1,
                    "dola_layers": [0, 2, 4, 6, 8, 10, 12, 14],
                }
            )

        chunk_size = ceil(len(dataset.all) / self.num_chunks)
        output_file = open(output_file_path, "w")

        start = datetime.datetime.now()
        for qid in tqdm(dataset.all[self.chunk_idx*chunk_size:(self.chunk_idx+1)*chunk_size], ncols=100, desc=f"Seed {seed} Chunk {self.chunk_idx} Inference"):

            # prepare for inputs
            img, qu, gt = dataset[qid]
            if "vhr" in self.method:
                generate_kwargs["vhr_score_mode"] = self.vhr_score_mode
                generate_kwargs["vhr_head_fraction"] = self.vhr_head_fraction
                generate_kwargs["vhr_positive_only"] = self.vhr_positive_only
                if "csvhr" in self.method:
                    generate_kwargs["vhr_score_mode"] = "directional"
                    generate_kwargs["vhr_positive_only"] = True
                if self.vhr_binary_fallback and is_binary_question(qu):
                    generate_kwargs["vhr_score_mode"] = "magnitude"
                    generate_kwargs["vhr_head_fraction"] = 0.5
                    generate_kwargs["vhr_positive_only"] = False
            prompt = self.template.format(question=qu)
            inputs = self.processor(text=prompt, images=img, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            img_start, img_end, respond_start, img_token_end = self.get_image_position(inputs['input_ids'])

            if "vhr" in self.method:
                generate_kwargs.update(
                    {
                        "vhr_image_start": img_start,
                        "vhr_image_end": img_token_end,
                    }
                )

            if generate_kwargs.get("vhr_score_mode") == "consensus":
                noise_seed = (int(seed) * 1_000_003 + int(qid)) % (2**63 - 1)
                noise_generator = torch.Generator(device=inputs["pixel_values"].device)
                noise_generator.manual_seed(noise_seed)
                images_noisy = add_diffusion_noise(
                    inputs["pixel_values"], noise_step=self.vhr_noise_step, generator=noise_generator
                ).to(self.device)

                if "instructblip" in self.model_name:
                    inputs_noisy = copy.deepcopy(inputs)
                    inputs_noisy["pixel_values"] = images_noisy
                    with torch.inference_mode():
                        query_output = self.model(**inputs_noisy).qformer_outputs[0][
                            :, : self.model.query_tokens.size(1), :
                        ]
                        images_noisy = self.model.language_projection(query_output)
                generate_kwargs["vhr_images_noisy"] = images_noisy

            if "vcd" in self.method:
                noise_seed = (int(seed) * 1_000_003 + int(qid)) % (2**63 - 1)
                noise_generator = torch.Generator(device=inputs["pixel_values"].device)
                noise_generator.manual_seed(noise_seed)
                images_cd = add_diffusion_noise(inputs["pixel_values"], noise_step=500, generator=noise_generator)
                images_cd = images_cd.to(self.device)
                
                if "instructblip" in self.model_name:
                    inputs_cd = copy.deepcopy(inputs)
                    inputs_cd['pixel_values'] = images_cd
                    with torch.inference_mode():
                        query_output = self.model(**inputs_cd).qformer_outputs[0][:, : self.model.query_tokens.size(1), :]
                        images_cd = self.model.language_projection(query_output)

                generate_kwargs.update(
                    {
                        "num_beams": 1,
                        "images_cd": images_cd,
                        "cd_alpha": self.cd_alpha,
                        "cd_beta": self.cd_beta,
                    }
                )

            elif "opera" in self.method:
                generate_kwargs.update(
                    {
                        "num_beams": 5,
                        "output_attentions": True,
                        "opera_key_position": 
                        {
                            "image_start": img_start,           # position in hidden_states
                            "image_end": img_end,               # position in hidden_states
                            "response_start": respond_start,    # position in hidden_states
                        }
                    }
                )

            elif "code" in self.method:
                prompt = "Provide a detailed description of the image, covering all visible elements and their interactions, so as to thoroughly answer any potential questions about the image."#"Describe the image with enough detail to answer any questions about the given image."
                prompt = self.template.format(question=prompt)
                inputs_desc = self.processor(text=prompt, images=img, return_tensors="pt")
                inputs_desc = {key: value.to(self.device) for key, value in inputs_desc.items()}
                
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs_desc, max_new_tokens=512, output_scores=True, return_dict_in_generate=True)
                
                ans_len = len(outputs.scores)
                desc = self.processor.batch_decode(outputs.sequences[:, -ans_len:], skip_special_tokens=True)[0]
                
                generate_kwargs.update(
                    {
                        "num_beams": 1,
                        "code_instead_images": outputs.sequences[:, -ans_len:],
                        "code_image_start": img_start,      # start in input_ids
                        "code_image_end": img_token_end,    # end in input_ids
                    }
                )

            elif "eah" in self.method:
                replace_attention_with_eah(self.model, beta=0.002, img_start=img_start, img_end=img_end-1, layers_to_modify=[1])

            inputs = {key: value.to(self.device) for key, value in inputs.items()}
           
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **generate_kwargs,  output_scores=True, return_dict_in_generate=True)

            gen_answer = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            gen_answer = gen_answer.split(self.ans_start)[1].strip() if self.ans_start is not None else gen_answer.strip()

            probs = outputs.scores[0][0].softmax(-1)
            ans_conf = probs.topk(1)[0].cpu().numpy()[0]
            ans_conf = str(ans_conf*100)[:5]

            output_file.write(json.dumps({"question_id": qid, "text": gen_answer, "gt": gt, "ans_conf": ans_conf, "question": qu})+"\n")

        output_file.close()
        end = datetime.datetime.now()
        print(f"Method: {self.method}\tChunk Idx: {self.chunk_idx}\tInference Time: {(end-start).seconds} s")

    def get_image_position(self, input_ids):
        if self.num_img_tokens is None:
            self.num_img_tokens = (input_ids==self.image_token_id).nonzero().shape[0]

        if self.image_token_id is None:
            # instructblip
            img_start = 0
            respond_start = input_ids.shape[1] + self.num_img_tokens
            img_token_end = None
        else:
            # llava
            img_start = (input_ids==self.image_token_id).nonzero()[0,1].item()
            respond_start = input_ids.shape[1] - (input_ids==self.image_token_id).nonzero().shape[0] + self.num_img_tokens
            img_token_end = (input_ids==self.image_token_id).nonzero()[-1, 1].item() + 1
        
        img_end = img_start + self.num_img_tokens


        return img_start, img_end, respond_start, img_token_end
    

def load_dataset(dataset_name):
    if "pope" in dataset_name:
        dataset = POPE(subset=dataset_name)
    elif dataset_name == "chair":
        dataset = CHAIRBench(500)
    elif dataset_name == "llava_wild":
        dataset = LLavaWild()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset


def is_binary_question(question):
    normalized = question.strip().lower()
    binary_prefixes = (
        "is ", "are ", "was ", "were ", "do ", "does ", "did ",
        "can ", "could ", "has ", "have ", "will ", "would ",
    )
    return normalized.startswith(binary_prefixes)


if __name__ == "__main__":
    args = argument_parser().parse_args()

    hal_test = HallucinationTest(args)
    
    datasets_to_run = [dataset.strip() for dataset in args.datasets.split(",") if dataset.strip()]
    if "chair" in datasets_to_run:
        chair_seeds = [int(seed.strip()) for seed in args.chair_seeds.split(",") if seed.strip()]
        for seed in chair_seeds:
            hal_test.infer_dataset("chair", seed)

    for dataset in datasets_to_run:
        if dataset != "chair":
            hal_test.infer_dataset(dataset, 0)

    # hal_test.infer_dataset("llava_wild", 0)
