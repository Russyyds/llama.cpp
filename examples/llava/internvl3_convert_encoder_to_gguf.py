import argparse
import logging
import sys
import torch
import json
import os
import numpy as np
from typing import cast, ContextManager, Any, Iterator, List
from pathlib import Path
from torch import Tensor
sys.path.append(str(Path(__file__).parent / "../../gguf-py"))
import gguf
from gemma3_convert_encoder_to_gguf import LazyTorchTensor
logger = logging.getLogger("internvl3-mmproj")

class InternVL3VisionTower:
    hparams: dict
    gguf_writer: gguf.GGUFWriter
    fname_out: Path
    ftype: gguf.LlamaFileType

    @staticmethod
    def load_hparams(dir_model: Path):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    
    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)
        part_names.sort()
        return part_names
    
    def __init__(
        self,
        dir_model: Path,
        fname_out: Path,
        ftype: gguf.LlamaFileType,
        is_big_endian: bool
    ):
        hparams = InternVL3VisionTower.load_hparams(dir_model=dir_model)
        self.hparams = hparams
        self.fname_out = fname_out
        self.ftype = ftype
        endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.gguf_writer = gguf.GGUFWriter(path=None, arch="clip", endianess=endianess)

        llm_config = hparams["llm_config"]
        vision_config = hparams["vision_config"]
        assert hparams["architectures"][0] == "InternVLChatModel"
        assert llm_config is not None
        assert vision_config is not None

        self.gguf_writer.add_string("clip.projector_type", "qwen2vl_merger")
        self.gguf_writer.add_bool("clip_has_text_encoder", False)
        self.gguf_writer.add_bool("clip.has_vision_encoder", True)
        self.gguf_writer.add_bool("clip.has_llava_projector", False)
        self.gguf_writer.add_uint32("clip.vision.image_size", vision_config["image_size"])
        self.gguf_writer.add_uint32("clip.vision.patch_size", vision_config["patch_size"])
        self.gguf_writer.add_uint32("clip.vision.embedding_length", vision_config["hidden_size"])
        self.gguf_writer.add_uint32("clip.vision.feed_forward_length", vision_config["intermediate_size"])
        self.gguf_writer.add_uint32("clip.vision.projection_dim", llm_config["hidden_size"])
        self.gguf_writer.add_uint32("clip.vision.block_count", vision_config["num_hidden_layers"])
        self.gguf_writer.add_uint32("clip.vision.attention.head_count", vision_config["num_attention_heads"])
        self.gguf_writer.add_float32("clip.vision.attention.layer_norm_epsilon", vision_config.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_array  ("clip.vision.image_mean", [0.5, 0.5, 0.5])
        self.gguf_writer.add_array  ("clip.vision.image_std",  [0.5, 0.5, 0.5])
        self.gguf_writer.add_bool   ("clip.use_gelu", True)
        self.gguf_writer.add_float32("clip.vision.downsample_ratio", hparams["downsample_ratio"])

        for name, data_torch in self.get_tensors(dir_model):
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)
            self.add_tensor(name, data_torch)
        
    def get_tensors(self, dir_model: Path) -> Iterator[tuple[str, Tensor]]:
        part_names = InternVL3VisionTower.get_model_part_names(dir_model, "model", ".safetensors")
        tensor_names_from_parts: set[str] = set()
        for part_name in part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            from safetensors import safe_open
            ctx = cast(ContextManager[Any], safe_open(dir_model / part_name, framework="pt", device="cpu"))
            with ctx as model_part:
                tensor_names_from_parts.update(model_part.keys())

                for name in model_part.keys():
                    data = model_part.get_slice(name)
                    data = LazyTorchTensor.from_safetensors_slice(data)
                    yield name, data
    
    def add_tensor(self, name: str, data_torch: Tensor):
        is_1d = len(data_torch) == 1
        is_embed = ".embeddings." in name
        old_dtype = data_torch.dtype
        can_quantize = not is_1d and not is_embed
        data_qtype = gguf.GGMLQuantizationType.F32
        if not name.startswith("vision_model.") and not name.startswith("mlp1."):
            return
        
        name = name.replace("vision_model.encoder.layers.", "v.blk.")
        name = name.replace("vision_model.", "v.")
        name = name.replace(".embeddings.patch_embedding.", ".patch_embed.")
        name = name.replace(".embeddings.position_embedding", ".position_embed.weight")
        name = name.replace(".class_embedding", "class_embed.weight")
        name = name.replace("mlp1.", "mm.")
        name = name.replace(".mlp.fc1.", ".ffn_down.")
        name = name.replace(".mlp.fc2.", ".ffn_up.")
        name = name.replace(".norm1.", ".ln1.")
        name = name.replace(".norm2.", ".ln2.")
        name = name.replace(".proj.", ".out.")
        name = name.replace(".attn.", ".attn_")

        if can_quantize:
            if self.ftype == gguf.LlamaFileType.ALL_F32:
                data_qtype = gguf.GGMLQuantizationType.F32
            elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                data_qtype = gguf.GGMLQuantizationType.F16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                data_qtype = gguf.GGMLQuantizationType.BF16
            elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                data_qtype = gguf.GGMLQuantizationType.Q8_0
            else:
                raise ValueError(f"Unsupported file type: {self.ftype}")

        if "soft_emb_norm.weight" in name:
            logger.info(f"Correcting norm value for '{name}'")
            data_torch = data_torch + 1
        # handle attn.qkv weight and bias
        if "qkv" in name:
            shape = data_torch.shape
            
            assert shape[0] % 3 == 0

            n = shape[0] // 3
            wq = data_torch[:n,]
            wk = data_torch[n:2 * n,]
            wv = data_torch[2*n:,]
            q_name = name.replace("qkv", "q")
            k_name = name.replace("qkv", "k")
            v_name = name.replace("qkv", "v")
            
            wq = wq.numpy()
            wk = wk.numpy()
            wv = wv.numpy()
            try:
                wq = gguf.quants.quantize(wq, data_qtype)
                wk = gguf.quants.quantize(wk, data_qtype)
                wv = gguf.quants.quantize(wv, data_qtype)
            except Exception as e:
                logger.error(f"Error quantizing tensor qkv, fallback to F16")
                data_qtype = gguf.GGMLQuantizationType.F16
                wq = gguf.quants.quantize(wq, data_qtype)
                wk = gguf.quants.quantize(wk, data_qtype)
                wv = gguf.quants.quantize(wv, data_qtype)
            shape_str = f"{{{', '.join(str(n) for n in reversed(wq.shape))}}}"
            logger.info(f"{f'%-32s' % f'{q_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")
            shape_str = f"{{{', '.join(str(n) for n in reversed(wk.shape))}}}"
            logger.info(f"{f'%-32s' % f'{k_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")
            shape_str = f"{{{', '.join(str(n) for n in reversed(wv.shape))}}}"
            logger.info(f"{f'%-32s' % f'{v_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")
            self.gguf_writer.add_tensor(q_name, wq, raw_dtype=data_qtype)
            self.gguf_writer.add_tensor(k_name, wk, raw_dtype=data_qtype)
            self.gguf_writer.add_tensor(v_name, wv, raw_dtype=data_qtype)

        else:
            data = data_torch.numpy()
            try:
                data = gguf.quants.quantize(data, data_qtype)
            except Exception as e:
                logger.error(f"Error quantizing tensor '{name}': {e}, fallback to F16")
                data_qtype = gguf.GGMLQuantizationType.F16
                data = gguf.quants.quantize(data, data_qtype)
            # reverse shape to make it similar to the internal ggml dimension order
            shape_str = f"{{{', '.join(str(n) for n in reversed(data_torch.shape))}}}"
            logger.info(f"{f'%-32s' % f'{name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

            self.gguf_writer.add_tensor(name, data, raw_dtype=data_qtype)    
           
    def write(self):
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser( description="Convert InternVL3 vision tower safetensors to GGUF format")
    parser.add_argument(
        "--outfile", type=Path, default="mmproj.gguf",
        help="path to save the converted gguf file"
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0"], default="f16",
        help="output format",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=Path,
        help="directory containing model file",
        nargs="?",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )

    args = parser.parse_args()
    if args.model is None:
        parser.error("the following arguments are required: model")
    return args

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dir_model = args.model
    if args.outfile.name == "mmproj.gguf":
        args.outfile = dir_model / args.outfile
    logger.info(f"Saving converted model to: {args.outfile}")

    if not dir_model.is_dir():
        logger.error(f'Error: {args.model} is not a directory')
        sys.exit(1)
    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }

    logger.info(f"Loading model: {dir_model.name}")
    with torch.inference_mode():
        internvl3_vision_tower = InternVL3VisionTower(
            dir_model=dir_model,
            fname_out=args.outfile,
            ftype=ftype_map[args.outtype],
            is_big_endian=args.bigendian
        )
        internvl3_vision_tower.write()

if __name__ == "__main__":
    main()