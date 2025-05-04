# src/generation/generator.py
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class QwenGenerator:
    def __init__(self, config):
        """
        Qwen2.5-VL生成器类
        
        Args:
            config: 配置信息，包含模型路径等参数
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型和处理器
        try:
            logger.info(f"Loading Qwen2.5-VL generator from {config['model']['model_path']}")
            self.processor = AutoProcessor.from_pretrained(config["model"]["model_path"])
            
            model_kwargs = {
                "device_map": config["model"]["device_map"],
                "torch_dtype": self._get_precision(config["model"]["precision"])
            }
            
            # 如果配置中启用了FlashAttention2
            if config["model"].get("use_flash_attention", False):
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model"]["model_path"],
                **model_kwargs
            )
            logger.info("Qwen2.5-VL generator loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL generator: {e}")
            raise
    
    def _get_precision(self, precision):
        """获取精度类型"""
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        return torch.float32
    
    def generate(self, image=None, prompt=None, contexts=None, **kwargs):
        """
        生成文本响应
        
        Args:
            image: 可选，输入图像 (PIL图像或图像路径)
            prompt: 可选，输入提示文本
            contexts: 可选，外部上下文信息列表
            **kwargs: 生成参数
            
        Returns:
            生成的文本响应
        """
        # 处理图像输入
        if image is not None and isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to open image {image}: {e}")
                raise
        
        # 构建输入消息
        messages = []
        
        # 构建用户消息的内容列表
        content = []
        
        # 添加图像（如果有）
        if image is not None:
            content.append({"type": "image", "image": image})
        
        # 构建文本部分，包括上下文（如果有）
        text_parts = []
        
        # 添加检索到的上下文
        if contexts:
            text_parts.append("参考信息:")
            for i, ctx in enumerate(contexts):
                if isinstance(ctx, dict) and "metadata" in ctx:
                    meta = ctx["metadata"]
                    if "title" in meta and "content" in meta:
                        text_parts.append(f"[{i+1}] {meta['title']}: {meta['content']}")
                    elif "content" in meta:
                        text_parts.append(f"[{i+1}] {meta['content']}")
                elif isinstance(ctx, str):
                    text_parts.append(f"[{i+1}] {ctx}")
            text_parts.append("\n")
        
        # 添加主提示
        if prompt:
            text_parts.append(prompt)
        
        # 将文本部分合并为一个字符串
        text_content = "\n".join(text_parts)
        
        # 添加文本内容（如果有）
        if text_content:
            content.append({"type": "text", "text": text_content})
        
        # 添加用户消息
        if content:
            messages.append({"role": "user", "content": content})
        
        # 如果没有消息则返回错误
        if not messages:
            raise ValueError("No valid input provided (neither image nor text)")
        
        # 应用聊天模板
        model_inputs = self.processor.apply_chat_template(
            messages, 
            return_tensors="pt"
        ).to(self.device)
        
        # 解析生成参数
        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config["model"]["max_new_tokens"]),
            "temperature": kwargs.get("temperature", self.config["model"]["temperature"]),
            "top_p": kwargs.get("top_p", self.config["model"]["top_p"]),
            "do_sample": kwargs.get("do_sample", True)
        }
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                model_inputs,
                **generation_config
            )
        
        # 解码输出
        response = self.processor.decode(outputs[0][model_inputs.shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    def generate_with_cot(self, image=None, prompt=None, contexts=None, **kwargs):
        """
        使用Chain-of-Thought提示进行文本生成
        
        Args:
            image: 可选，输入图像
            prompt: 可选，输入提示文本
            contexts: 可选，外部上下文信息列表
            **kwargs: 生成参数
            
        Returns:
            生成的文本响应
        """
        # 构建CoT提示
        cot_prompt = prompt
        if prompt:
            # 添加思考步骤提示
            cot_prompt = (
                f"{prompt}\n\n"
                "请按照以下步骤思考:\n"
                "步骤1: 首先分析图像中的关键视觉元素和场景。\n"
                "步骤2: 参考提供的相关新闻内容，找出与图像相关的事件信息。\n"
                "步骤3: 将视觉分析与事件信息整合，形成完整的事件描述。\n\n"
                "现在，请先分析后回答:"
            )
        
        # 调用基本生成方法
        return self.generate(image=image, prompt=cot_prompt, contexts=contexts, **kwargs)