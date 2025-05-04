# src/utils/prompt_templates.py

class PromptTemplates:
    """提示词模板集合"""
    
    @staticmethod
    def task1_image_description(retrieved_contexts=None):
        """
        任务1: 事件增强图像描述的提示模板
        
        Args:
            retrieved_contexts: 可选，检索到的上下文信息
            
        Returns:
            格式化的提示词
        """
        context_text = ""
        if retrieved_contexts:
            context_parts = ["以下是相关的新闻内容:"]
            for i, ctx in enumerate(retrieved_contexts):
                if isinstance(ctx, dict) and "metadata" in ctx:
                    meta = ctx["metadata"]
                    if "title" in meta and "content" in meta:
                        context_parts.append(f"[{i+1}] 标题: {meta['title']}\n内容: {meta['content']}")
                    elif "content" in meta:
                        context_parts.append(f"[{i+1}] {meta['content']}")
                elif isinstance(ctx, str):
                    context_parts.append(f"[{i+1}] {ctx}")
            context_text = "\n\n".join(context_parts) + "\n\n"
        
        prompt = (
            f"{context_text}"
            "请分析图片，并结合提供的新闻内容，生成一个详细的描述。"
            "在描述中应包含:\n"
            "1. 图片中可见的人物、场景和活动\n"
            "2. 与图片相关的事件背景信息\n"
            "3. 事件的时间、地点和重要细节\n\n"
            "请首先分析图片中的主要元素，然后参考新闻内容丰富事件信息，最后生成一个完整、准确的事件描述。"
        )
        
        return prompt
    
    @staticmethod
    def task1_cot_image_description(retrieved_contexts=None):
        """
        任务1: 使用Chain-of-Thought策略的图像描述提示
        
        Args:
            retrieved_contexts: 可选，检索到的上下文信息
            
        Returns:
            格式化的提示词
        """
        context_text = ""
        if retrieved_contexts:
            context_parts = ["以下是相关的新闻内容:"]
            for i, ctx in enumerate(retrieved_contexts):
                if isinstance(ctx, dict) and "metadata" in ctx:
                    meta = ctx["metadata"]
                    if "title" in meta and "content" in meta:
                        context_parts.append(f"[{i+1}] 标题: {meta['title']}\n内容: {meta['content']}")
                    elif "content" in meta:
                        context_parts.append(f"[{i+1}] {meta['content']}")
                elif isinstance(ctx, str):
                    context_parts.append(f"[{i+1}] {ctx}")
            context_text = "\n\n".join(context_parts) + "\n\n"
        
        prompt = (
            f"{context_text}"
            "请分析图片并生成一个详细的事件描述。请按照以下步骤思考:\n\n"
            "步骤1: 仔细分析图片中的视觉元素，包括人物、场景、活动和任何显著特征。\n"
            "步骤2: 根据这些视觉元素确定图片可能展示的事件类型。\n"
            "步骤3: 参考提供的新闻内容，找出与图像相关的具体事件信息。\n"
            "步骤4: 将视觉分析与事件背景信息整合，形成完整的事件描述。\n\n"
            "现在，请按步骤分析并生成一个详细、准确的事件描述。"
        )
        
        return prompt
    
    @staticmethod
    def task2_image_retrieval():
        """
        任务2: 基于事件的图像检索提示
        
        Returns:
            格式化的提示词
        """
        prompt = (
            "请分析这个事件描述，提取关键元素以帮助我检索相关图像:\n\n"
            "1. 首先，确定事件的主要人物、地点和活动\n"
            "2. 识别可能在图像中出现的视觉元素和特征\n"
            "3. 提取能区分此事件的独特特征\n\n"
            "提取完毕后，请生成可用于图像检索的核心关键词和短语。"
        )
        
        return prompt
    
    @staticmethod
    def task2_cot_image_retrieval():
        """
        任务2: 使用Chain-of-Thought策略的图像检索提示
        
        Returns:
            格式化的提示词
        """
        prompt = (
            "请分析以下事件描述，并帮助我找到最相关的新闻图片。请按照这些步骤思考:\n\n"
            "步骤1: 仔细分析事件描述，确定事件的主要人物、地点、时间和活动。\n"
            "步骤2: 想象这一事件在图像中可能会如何呈现，考虑可能的视觉元素。\n"
            "步骤3: 提取能唯一标识此事件的关键词和特征。\n"
            "步骤4: 根据这些关键要素，确定最适合用于检索相关图像的查询词。\n\n"
            "现在，请按步骤思考并提取适用于图像检索的关键信息。"
        )
        
        return prompt