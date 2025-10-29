"""
MMBrowseComp Evaluation Script for Agent-R1
基于Agent-R1框架的MMBrowseComp评估脚本
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import argparse

# 添加Agent-R1到路径
WORKSPACE_DIR = os.getenv('WORKSPACE_DIR', os.path.expanduser('~/agent_r1_workspace'))
AGENT_R1_DIR = os.path.join(WORKSPACE_DIR, 'Agent-R1')
sys.path.insert(0, AGENT_R1_DIR)

from dotenv import load_dotenv
load_dotenv()


class MMBrowseCompEvaluator:
    """MMBrowseComp评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = config.get('data_path')
        self.output_dir = config.get('output_dir', './outputs')
        self.max_tool_calls = config.get('max_tool_calls', 20)
        self.timeout = config.get('timeout', 300)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载数据
        self.data = self.load_data()
        
        # 初始化工具
        from mmbrowsecomp_tools import MMBrowseCompTools
        self.tools = MMBrowseCompTools(config)
        
        print(f"Loaded {len(self.data)} questions from {self.data_path}")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载MMBrowseComp数据"""
        data = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        # 如果指定了子集大小，随机采样
        subset_size = self.config.get('subset_size')
        if subset_size and subset_size < len(data):
            import random
            random.seed(42)
            data = random.sample(data, subset_size)
        
        return data
    
    def build_prompt(self, question_data: Dict[str, Any]) -> str:
        """构建Agent-R1的输入prompt"""
        images = question_data.get('images', [])
        question = question_data['question']
        
        prompt = """You are an AI agent capable of browsing the web and analyzing multimodal content to answer complex questions.

Available Tools:
- web_search(query): Search the web for information
- reverse_image_search(image_url): Find similar images or identify objects
- web_browse(url): Visit a webpage and extract its content
- analyze_image_vlm(image_url, question): Analyze an image with native VLM
- download_image(image_url): Download image for direct analysis
- extract_pdf_text(pdf_url): Extract text from PDF files

Instructions:
1. First, provide a problem-solving roadmap explaining your approach
2. Use multiple tool calls to gather information systematically
3. Pay special attention to visual content - images and videos often contain critical information
4. For visual questions, prefer downloading images and analyzing them directly over using captions
5. Provide your final answer clearly

"""
        
        if images:
            prompt += f"\nImages in the question:\n"
            for idx, img_url in enumerate(images):
                prompt += f"  Image {idx+1}: {img_url}\n"
            prompt += "\n"
        
        prompt += f"Question: {question}\n\n"
        prompt += "Please answer the question and provide your problem-solving roadmap.\n"
        
        return prompt
    
    def run_agent_react(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行Agent-R1 ReAct流程（简化版本）"""
        prompt = self.build_prompt(question_data)
        
        result = {
            'question_id': question_data['id'],
            'question': question_data['question'],
            'ground_truth': question_data['answer'],
            'checklist': question_data['checklist'],
            'predicted_answer': '',
            'roadmap': '',
            'tool_calls': [],
            'success': False,
            'error': None
        }
        
        try:
            # 这里应该调用Agent-R1的推理循环
            # 当前是占位符实现
            
            # 示例：简单的工具调用流程
            if question_data.get('images'):
                for img_url in question_data['images']:
                    search_results = self.tools.reverse_image_search(img_url)
                    result['tool_calls'].append({
                        'tool': 'reverse_image_search',
                        'input': img_url,
                        'output': search_results
                    })
            
            search_results = self.tools.web_search(question_data['question'])
            result['tool_calls'].append({
                'tool': 'web_search',
                'input': question_data['question'],
                'output': search_results
            })
            
            if search_results:
                top_url = search_results[0]['url']
                content = self.tools.web_browse(top_url)
                result['tool_calls'].append({
                    'tool': 'web_browse',
                    'input': top_url,
                    'output': content[:500]
                })
            
            result['predicted_answer'] = "TODO: Implement full Agent-R1 reasoning"
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    def evaluate_answer(self, predicted: str, ground_truth: str) -> bool:
        """评估答案是否正确"""
        predicted = predicted.strip().lower()
        ground_truth = ground_truth.strip().lower()
        
        if predicted == ground_truth:
            return True
        
        if ground_truth in predicted or predicted in ground_truth:
            return True
        
        return False
    
    def evaluate_checklist(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """评估checklist完成情况"""
        checklist = result['checklist']
        checklist_scores = [0] * len(checklist)
        
        # TODO: 实现基于LLM的checklist评估
        
        return {
            'checklist_scores': checklist_scores,
            'completed_items': sum(checklist_scores),
            'total_items': len(checklist),
            'checklist_score': sum(checklist_scores) / len(checklist) if checklist else 0
        }
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算评估指标"""
        total = len(results)
        
        oa = sum(1 for r in results if r.get('correct', False)) / total if total > 0 else 0
        
        sa = sum(1 for r in results 
                if r.get('correct', False) and r.get('checklist_eval', {}).get('checklist_score', 0) == 1.0
                ) / total if total > 0 else 0
        
        avg_cs = sum(r.get('checklist_eval', {}).get('checklist_score', 0) for r in results) / total if total > 0 else 0
        
        return {
            'overall_accuracy': oa * 100,
            'strict_accuracy': sa * 100,
            'avg_checklist_score': avg_cs * 100,
            'total_questions': total
        }
    
    def run_evaluation(self):
        """运行完整评估"""
        print("\n" + "="*60)
        print("Starting MMBrowseComp Evaluation")
        print("="*60)
        print(f"Total questions: {len(self.data)}")
        print(f"Max tool calls: {self.max_tool_calls}")
        print(f"Timeout: {self.timeout}s")
        print("="*60 + "\n")
        
        all_results = []
        
        for idx, question_data in enumerate(tqdm(self.data, desc="Evaluating")):
            print(f"\n[Question {idx+1}/{len(self.data)}] ID: {question_data['id']}")
            print(f"Category: {question_data.get('category', 'N/A')}")
            print(f"Subtask: {question_data.get('subtask', 'N/A')}")
            
            result = self.run_agent_react(question_data)
            
            result['correct'] = self.evaluate_answer(
                result['predicted_answer'],
                result['ground_truth']
            )
            
            result['checklist_eval'] = self.evaluate_checklist(result)
            
            all_results.append(result)
            
            if (idx + 1) % 10 == 0:
                self.save_results(all_results, suffix='_partial')
        
        metrics = self.compute_metrics(all_results)
        
        self.save_results(all_results, metrics)
        
        self.print_results(metrics)
        
        return all_results, metrics
    
    def save_results(self, results: List[Dict[str, Any]], 
                    metrics: Optional[Dict[str, float]] = None,
                    suffix: str = ''):
        """保存评估结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_file = os.path.join(
            self.output_dir,
            f'results{suffix}_{timestamp}.json'
        )
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_file}")
        
        if metrics:
            metrics_file = os.path.join(
                self.output_dir,
                f'metrics_{timestamp}.json'
            )
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Metrics saved to: {metrics_file}")
    
    def print_results(self, metrics: Dict[str, float]):
        """打印评估结果"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy (OA):     {metrics['overall_accuracy']:.2f}%")
        print(f"Strict Accuracy (SA):      {metrics['strict_accuracy']:.2f}%")
        print(f"Avg Checklist Score (CS):  {metrics['avg_checklist_score']:.2f}%")
        print(f"Total Questions:           {metrics['total_questions']}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='MMBrowseComp Evaluation with Agent-R1')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to MMBrowseComp data file (JSONL)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Evaluate on a subset (for testing)')
    parser.add_argument('--max_tool_calls', type=int, default=20,
                       help='Maximum number of tool calls per question')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per question (seconds)')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash-preview-05-20',
                       help='Backbone model to use')
    
    args = parser.parse_args()
    
    config = {
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'subset_size': args.subset_size,
        'max_tool_calls': args.max_tool_calls,
        'timeout': args.timeout,
        'model': args.model
    }
    
    evaluator = MMBrowseCompEvaluator(config)
    results, metrics = evaluator.run_evaluation()
    
    return results, metrics


if __name__ == "__main__":
    main()
