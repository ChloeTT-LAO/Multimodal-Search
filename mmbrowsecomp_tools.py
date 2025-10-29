"""
MMBrowseComp Tools Configuration
为MMBrowseComp评估定制的工具配置
"""

import os
import requests
from typing import Dict, Any, List
from PIL import Image
import io
import base64

class MMBrowseCompTools:
    """MMBrowseComp专用工具集"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bright_data_api_key = os.getenv('BRIGHT_DATA_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

    # 在 MMBrowseCompTools 类中修改 web_search 方法
    def web_search(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """使用SerpAPI进行搜索"""
        import serpapi

        client = serpapi.Client(api_key=os.getenv('SERPAPI_KEY'))

        results = client.search({
            "engine": "google",
            "q": query,
            "num": num_results
        })

        formatted_results = []
        for item in results.get('organic_results', [])[:num_results]:
            formatted_results.append({
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', '')
            })

        return formatted_results
    
    def reverse_image_search(self, image_url: str) -> List[Dict[str, str]]:
        """
        使用Google Lens进行反向图像搜索
        
        Args:
            image_url: 图像URL
            
        Returns:
            搜索结果列表
        """
        if not self.bright_data_api_key:
            raise ValueError("BRIGHT_DATA_API_KEY not found in environment")
        
        url = "https://api.brightdata.com/serp/google/lens"
        headers = {
            "Authorization": f"Bearer {self.bright_data_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "url": image_url
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            results = response.json()
            formatted_results = []
            
            for item in results.get('visual_matches', [])[:10]:
                formatted_results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'source': item.get('source', ''),
                    'thumbnail': item.get('thumbnail', '')
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Reverse image search error: {e}")
            return []
    
    def web_browse(self, url: str) -> str:
        """
        访问网页并提取文本内容
        
        Args:
            url: 网页URL
            
        Returns:
            网页文本内容
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 简单的HTML到文本转换
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除script和style标签
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 获取文本
            text = soup.get_text()
            
            # 清理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # 限制长度
            max_length = 8000
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            return text
            
        except Exception as e:
            return f"Error browsing {url}: {str(e)}"
    
    def analyze_image_vlm(self, image_url: str, question: str) -> str:
        """
        使用VLM直接分析图像（原生多模态方法）
        
        Args:
            image_url: 图像URL
            question: 关于图像的问题
            
        Returns:
            分析结果
        """
        # 这里应该调用VLM模型
        # 为了演示，返回一个占位符
        return f"[VLM Analysis needed for {image_url} with question: {question}]"
    
    def analyze_image_caption(self, image_url: str) -> str:
        """
        使用captioning工具分析图像（传统方法）
        
        Args:
            image_url: 图像URL
            
        Returns:
            图像描述
        """
        # 这是传统的captioning方法，会导致信息丢失
        return f"[Caption needed for {image_url}]"
    
    def download_image(self, image_url: str) -> str:
        """
        下载图像到本地
        
        Args:
            image_url: 图像URL
            
        Returns:
            本地路径
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # 保存到临时目录
            import tempfile
            import hashlib
            
            # 生成唯一文件名
            url_hash = hashlib.md5(image_url.encode()).hexdigest()
            ext = image_url.split('.')[-1] if '.' in image_url else 'jpg'
            
            temp_dir = tempfile.gettempdir()
            local_path = os.path.join(temp_dir, f"img_{url_hash}.{ext}")
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            return local_path
            
        except Exception as e:
            return f"Error downloading image: {str(e)}"
    
    def extract_pdf_text(self, pdf_url: str) -> str:
        """
        从PDF提取文本
        
        Args:
            pdf_url: PDF URL
            
        Returns:
            PDF文本内容
        """
        try:
            import pdfplumber
            import tempfile
            
            # 下载PDF
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            # 提取文本
            text = ""
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages[:10]:  # 限制前10页
                    text += page.extract_text() or ""
            
            # 清理临时文件
            os.unlink(tmp_path)
            
            # 限制长度
            max_length = 10000
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            return text
            
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"


class ToolRegistry:
    """工具注册表"""
    
    @staticmethod
    def get_tool_definitions() -> List[Dict[str, Any]]:
        """
        返回Agent-R1格式的工具定义
        """
        return [
            {
                "name": "web_search",
                "description": "Search the web for information using Google Search API. Returns a list of relevant web pages with titles, URLs, and snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "reverse_image_search",
                "description": "Perform reverse image search using Google Lens. Useful for identifying objects, landmarks, or finding visually similar images.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "URL of the image to search"
                        }
                    },
                    "required": ["image_url"]
                }
            },
            {
                "name": "web_browse",
                "description": "Browse a specific webpage and extract its text content. Use this after getting URLs from web_search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to browse"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "analyze_image_vlm",
                "description": "Analyze an image using native multimodal VLM capabilities. This preserves full visual information. Use this for detailed image understanding tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "URL of the image to analyze"
                        },
                        "question": {
                            "type": "string",
                            "description": "Question about the image"
                        }
                    },
                    "required": ["image_url", "question"]
                }
            },
            {
                "name": "download_image",
                "description": "Download an image to local file system for direct VLM processing. This is the preferred method for detailed visual analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "URL of the image to download"
                        }
                    },
                    "required": ["image_url"]
                }
            },
            {
                "name": "extract_pdf_text",
                "description": "Extract text content from PDF files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_url": {
                            "type": "string",
                            "description": "URL of the PDF file"
                        }
                    },
                    "required": ["pdf_url"]
                }
            }
        ]


if __name__ == "__main__":
    # 测试工具
    print("Testing MMBrowseComp Tools...")
    
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    config = {}
    tools = MMBrowseCompTools(config)
    
    # 测试web_search
    print("\n1. Testing web_search...")
    results = tools.web_search("Python programming", num_results=3)
    print(f"Found {len(results)} results")
    
    print("\nTool definitions:")
    for tool_def in ToolRegistry.get_tool_definitions():
        print(f"  - {tool_def['name']}: {tool_def['description'][:50]}...")
