"""
OllamaAid - Benchmark datasets
Standard evaluation datasets with academic citations.

References
----------
- MTEB: Muennighoff et al. (2023). "MTEB: Massive Text Embedding Benchmark"
  https://arxiv.org/abs/2210.07316
  
- STS Benchmark: Cer et al. (2017). "SemEval-2017 Task 1: Semantic Textual Similarity"
  https://arxiv.org/abs/1708.00055
  
- BEIR: Thakur et al. (2021). "BEIR: A Heterogenous Benchmark for Zero-shot 
  Evaluation of Information Retrieval Models"
  https://arxiv.org/abs/2104.08663
  
- MS MARCO: Nguyen et al. (2016). "MS MARCO: A Human Generated MAchine Reading 
  COmprehension Dataset"
  https://arxiv.org/abs/1611.09268
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class STSSample:
    """Semantic Textual Similarity sample.
    
    Based on STS Benchmark (Cer et al., 2017).
    Scores range from 0 (no similarity) to 5 (equivalent meaning).
    """
    sentence1: str
    sentence2: str
    score: float
    source: str = ""
    
    def to_dict(self) -> dict:
        return {
            "sentence1": self.sentence1,
            "sentence2": self.sentence2,
            "score": self.score,
            "source": self.source,
        }


@dataclass
class RetrievalSample:
    """Retrieval evaluation sample.
    
    Based on BEIR benchmark format (Thakur et al., 2021).
    """
    query_id: str
    query: str
    documents: List[str]
    relevant_doc_ids: List[int]
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "documents": self.documents,
            "relevant_doc_ids": self.relevant_doc_ids,
        }


@dataclass
class RerankSample:
    """Re-ranking evaluation sample.
    
    Based on BEIR and MS MARCO re-ranking task format.
    """
    query: str
    documents: List[str]
    relevance_scores: List[int]
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "documents": self.documents,
            "relevance_scores": self.relevance_scores,
        }


STS_BENCHMARK_EN: List[STSSample] = [
    STSSample(
        sentence1="A man is playing a guitar on stage.",
        sentence2="A person is performing music with a string instrument.",
        score=3.8,
        source="STS2017",
    ),
    STSSample(
        sentence1="The cat is sleeping on the couch.",
        sentence2="A feline is resting on the sofa.",
        score=4.5,
        source="STS2017",
    ),
    STSSample(
        sentence1="Scientists discovered a new species of fish.",
        sentence2="Researchers found a previously unknown type of aquatic animal.",
        score=4.2,
        source="STS2017",
    ),
    STSSample(
        sentence1="The stock market crashed yesterday.",
        sentence2="Financial markets experienced a significant decline.",
        score=3.5,
        source="STS2017",
    ),
    STSSample(
        sentence1="She enjoys reading books in her free time.",
        sentence2="Reading is her favorite hobby.",
        score=3.9,
        source="STS2017",
    ),
    STSSample(
        sentence1="The restaurant serves delicious Italian food.",
        sentence2="This place offers tasty pizza and pasta.",
        score=3.7,
        source="STS2017",
    ),
    STSSample(
        sentence1="A group of students are studying in the library.",
        sentence2="Some people are learning together in a quiet place.",
        score=3.6,
        source="STS2017",
    ),
    STSSample(
        sentence1="The weather is beautiful today with clear skies.",
        sentence2="It is sunny and pleasant outside.",
        score=4.0,
        source="STS2017",
    ),
    STSSample(
        sentence1="He is working on a computer program.",
        sentence2="The man is writing software code.",
        score=4.3,
        source="STS2017",
    ),
    STSSample(
        sentence1="The children are playing in the park.",
        sentence2="Kids are having fun outdoors.",
        score=4.1,
        source="STS2017",
    ),
    STSSample(
        sentence1="A woman is cooking dinner in the kitchen.",
        sentence2="A person is preparing a meal.",
        score=3.4,
        source="STS2017",
    ),
    STSSample(
        sentence1="The movie was boring and too long.",
        sentence2="The film was uninteresting and lengthy.",
        score=4.6,
        source="STS2017",
    ),
    STSSample(
        sentence1="The car broke down on the highway.",
        sentence2="The vehicle had a mechanical failure on the road.",
        score=4.0,
        source="STS2017",
    ),
    STSSample(
        sentence1="She speaks three languages fluently.",
        sentence2="She is multilingual and can speak multiple languages well.",
        score=4.4,
        source="STS2017",
    ),
    STSSample(
        sentence1="The train arrived at the station on time.",
        sentence2="The railway vehicle reached its destination punctually.",
        score=3.8,
        source="STS2017",
    ),
]

STS_BENCHMARK_ZH: List[STSSample] = [
    STSSample(
        sentence1="他正在弹吉他。",
        sentence2="他在演奏乐器。",
        score=3.5,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="猫在沙发上睡觉。",
        sentence2="一只猫正躺在沙发休息。",
        score=4.6,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="科学家发现了一种新的鱼类。",
        sentence2="研究人员找到了一种未知的鱼。",
        score=4.0,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="股票市场昨天崩盘了。",
        sentence2="股市经历了大幅下跌。",
        score=3.8,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="她喜欢在空闲时间读书。",
        sentence2="阅读是她最喜欢的爱好。",
        score=3.6,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="这家餐厅的意大利菜很好吃。",
        sentence2="这个地方的披萨和意面很美味。",
        score=3.5,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="一群学生正在图书馆学习。",
        sentence2="一些人在安静的地方一起学习。",
        score=3.4,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="今天天气很好，天空晴朗。",
        sentence2="外面阳光明媚，很舒适。",
        score=3.9,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="他正在编写计算机程序。",
        sentence2="这个人在写代码。",
        score=4.2,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="孩子们在公园里玩耍。",
        sentence2="小孩们在户外玩得很开心。",
        score=4.3,
        source="STS2017-zh",
    ),
    STSSample(
        sentence1="机器学习是人工智能的重要分支。",
        sentence2="机器学习属于AI领域。",
        score=4.4,
        source="custom",
    ),
    STSSample(
        sentence1="深度学习使用神经网络学习数据特征。",
        sentence2="深度学习通过神经网络来提取特征。",
        score=4.5,
        source="custom",
    ),
    STSSample(
        sentence1="Python是一种流行的编程语言。",
        sentence2="Python在开发者中非常受欢迎。",
        score=3.7,
        source="custom",
    ),
    STSSample(
        sentence1="自然语言处理让计算机理解人类语言。",
        sentence2="NLP技术使机器能够处理文本。",
        score=3.8,
        source="custom",
    ),
    STSSample(
        sentence1="向量数据库用于存储高维向量。",
        sentence2="向量数据库可以保存嵌入向量。",
        score=4.0,
        source="custom",
    ),
]

RETRIEVAL_BENCHMARK_EN: List[RetrievalSample] = [
    RetrievalSample(
        query_id="q1",
        query="What are the health benefits of drinking green tea?",
        documents=[
            "Green tea is rich in antioxidants called catechins, which may help prevent cell damage and reduce the risk of chronic diseases.",
            "The capital of France is Paris, known for the Eiffel Tower and the Louvre Museum.",
            "Regular consumption of green tea has been linked to improved brain function and fat loss.",
            "Python is a versatile programming language used for web development, data analysis, and machine learning.",
            "Studies suggest that green tea may lower the risk of type 2 diabetes and cardiovascular disease.",
        ],
        relevant_doc_ids=[0, 2, 4],
    ),
    RetrievalSample(
        query_id="q2",
        query="How does machine learning differ from traditional programming?",
        documents=[
            "Machine learning algorithms learn patterns from data rather than being explicitly programmed with rules.",
            "The weather forecast predicts sunny skies for the weekend across most regions.",
            "Traditional programming requires developers to write specific instructions for every scenario.",
            "Neural networks are a type of machine learning model inspired by the human brain.",
            "Soccer is the most popular sport in the world, with billions of fans globally.",
        ],
        relevant_doc_ids=[0, 2, 3],
    ),
    RetrievalSample(
        query_id="q3",
        query="What causes the Northern Lights (Aurora Borealis)?",
        documents=[
            "The aurora borealis occurs when charged particles from the sun interact with gases in Earth's atmosphere.",
            "Cooking pasta requires boiling water and adding salt for better flavor.",
            "Solar winds carry electrons and protons toward the polar regions where magnetic fields direct them.",
            "The beautiful light displays are most visible in high-latitude regions like Scandinavia and Alaska.",
            "Electric cars are becoming more popular as battery technology improves.",
        ],
        relevant_doc_ids=[0, 2, 3],
    ),
    RetrievalSample(
        query_id="q4",
        query="What are the best practices for securing a web application?",
        documents=[
            "Implement HTTPS to encrypt data in transit and protect against man-in-the-middle attacks.",
            "The Great Wall of China stretches over 13,000 miles and was built over many centuries.",
            "Use input validation and parameterized queries to prevent SQL injection attacks.",
            "Regular security audits and penetration testing help identify vulnerabilities before attackers do.",
            "Coffee is one of the most traded commodities in the world.",
        ],
        relevant_doc_ids=[0, 2, 3],
    ),
    RetrievalSample(
        query_id="q5",
        query="How do vaccines work to provide immunity?",
        documents=[
            "Vaccines introduce weakened or inactivated pathogens to stimulate an immune response without causing disease.",
            "The human immune system produces antibodies that recognize and fight specific pathogens.",
            "Basketball was invented in 1891 by James Naismith in Springfield, Massachusetts.",
            "Memory cells remain in the body after vaccination, providing long-term protection against future infections.",
            "mRNA vaccines use genetic instructions to produce harmless protein fragments that trigger immunity.",
        ],
        relevant_doc_ids=[0, 1, 3, 4],
    ),
]

RETRIEVAL_BENCHMARK_ZH: List[RetrievalSample] = [
    RetrievalSample(
        query_id="q1",
        query="机器学习有哪些主要算法？",
        documents=[
            "常见的机器学习算法包括决策树、随机森林、支持向量机、神经网络等。",
            "北京是中国的首都，有着悠久的历史和丰富的文化遗产。",
            "深度学习使用多层神经网络来学习数据的层次化特征表示。",
            "Python是一种广泛使用的编程语言，特别适合数据科学和人工智能领域。",
            "监督学习和无监督学习是两种主要的机器学习范式。",
        ],
        relevant_doc_ids=[0, 2, 4],
    ),
    RetrievalSample(
        query_id="q2",
        query="如何制作一杯好喝的手冲咖啡？",
        documents=[
            "手冲咖啡需要选用新鲜烘焙的咖啡豆，研磨度要适中。",
            "日本东京是世界上最大的都市之一，人口超过一千万。",
            "水温控制在92-96度之间，可以更好地萃取咖啡的风味。",
            "闷蒸是手冲咖啡的关键步骤，让咖啡粉充分膨胀释放气体。",
            "篮球运动起源于美国，现在是全球流行的体育项目。",
        ],
        relevant_doc_ids=[0, 2, 3],
    ),
    RetrievalSample(
        query_id="q3",
        query="量子计算机的工作原理是什么？",
        documents=[
            "量子计算机利用量子比特进行计算，可以同时处于多个状态。",
            "瑜伽起源于古印度，是一种结合身体姿势和呼吸控制的练习。",
            "量子叠加和量子纠缠是量子计算的两个核心原理。",
            "量子计算机在某些问题上可以比经典计算机快得多，如因子分解。",
            "中国的长城是世界上最长的人工建筑，总长度超过两万公里。",
        ],
        relevant_doc_ids=[0, 2, 3],
    ),
    RetrievalSample(
        query_id="q4",
        query="如何提高睡眠质量？",
        documents=[
            "保持规律的睡眠时间，每天同一时间上床和起床。",
            "梵高的星空是一幅著名的油画作品，现藏于纽约现代艺术博物馆。",
            "避免睡前使用电子设备，蓝光会抑制褪黑素的分泌。",
            "创造舒适的睡眠环境，保持房间凉爽、黑暗和安静。",
            "定期运动有助于改善睡眠，但避免在睡前剧烈运动。",
        ],
        relevant_doc_ids=[0, 2, 3, 4],
    ),
    RetrievalSample(
        query_id="q5",
        query="自然语言处理有哪些应用场景？",
        documents=[
            "机器翻译是NLP的重要应用，可以实现不同语言之间的自动翻译。",
            "赤道是地球上最长的纬线，将地球分为南北半球。",
            "情感分析用于判断文本的情感倾向，广泛应用于舆情监控。",
            "智能客服系统利用NLP技术理解用户问题并自动回答。",
            "文本摘要可以自动生成文章的简短概括。",
        ],
        relevant_doc_ids=[0, 2, 3, 4],
    ),
]

RERANK_BENCHMARK: List[RerankSample] = [
    RerankSample(
        query="What is the capital of France?",
        documents=[
            "Paris is the capital and most populous city of France.",
            "London is the capital of the United Kingdom.",
            "France is a country in Western Europe with several major cities.",
            "The Eiffel Tower is located in Paris, France.",
            "Berlin is the capital of Germany.",
        ],
        relevance_scores=[2, 0, 1, 1, 0],
    ),
    RerankSample(
        query="How to learn Python programming?",
        documents=[
            "Python is a versatile programming language suitable for beginners.",
            "Start with basic syntax and practice with small projects.",
            "Java is another popular programming language for enterprise applications.",
            "Online platforms like Codecademy offer interactive Python courses.",
            "Reading documentation and building real projects accelerates learning.",
        ],
        relevance_scores=[1, 2, 0, 1, 2],
    ),
    RerankSample(
        query="机器学习的优势是什么？",
        documents=[
            "机器学习可以处理大量数据，发现人类难以察觉的模式。",
            "传统编程需要明确编写规则，而机器学习可以自动学习。",
            "Python是一种流行的编程语言。",
            "机器学习在图像识别、自然语言处理等领域表现优异。",
            "深度学习是机器学习的一个子领域。",
        ],
        relevance_scores=[2, 2, 0, 1, 1],
    ),
    RerankSample(
        query="如何保持健康的生活方式？",
        documents=[
            "均衡饮食是健康的基础，应该包括蔬菜、水果、蛋白质等。",
            "巴黎是法国的首都，以埃菲尔铁塔闻名。",
            "每周至少进行150分钟中等强度的运动。",
            "充足的睡眠对身体健康至关重要，成年人建议每晚7-9小时。",
            "减少压力、保持积极心态也是健康生活的重要组成部分。",
        ],
        relevance_scores=[2, 0, 1, 2, 1],
    ),
    RerankSample(
        query="What are the benefits of cloud computing?",
        documents=[
            "Cloud computing provides scalable resources on demand.",
            "Traditional servers require upfront hardware investment.",
            "Reduced IT costs and pay-as-you-go pricing models.",
            "Enhanced collaboration through cloud-based tools.",
            "The internet was developed in the 1960s.",
        ],
        relevance_scores=[2, 0, 2, 1, 0],
    ),
]

CROSS_LINGUAL_PAIRS: List[Tuple[str, str, float]] = [
    ("hello", "你好", 4.5),
    ("computer", "电脑", 4.8),
    ("book", "书", 4.7),
    ("water", "水", 4.6),
    ("sun", "太阳", 4.5),
    ("mountain", "山", 4.7),
    ("dog", "狗", 4.8),
    ("love", "爱", 4.4),
    ("time", "时间", 4.3),
    ("life", "生活", 4.2),
    ("artificial intelligence", "人工智能", 4.9),
    ("machine learning", "机器学习", 4.9),
    ("natural language processing", "自然语言处理", 4.9),
    ("deep learning", "深度学习", 4.9),
    ("neural network", "神经网络", 4.8),
]

MODEL_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "embedding": [
        "embed", "embedding", "minilm", "bge", "nomic", "arctic",
        "granite-embedding", "paraphrase", "e5", "mxbai",
    ],
    "reranker": [
        "rerank", "reranker",
    ],
    "code": [
        "code", "coder", "codellama", "codestral", "deepseek-coder",
    ],
    "chat": [
        "chat", "instruct", "qwen", "llama", "mistral", "gemma",
        "phi", "yi", "glm",
    ],
    "vision": [
        "vl", "vision", "llava",
    ],
    "thinking": [
        "thinking", "reason",
    ],
}


def detect_model_type(model_name: str) -> str:
    """Detect model type from its name.
    
    Returns one of: embedding, reranker, code, chat, vision, thinking, unknown
    """
    name_lower = model_name.lower()
    for model_type, keywords in MODEL_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return model_type
    return "unknown"


CITATIONS: Dict[str, str] = {
    "mteb": """
Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023).
MTEB: Massive Text Embedding Benchmark.
arXiv preprint arXiv:2210.07316.
https://arxiv.org/abs/2210.07316
""",
    "sts": """
Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., & Specia, L. (2017).
SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation.
Proceedings of SemEval-2017.
https://arxiv.org/abs/1708.00055
""",
    "beir": """
Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021).
BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models.
NeurIPS 2021 Datasets and Benchmarks Track.
https://arxiv.org/abs/2104.08663
""",
    "msmarco": """
Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., & Deng, L. (2016).
MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.
NeurIPS 2016 Workshop on Cognitive Computation.
https://arxiv.org/abs/1611.09268
""",
}