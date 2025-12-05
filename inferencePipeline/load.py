"""
Inference pipeline using vLLM for GPU-accelerated inference.
Loads HuggingFace models directly and runs on GPU with efficient batching.
"""

import os
import sys
import re
import time
import subprocess
from typing import List, Dict, Any

# Environment setup - Enable GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Force vLLM to avoid FLASHINFER backend (needs nvcc, not available in eval env)
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
os.environ["VLLM_USE_FLASHINFER"] = "0"
os.environ["FLASHINFER_DISABLE"] = "1"

# Make Triton/vLLM see bundled Python headers (for Python.h)
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PY_INC_DIR = os.path.join(
    PKG_DIR,
    'python_include',
    f'python{sys.version_info.major}.{sys.version_info.minor}',
)

if os.path.exists(PY_INC_DIR):
    old_c = os.environ.get('C_INCLUDE_PATH', '')
    os.environ['C_INCLUDE_PATH'] = PY_INC_DIR + (':' + old_c if old_c else '')
    old_cpp = os.environ.get('CPPFLAGS', '')
    os.environ['CPPFLAGS'] = f'-I{PY_INC_DIR}' + (f' {old_cpp}' if old_cpp else '')
    print(f"[INFO] Added Python headers to C_INCLUDE_PATH: {PY_INC_DIR}")
else:
    print(
        f"[WARN] Python include dir not found at {PY_INC_DIR}. "
        f"vLLM/Triton may fail to compile kernels (missing Python.h)."
    )


def _auto_install_dependencies():
    """
    Automatically install missing dependencies from vendor_wheels.
    This enables automatic deployment on target servers.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    vendor_wheels = os.path.join(pkg_dir, 'vendor_wheels')

    if not os.path.exists(vendor_wheels):
        print(f"Warning: vendor_wheels not found at {vendor_wheels}")
        return False

    # Check if vLLM is available
    try:
        import vllm  # noqa: F401
        return True  # Already installed
    except ImportError:
        pass

    print("vLLM not found. Installing dependencies from vendor_wheels...")
    print(f"Using wheels from: {vendor_wheels}")

    try:
        # Install from vendor_wheels with no index
        cmd = [
            sys.executable,
            '-m',
            'pip',
            'install',
            '--find-links',
            vendor_wheels,
            'vllm',
            'transformers',
            'tokenizers',
            'safetensors',
            'huggingface-hub',
            'ray',
            'fastapi',
            'uvicorn',
            'pydantic',
            'aiohttp',
            'psutil',
            'pynvml',
            '--quiet',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print("Warning: Installation completed with warnings.")
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if 'ERROR' in line or 'CRITICAL' in line:
                        print(f"  {line}")
        else:
            print("✓ Dependencies installed successfully")

        return True

    except Exception as e:
        print(f"Failed to auto-install dependencies: {e}")
        print("Please manually install: pip install vllm transformers")
        return False


# Auto-install dependencies if needed
_auto_install_dependencies()

# Now import required modules
from transformers import AutoTokenizer  # noqa: E402

# Import vLLM
try:
    from vllm import LLM, SamplingParams  # noqa: E402
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("Error: vLLM not available after installation attempt.")
    print("Please manually install: pip install vllm")
    sys.exit(1)

# ============================================================================
# Prompt & routing configuration (subject + Chinese题型路由)
# ============================================================================


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def detect_subject(question: str) -> str:
    """
    Very simple subject router:
    - Contains Chinese characters → chinese
    - Otherwise, look for typical English keywords for algebra/geography/history
    - Fallback → general
    """
    q = question.strip()
    if has_chinese(q):
        return "chinese"

    low = q.lower()

    algebra_kw = [
        "solve for",
        "simplify",
        "prove",
        "matrix",
        "det(",
        "determinant",
        "eigenvalue",
        "eigenvalues",
        "collinear",
        "quadratic",
        "slope-intercept",
        "transformation preserves vector length",
    ]
    geography_kw = [
        "latitude",
        "permafrost",
        "urbanization",
        "el niño",
        "el nino",
        "ocean current",
        "mid-ocean ridge",
        "island arc",
        "atmospheric circulation",
        "arid zones",
        "hydrology",
        "heat island",
    ]
    history_kw = [
        "renaissance",
        "treaty of versailles",
        "cold war",
        "enlightenment",
        "nationalism",
        "collapse of empires",
        "revolution",
        "decolonization",
        "industrialization",
        "social class structures",
        "16th century",
    ]

    if any(k in low for k in algebra_kw):
        return "algebra"
    if any(k in low for k in geography_kw):
        return "geography"
    if any(k in low for k in history_kw):
        return "history"

    return "general"


SYSTEM_PROMPT_GENERAL = """
You are an academically strict and concise exam assistant.

Rules:
1. Always answer in the same language as the question.
2. Give only the minimal content needed to be fully correct.
3. Prefer a single short sentence; do NOT add background, examples, or extra causes.
4. Do not repeat the question. Do not show reasoning or steps. Only output the final answer.
""".strip()

SYSTEM_PROMPT_ALGEBRA = """
You are an exam assistant for school mathematics (especially algebra and linear algebra).

Rules:
1. Questions are in English; answer in English.
2. For 'Solve' or 'Simplify' questions, output only the final result (e.g. 'x = 5', '3x + 6') with no explanation.
3. For 'Explain' or 'Why' questions, give ONE short textbook-style reason in a single sentence.
4. Do NOT repeat the question, do NOT list extra cases, and do NOT show intermediate steps.
5. Do not output chain-of-thought; only the final concise answer.
""".strip()

SYSTEM_PROMPT_GEOGRAPHY = """
You are an exam assistant for school geography.

Rules:
1. Answer in the same language as the question (here: English).
2. Give only the key mechanism or 1–2 standard factors that textbooks use.
3. One short sentence is preferred; avoid long explanations or extra background.
4. Do not add causes or impacts that the question did not ask for.
5. Do not repeat the question; output only the final concise answer.
""".strip()

SYSTEM_PROMPT_HISTORY = """
You are an exam assistant for school world history.

Rules:
1. Answer in the same language as the question (here: English).
2. Focus on the 1–2 core causes, effects, or features that appear in standard textbooks.
3. Keep the answer to one short sentence when possible.
4. Do NOT add extra ideological commentary or additional causes beyond what is necessary.
5. Do not repeat the question; output only the final concise answer.
""".strip()

BASE_SYSTEM_PROMPT_CHINESE = """
你是一名严格按照中学语文考试标准阅卷的答题助手。

总原则：
1. 严格按题干要求作答，只写得分点，不扩展、不发挥。
2. 用简洁规范的书面语，不出现口语、网络用语。
3. 优先用短语或几个关键词作答，必要时用一短句说明。
4. 不要复述题干，不要写解题过程或分析过程。
""".strip()


def build_chinese_system_prompt(question: str) -> str:
    """根据题干里的信号词，为语文题添加题型专用规则。"""
    q = question.strip()
    extra_rules = []

    # 1) 成语 / 词语解释
    if ("成语" in q or "词语" in q or "解释" in q) and "修辞" not in q:
        extra_rules.append(
            "【成语 / 词语解释题】直接写出核心寓意或用法，一句话或短语即可，不讲故事。"
        )

    # 2) 修辞手法
    if "修辞手法" in q:
        extra_rules.append(
            "【修辞手法题】只写修辞名称，如“比喻”“拟人”“夸张”等；"
            "若有两种，用顿号隔开，如“比喻、拟人”；不要写解释或例句。"
        )

    # 3) 说明方法 / 写作手法 / 表达方式 / 记叙方式
    if "说明方法" in q or "写作手法" in q or "表达方式" in q or "记叙方式" in q:
        extra_rules.append(
            "【方法名称类题】只写方法名称，如“举例子、列数字”“记叙、抒情”等，不加解释。"
        )

    # 4) 结构作用 / 结构方式
    if "结构" in q and ("作用" in q or "方式" in q):
        extra_rules.append(
            "【结构题】用2~3个短语概括，如“承上启下”“总分总结构”等，不写长段分析。"
        )

    # 5) 意境特点 / 情感特点 / 氛围
    if "意境" in q or "情感特点" in q or "氛围" in q:
        extra_rules.append(
            "【意境 / 情感题】从画面氛围和情感两方面，"
            "用一短句概括，如“营造出幽静凄清的意境，表达了思乡之情”；"
            "不要列举修辞名称。"
        )

    # 6) 态度转变 / 形象特点 / 性格特点
    if "态度转变" in q or "态度的转变过程" in q:
        extra_rules.append(
            "【态度转变题】用“先……，后……”或“由……到……”的格式，"
            "概括出两个或三个阶段，不讲具体情节。"
        )

    if "形象" in q or "性格特点" in q or "人物形象" in q:
        extra_rules.append(
            "【人物形象题】用2~3个四字短语或短语概括人物性格，如“慈爱、朴素、为儿子默默付出”。"
        )

    # 7) 价值观 / 学习态度 / 品格
    if "价值观" in q or "体现了" in q or "学习态度" in q:
        extra_rules.append(
            "【价值观 / 态度题】用1~3个四字短语作答，如“谦虚好学”“重义轻生”“勇敢机智”，"
            "不要写成长句。"
        )

    # 8) “以小见大”
    if "以小见大" in q:
        extra_rules.append(
            "【以小见大】说明这种手法的好处时，用1~2个短句或短语，"
            "如“通过细节折射人物性格”“用小事体现时代背景”。"
        )

    # 9) “简要概括 / 简要分析”
    if "简要概括" in q or "简要分析" in q:
        extra_rules.append(
            "【简要题】严格控制字数，用2~3个短语或一个很简短的句子回答，只保留核心要点。"
        )

    if extra_rules:
        return BASE_SYSTEM_PROMPT_CHINESE + "\n\n" + "\n".join(extra_rules)
    else:
        return BASE_SYSTEM_PROMPT_CHINESE


def build_messages(question: str):
    """根据题目自动选择合适的 System Prompt。"""
    subject = detect_subject(question)

    if subject == "algebra":
        sys_prompt = SYSTEM_PROMPT_ALGEBRA
    elif subject == "geography":
        sys_prompt = SYSTEM_PROMPT_GEOGRAPHY
    elif subject == "history":
        sys_prompt = SYSTEM_PROMPT_HISTORY
    elif subject == "chinese":
        sys_prompt = build_chinese_system_prompt(question)
    else:
        sys_prompt = SYSTEM_PROMPT_GENERAL

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]


class MyModel:
    """
    Inference pipeline using vLLM for efficient GPU inference.

    Uses direct batch processing with PagedAttention for optimal GPU utilization.
    """

    # ========== Configuration Hub ==========
    CONFIG = {
        'model_name': 'Qwen/Qwen3-4B',
        'max_tokens': 64,
        # 考试型任务偏稳定，适当降低温度
        'temperature': 0.6,

        # GPU Configuration
        'gpu_memory_utilization': 0.7,  # 留一点余量，避免评测环境上 OOM
        'max_model_len': 2048,          # Context window
        'max_num_seqs': 32,             # Parallel request slots
        'tensor_parallel_size': 1,      # Single GPU
    }
    # =======================================

    def __init__(self):
        self.llm = None
        self.sampling_params = None
        self.get_model()

    def __call__(self, questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Process a batch of questions and return answers."""
        return self._vllm_inference(questions)

    def get_model(self):
        """Load model and prepare for inference."""
        cfg = self.CONFIG
        self.model_name = cfg['model_name']
        # Support both server path and HF_HOME
        self.cache_dir = os.environ.get(
            'HF_HOME',
            os.environ.get('TRANSFORMERS_CACHE', '/app/models'),
        )
        self.max_new_tokens = cfg['max_tokens']
        self.temperature = cfg['temperature']

        print(f"Loading model: {self.model_name}")
        print("Inference mode: vLLM (GPU)")

        # Load tokenizer (needed for chat template)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Find HuggingFace snapshot directory
        hf_snapshot = self._find_hf_snapshot()
        print(f"Found HF snapshot: {hf_snapshot}")

        # Initialize vLLM
        self._load_vllm(hf_snapshot)
        print("vLLM model loaded successfully!")

    def _find_hf_snapshot(self):
        """Find the HuggingFace model snapshot directory."""
        model_slug = self.model_name.replace('/', '--')
        model_base = os.path.join(self.cache_dir, f'models--{model_slug}')

        if not os.path.exists(model_base):
            raise FileNotFoundError(f"Model not found in cache: {model_base}")

        snapshots_dir = os.path.join(model_base, 'snapshots')
        if not os.path.exists(snapshots_dir):
            raise FileNotFoundError(f"No snapshots directory: {snapshots_dir}")

        snapshots = os.listdir(snapshots_dir)
        if not snapshots:
            raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")

        return os.path.join(snapshots_dir, snapshots[0])

    def _load_vllm(self, model_path: str):
        """Load model using vLLM."""
        cfg = self.CONFIG

        print("Initializing vLLM...")
        print(
            f"GPU config: memory={cfg['gpu_memory_utilization']}, "
            f"context={cfg['max_model_len']}, slots={cfg['max_num_seqs']}",
        )

        # Initialize LLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=cfg['tensor_parallel_size'],
            gpu_memory_utilization=cfg['gpu_memory_utilization'],
            max_model_len=cfg['max_model_len'],
            max_num_seqs=cfg['max_num_seqs'],
            trust_remote_code=True,
            enforce_eager=True,        # use eager; CUDA graphs disabled,稳定
            disable_log_stats=True,    # reduce logging overhead
        )

        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            skip_special_tokens=True,
        )

    def _vllm_inference(self, questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Process questions using vLLM batch inference."""
        total = len(questions)
        print(f"Processing {total} questions via vLLM...")

        start_time = time.time()

        # Format all prompts with routing
        prompts: List[str] = []
        for q in questions:
            messages = build_messages(q['question'])
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Disable Qwen3 thinking feature
            )
            prompts.append(prompt)

        # Batch inference (vLLM handles parallelization internally)
        print("Running batch inference...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Extract answers
        answers: List[Dict[str, str]] = []
        for q, output in zip(questions, outputs):
            raw_answer = output.outputs[0].text
            clean_answer = self._extract_answer(raw_answer)
            answers.append(
                {
                    'questionID': q['questionID'],
                    'answer': clean_answer[:5000],
                }
            )

        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0.0
        print(f"Completed {total} questions in {elapsed:.2f}s ({rate:.2f} q/s)")

        return answers

    def _extract_answer(self, raw_output: str) -> str:
        """Extract clean answer from model output."""
        if not raw_output:
            return ''

        # Remove <think> tags (if any)
        answer = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)

        # Remove special tokens and obvious role tags
        for token in [
            '[end of text]',
            '<|im_end|>',
            '<|endoftext|>',
            '<|im_start|>',
            'system',
            'user',
            'assistant',
        ]:
            answer = answer.replace(token, '')

        # Clean whitespace
        answer = ' '.join(answer.split())
        return answer


def loadPipeline():
    """
    Load the inference pipeline.
    This function is called by run.py as per task requirements.
    """
    return MyModel()


if __name__ == '__main__':
    # Simple local test
    import json

    pipeline = loadPipeline()

    questions = [
        {'questionID': '1', 'question': 'What is the capital of Ireland?'},
        {'questionID': '2', 'question': 'What is 2 + 2?'},
        {'questionID': '3', 'question': '解释成语“画蛇添足”的意思。'},
        {'questionID': '4', 'question': '简要概括《背影》中父亲形象的主要特点。'},
    ]

    answers = pipeline(questions)
    print(json.dumps(answers, ensure_ascii=False, indent=2))
