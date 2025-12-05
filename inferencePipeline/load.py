"""
Inference pipeline using llama.cpp for CPU-optimized inference.
Converts HuggingFace models to GGUF format and runs quantized inference.
"""

import os
import sys
import json
import re
import shutil
import subprocess
import tempfile
import multiprocessing

# Environment setup - must be before any ML library imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoTokenizer


def _parallel_inference_worker(args):
    """
    Worker function for parallel inference.
    Must be at module level for multiprocessing.pickle.
    """
    question_data, model_path, bin_path, tokenizer_name, cache_dir, max_tokens, threads = args
    
    try:
        # Load tokenizer in worker process
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Format prompt
        system_prompt = (
            "You are an expert tutor for algebra, geography, and history. "
            "Always answer within 1 sentence (<=64 new tokens). "
            "Put the final answer first. "
            "Algebra: give the result or a minimal formula first; at most one brief justification. "
            "Geography/History: one factual sentence, include a key date/name if needed. "
            "Avoid long explanations and lists; do not repeat the question or any role tags."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_data['question']}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Write prompt to temp file
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.txt') as tf:
            tf.write(prompt)
            prompt_path = tf.name
        
        try:
            # Clean environment
            clean_env = os.environ.copy()
            for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                clean_env.pop(key, None)
            
            # Run llama-cli
            cmd = [
                bin_path,
                '-m', model_path,
                '-f', prompt_path,
                '-n', str(max_tokens),
                '-t', str(threads),
                '-no-cnv',
                '-ngl', os.environ.get('LLAMA_NGL', '100'),
                '--no-warmup',
                '-c', '2048',
                '-b', '512',
                '--mlock',
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=60,
                env=clean_env,
                stdin=subprocess.DEVNULL
            )
            
            raw_output = result.stdout if result.returncode == 0 else ''
            
            # Extract answer
            answer = _extract_answer_static(raw_output)
            
            return {
                'questionID': question_data['questionID'],
                'answer': answer[:5000]
            }
            
        finally:
            try:
                os.unlink(prompt_path)
            except:
                pass
                
    except Exception as e:
        return {
            'questionID': question_data['questionID'],
            'answer': ''
        }


def _extract_answer_static(raw_output):
    """Static version of answer extraction for use in worker processes."""
    if not raw_output:
        return ''
    
    # Look for "assistant" marker
    if 'assistant' in raw_output:
        parts = raw_output.split('assistant')
        if len(parts) > 1:
            answer = parts[-1].strip()
        else:
            answer = raw_output.strip()
    else:
        answer = raw_output.strip()
    
    # Remove artifacts
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    
    for token in ['[end of text]', '<|im_end|>', '<|endoftext|>', 
                  '<|im_start|>', 'system', 'user']:
        answer = answer.replace(token, '')
    
    answer = ' '.join(answer.split())
    
    return answer


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class MyModelGPU:
    def __init__(self):
        self.model_name = 'Qwen/Qwen3-4B'
        self.cache_dir = '/app/models'
        self.max_new_tokens = 64
        self.batch_size = int(os.environ.get('BATCH_SIZE', '8'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

    def __call__(self, questions):
        answers = []
        # 按 batch 切分
        for i in range(0, len(questions), self.batch_size):
            batch = questions[i:i + self.batch_size]
            batch_answers = self._answer_batch(batch)
            answers.extend(batch_answers)
        return answers

    def _answer_batch(self, batch):
        prompts = []
        for q in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q['question']}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(prompt)

        enc = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy，稳定 & 快
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        results = []
        for q, text in zip(batch, decoded):
            ans = _extract_answer_static(text)  # 复用你们的清洗函数
            results.append({
                'questionID': q['questionID'],
                'answer': ans[:5000]
            })
        return results


class MyModel:
    """
    Inference pipeline using llama.cpp for efficient CPU inference.
    """
    
    def __init__(self):
        self.get_model()
    
    def __call__(self, questions):
        """Process a batch of questions and return answers."""
        # Check if parallel processing is enabled
        if self.num_workers > 1:
            return self._parallel_inference(questions)
        else:
            return self._serial_inference(questions)
    
    def _serial_inference(self, questions):
        """Original serial processing."""
        answers = []
        total = len(questions)
        
        for idx, q in enumerate(questions, 1):
            print(f"[{idx}/{total}] Processing question {q['questionID']}...", end=' ', flush=True)
            try:
                answer = self.get_answer(q['question'])
            except Exception as e:
                print(f"Error: {e}")
                answer = ''
            else:
                print("Done")
            
            answers.append({
                'questionID': q['questionID'],
                'answer': answer[:5000]
            })
        
        return answers
    
    def _parallel_inference(self, questions):
        """Parallel processing using multiprocessing."""
        total = len(questions)
        print(f"Processing {total} questions with {self.num_workers} parallel workers...")
        
        # Prepare arguments for workers
        worker_args = [
            (q, self.gguf_path, self.llama_bin, self.model_name, 
             self.cache_dir, self.max_new_tokens, self.worker_threads)
            for q in questions
        ]
        
        # Use multiprocessing pool
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            answers = pool.map(_parallel_inference_worker, worker_args)
        
        print(f"Completed {total} questions")
        return answers
    
    def get_model(self):
        """
        Load model and prepare for inference.
        - Resolves HuggingFace model from /app/models cache
        - Converts to GGUF format if needed
        - Selects appropriate llama.cpp binary
        """
        # Configuration
        self.model_name = 'Qwen/Qwen3-1.7B'
        # Quantization options:
        # - q4_0: Fastest, slightly lower quality
        # - q4_K_M: Good balance (current default)
        # - q5_K_M: Higher quality, slightly slower
        # - q8_0: Best quality, slower and larger
        self.quantization = os.environ.get('MODEL_QUANT', 'q4_K_M')
        # self.quantization = os.environ.get('MODEL_QUANT', 'q5_K_M')
        self.cache_dir = '/app/models'
        self.max_new_tokens = 64
        
        print(f"Loading model: {self.model_name} (quantization: {self.quantization})")
        
        # Load tokenizer (needed for chat template)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Find HuggingFace snapshot directory
        hf_snapshot = self._find_hf_snapshot()
        print(f"Found HF snapshot: {hf_snapshot}")
        
        # Prepare GGUF model
        self.gguf_path = self._prepare_gguf(hf_snapshot)
        print(f"GGUF model ready: {self.gguf_path}")
        
        # Select llama.cpp binary
        self.llama_bin = self._select_llama_binary()
        
        # Parallel processing configuration
        # PARALLEL_WORKERS: number of parallel processes (default: 8 for 16-core CPU)
        # Set to 1 to disable parallel processing
        self.num_workers = int(os.environ.get('PARALLEL_WORKERS', '8'))
        
        if self.num_workers > 1:
            # Parallel mode: distribute threads across workers
            # Each worker gets fewer threads to avoid oversubscription
            total_threads = int(os.environ.get('INFERENCE_NUM_THREADS', '16'))
            self.worker_threads = max(1, total_threads // self.num_workers)
            self.llama_threads = self.worker_threads  # For serial fallback
            print(f"Using llama-cli: {os.path.basename(self.llama_bin)}")
            print(f"Parallel mode: {self.num_workers} workers × {self.worker_threads} threads = {self.num_workers * self.worker_threads} total")
        else:
            # Serial mode: use all threads in single process
            self.llama_threads = int(os.environ.get('INFERENCE_NUM_THREADS', '16'))
            self.worker_threads = self.llama_threads
            self.llama_threads_batch = self.llama_threads
            print(f"Using llama-cli: {os.path.basename(self.llama_bin)} (threads: {self.llama_threads})")
            print(f"Serial mode: 1 worker × {self.llama_threads} threads")
    
    def _find_hf_snapshot(self):
        """Find the HuggingFace model snapshot directory."""
        model_slug = self.model_name.replace('/', '--')
        model_base = os.path.join(self.cache_dir, f'models--{model_slug}')
        
        if not os.path.exists(model_base):
            raise FileNotFoundError(f"Model not found in cache: {model_base}")
        
        snapshots_dir = os.path.join(model_base, 'snapshots')
        if not os.path.exists(snapshots_dir):
            raise FileNotFoundError(f"No snapshots directory: {snapshots_dir}")
        
        # Get the first (and usually only) snapshot
        snapshots = os.listdir(snapshots_dir)
        if not snapshots:
            raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
        
        return os.path.join(snapshots_dir, snapshots[0])
    
    def _prepare_gguf(self, hf_snapshot):
        """
        Convert HuggingFace model to GGUF format if needed.
        Returns path to the final GGUF file.
        """
        pkg_dir = os.path.dirname(__file__)
        gguf_dir = os.path.join(pkg_dir, 'models-gguf')
        os.makedirs(gguf_dir, exist_ok=True)
        
        # Generate GGUF filename
        model_slug = self.model_name.replace('/', '--')
        gguf_filename = f"{model_slug}-{self.quantization}.gguf"
        gguf_path = os.path.join(gguf_dir, gguf_filename)
        
        # Check if already exists
        if os.path.exists(gguf_path) and os.path.getsize(gguf_path) > 1024*1024:
            print(f"Using existing GGUF: {gguf_filename}")
            return gguf_path
        
        # Need to convert
        print(f"Converting to GGUF format...")
        
        # Step 1: Install conversion dependencies offline
        self._install_conversion_deps()
        
        # Step 2: Convert to f16 first (convert-hf-to-gguf.py only supports f16/f32)
        f16_filename = f"{model_slug}-f16.gguf"
        f16_path = os.path.join(gguf_dir, f16_filename)
        
        if not os.path.exists(f16_path) or os.path.getsize(f16_path) < 1024*1024:
            self._convert_to_f16(hf_snapshot, f16_path)
        
        # Step 3: Quantize to target format if not f16
        if self.quantization != 'f16':
            self._quantize_gguf(f16_path, gguf_path)
        else:
            gguf_path = f16_path
        
        # Verify final file
        if not os.path.exists(gguf_path) or os.path.getsize(gguf_path) < 1024*1024:
            raise RuntimeError(f"GGUF conversion failed: {gguf_path}")
        
        size_mb = os.path.getsize(gguf_path) / (1024*1024)
        print(f"GGUF ready: {gguf_filename} ({size_mb:.0f} MB)")
        
        return gguf_path
    
    def _install_conversion_deps(self):
        """Install numpy, sentencepiece, tokenizers from vendor wheels."""
        pkg_dir = os.path.dirname(__file__)
        wheels_dir = os.path.join(pkg_dir, 'vendor_wheels')
        
        if not os.path.exists(wheels_dir):
            print("Warning: vendor_wheels not found, skipping offline install")
            return
        
        # Find python executable (avoid Cursor wrapper)
        python_exe = self._find_python()
        
        # Install all wheels
        wheels = [f for f in os.listdir(wheels_dir) if f.endswith('.whl')]
        if wheels:
            print(f"Installing {len(wheels)} offline dependencies...")
            subprocess.run(
                [python_exe, '-m', 'pip', 'install', '--no-index', '--find-links', wheels_dir] + 
                ['numpy', 'sentencepiece', 'tokenizers'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120
            )
    
    def _convert_to_f16(self, hf_snapshot, output_path):
        """Convert HuggingFace model to f16 GGUF using convert-hf-to-gguf.py."""
        pkg_dir = os.path.dirname(__file__)
        convert_script = os.path.join(pkg_dir, 'scripts', 'convert-hf-to-gguf.py')
        
        if not os.path.exists(convert_script):
            raise FileNotFoundError(f"Conversion script not found: {convert_script}")
        
        python_exe = self._find_python()
        
        print(f"Converting {self.model_name} to f16 GGUF...")
        result = subprocess.run(
            [python_exe, convert_script, hf_snapshot, '--outfile', output_path, '--outtype', 'f16'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"Conversion stderr: {result.stderr[-1000:]}")
            raise RuntimeError(f"Conversion failed with exit code {result.returncode}")
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"Conversion completed but output file missing: {output_path}")
    
    def _quantize_gguf(self, f16_path, output_path):
        """Quantize f16 GGUF to target quantization using llama-quantize."""
        pkg_dir = os.path.dirname(__file__)
        
        # Find llama-quantize binary
        quantize_bin = None
        for candidate in ['llama-quantize-avx2', 'llama-quantize-generic']:
            path = os.path.join(pkg_dir, 'bin', candidate)
            if os.path.exists(path):
                quantize_bin = path
                break
        
        if not quantize_bin:
            print(f"Warning: llama-quantize not found, using f16")
            return f16_path
        
        print(f"Quantizing to {self.quantization}...")
        result = subprocess.run(
            [quantize_bin, f16_path, output_path, self.quantization],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"Quantization stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"Quantization failed with exit code {result.returncode}")
    
    def _select_llama_binary(self):
        """Select appropriate llama-cli binary (AVX2 or generic)."""
        pkg_dir = os.path.dirname(__file__)
        
        # Try AVX2 first (faster)
        avx2_bin = os.path.join(pkg_dir, 'bin', 'llama-cli-avx2')
        if os.path.exists(avx2_bin):
            # Test if AVX2 works
            try:
                subprocess.run([avx2_bin, '--version'], 
                             capture_output=True, timeout=5, check=True)
                return avx2_bin
            except:
                pass
        
        # Fall back to generic
        generic_bin = os.path.join(pkg_dir, 'bin', 'llama-cli-generic')
        if os.path.exists(generic_bin):
            return generic_bin
        
        raise FileNotFoundError("No llama-cli binary found")
    
    def _find_python(self):
        """Find the correct Python executable (avoid Cursor wrapper)."""
        # Try conda/venv first
        if 'CONDA_PREFIX' in os.environ:
            conda_python = os.path.join(os.environ['CONDA_PREFIX'], 'bin', 'python3')
            if os.path.exists(conda_python):
                return conda_python
        
        if 'VIRTUAL_ENV' in os.environ:
            venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'bin', 'python3')
            if os.path.exists(venv_python):
                return venv_python
        
        # Try system python
        for cmd in ['python3', 'python']:
            python_path = shutil.which(cmd)
            if python_path and 'cursor' not in python_path.lower():
                return python_path
        
        # Last resort
        return sys.executable
    
    def get_answer(self, question):
        """
        Generate answer for a single question using llama.cpp.
        """
        # Format prompt using tokenizer's chat template
        system_prompt = (
            "You are an expert tutor for algebra, geography, and history. "
            "Always answer within 1 sentence (<=64 new tokens). "
            "Put the final answer first. "
            "Algebra: give the result or a minimal formula first; at most one brief justification. "
            "Geography/History: one factual sentence, include a key date/name if needed. "
            "Avoid long explanations and lists; do not repeat the question or any role tags."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Run llama-cli in isolated process
        output = self._run_llama_cli(prompt, self.max_new_tokens)
        
        # Post-process answer
        answer = self._extract_answer(output)
        return answer
    
    def _run_llama_cli(self, prompt_text, max_new_tokens):
        """
        Run llama-cli using subprocess directly.
        We avoid multiprocessing because it can inherit Cursor's process context.
        """
        # Write prompt to temp file
        with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.txt') as tf:
            tf.write(prompt_text)
            prompt_path = tf.name
        
        try:
            # Clean environment from torch's thread settings
            clean_env = os.environ.copy()
            for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 
                        'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                clean_env.pop(key, None)
            
            # Build command with performance optimizations
            cmd = [
                self.llama_bin,
                '-m', self.gguf_path,
                '-f', prompt_path,
                '-n', str(max_new_tokens),
                '-t', str(self.llama_threads),           # Main threads
                '-tb', str(self.llama_threads_batch),    # Batch processing threads
                '-no-cnv',                                # Disable conversation mode
                '--no-warmup',                            # Skip warmup
                '-c', '2048',                             # Context size (smaller = faster)
                '-b', '512',                              # Batch size for prompt processing
                '--mlock',                                # Lock model in RAM (prevent swapping)
            ]
            
            # Run llama-cli
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Suppress llama.cpp logs
                text=True,
                timeout=60,
                env=clean_env,
                stdin=subprocess.DEVNULL
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return ''
                
        except subprocess.TimeoutExpired:
            return ''
        except Exception as e:
            print(f"Error running llama-cli: {e}")
            return ''
        finally:
            try:
                os.unlink(prompt_path)
            except:
                pass
    
    def _extract_answer(self, raw_output):
        """Extract clean answer from llama.cpp output."""
        if not raw_output:
            return ''
        
        # The output includes the full prompt + generation
        # We need to extract only the assistant's response
        
        # Look for "assistant" marker (from chat template)
        if 'assistant' in raw_output:
            # Split at last occurrence of "assistant"
            parts = raw_output.split('assistant')
            if len(parts) > 1:
                answer = parts[-1].strip()
            else:
                answer = raw_output.strip()
        else:
            answer = raw_output.strip()
        
        # Remove common artifacts
        # Remove <think> tags and their content
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        
        # Remove special tokens
        for token in ['[end of text]', '<|im_end|>', '<|endoftext|>', 
                      '<|im_start|>', 'system', 'user']:
            answer = answer.replace(token, '')
        
        # Clean up whitespace
        answer = ' '.join(answer.split())
        
        return answer[:5000]  # Enforce character limit


def loadPipeline():
    """
    Load the inference pipeline.
    This function is called by run.py as per task requirements.
    """
    use_gpu = os.environ.get('USE_GPU_TRANSFORMERS', '0') == '1'
    if use_gpu:
        return MyModelGPU()
    else:
        return MyModel()


if __name__ == '__main__':
    # Test the pipeline
    pipeline = loadPipeline()
    
    questions = [
        {'questionID': 1, 'question': 'What is the capital of Ireland?'},
        {'questionID': 2, 'question': 'What is 2 + 2?'}
    ]
    
    answers = pipeline(questions)
    print(json.dumps(answers, indent=2))
