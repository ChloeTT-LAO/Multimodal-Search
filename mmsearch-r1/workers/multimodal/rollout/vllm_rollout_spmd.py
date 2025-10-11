"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import copy
import pickle
import re
import threading
from concurrent.futures import (  # parallelize search call
    ThreadPoolExecutor,
    as_completed,
)
from copy import deepcopy
from typing import Any, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tqdm import tqdm
from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils import hf_processor
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _repeat_interleave,
    vLLMRollout,
)

from mmsearch_r1.utils.tools.image_search import call_image_search
from mmsearch_r1.utils.tools.text_search import call_text_search
from mmsearch_r1.utils.torch_functional import get_final_eos_mask

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(
            torch.cat([t, torch.tensor([pad_token_id] * (max_len - t.size(0)), device=t.device, dtype=t.dtype)], dim=0)
        )
    return torch.stack(padded_tensor_list, dim=dim)


class vLLMRollout_MultiTurn_MMSearch_R1(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):

        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        # add tokenizer
        self.tokenizer = tokenizer
        # add processor
        self.processor = hf_processor(model_path)

        # 新增：query拆分和综合的配置
        self.enable_query_decomposition = config.get('enable_query_decomposition', False)
        self.num_sub_queries = config.get('num_sub_queries', 3)


        self.user_prompt_after_image_search = None
        self.user_prompt_after_text_search = None
        try:
            with open(config.search.user_prompt_after_image_search, 'rb') as file:
                self.user_prompt_after_image_search = pickle.load(file)
        except Exception as e:
            print(f"Error: {e} | user_prompt_after_image_search default to None")
        try:
            with open(config.search.user_prompt_after_text_search, 'rb') as file:
                self.user_prompt_after_text_search = pickle.load(file)
        except Exception as e:
            print(f"Error: {e} | user_prompt_after_text_search default to None")

        print(f"[Prompt Set] user_prompt_after_text_search: {self.user_prompt_after_text_search}")
        print(f"[Prompt Set] user_prompt_after_image_search: {self.user_prompt_after_image_search}")

    def decompose_query(self, original_query: str, image_inputs: list = None) -> list:
        """
        将原始query从多个角度拆分为不同的子query

        Args:
            original_query: 原始问题
            image_data: 图像数据

        Returns:
            sub_queries: 子query列表
        """
        decomposition_prompt = f"""You are a helpful assistant that decomposes complex questions into simpler sub-questions.

            Given the following question, please break it down into {self.num_sub_queries} different sub-questions that explore different aspects or perspectives.
            
            Each sub-question should:
            1. Be self-contained and answerable independently
            2. Focus on a specific aspect (e.g., what, where, when, who, why, how, which details)
            3. Help answer the original question when combined
            
            Original Question: {original_query}
            
            Please provide exactly {self.num_sub_queries} sub-questions in this format:
            1. [First sub-question]
            2. [Second sub-question]
            3. [Third sub-question]
            
            Sub-questions:"""

        messages = []
        content_items = [{"type": "text", "content": decomposition_prompt}]
        if image_inputs:
            for img in image_inputs:
                content_items.append({"type": "image", "image": img})
        messages.append({"role": "user", "content": content_items})

        # 使用processor的apply_chat_template方法
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 使用tokenizer编码
        prompt_token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # 准备multi_modal_data
        multi_modal_data = {}
        if image_inputs:
            multi_modal_data['image'] = image_inputs

        vllm_input = {
            'prompt_token_ids': prompt_token_ids,
            'multi_modal_data': multi_modal_data
        }

        # 生成
        original_sampling_params = copy.deepcopy(self.sampling_params)
        self.sampling_params.temperature = 0.7
        self.sampling_params.max_tokens = 512
        self.sampling_params.n = 1

        # 在生成之前
        print(f"[DEBUG] decomposition_prompt: {decomposition_prompt[:200]}")
        print(f"[DEBUG] prompt_token_ids length: {len(prompt_token_ids)}")
        print(f"[DEBUG] image_inputs count: {len(image_inputs) if image_inputs else 0}")
        print(f"[DEBUG] vllm_input: {vllm_input.keys()}")

        outputs = self.inference_engine.generate(
            prompts=[vllm_input],
            sampling_params=self.sampling_params,
            use_tqdm=False
        )

        # 在生成之后
        print(f"[DEBUG] outputs count: {len(outputs)}")
        print(f"[DEBUG] outputs[0].outputs count: {len(outputs[0].outputs)}")
        response_text = outputs[0].outputs[0].text
        print(f"[DEBUG] response_text: '{response_text}'")  # 关键！看看是不是空的
        print(f"[DEBUG] response_text length: {len(response_text)}")

        # 恢复原sampling params
        self.sampling_params = original_sampling_params

        # 解析子queries
        response_text = outputs[0].outputs[0].text
        sub_queries = self._parse_sub_queries(response_text)

        print(f"\n{'=' * 60}")
        print(f"[Query Decomposition]")
        print(f"Original: {original_query}")
        for idx, sq in enumerate(sub_queries):
            print(f"  Sub-{idx + 1}: {sq}")
        print(f"{'=' * 60}\n")

        return sub_queries

    def _parse_sub_queries(self, response_text: str) -> list:
        """从模型响应中解析子queries"""
        sub_queries = []

        # 方法1: 匹配 "1. xxx" 或 "1) xxx" 格式
        lines = response_text.strip().split('\n')
        for line in lines:
            match = re.match(r'^\s*(\d+)[\.\)]\s*(.+)', line)
            if match:
                query = match.group(2).strip()
                if query and len(query) > 5:
                    sub_queries.append(query)

        # 方法2: 如果方法1失败，使用正则表达式
        if not sub_queries:
            pattern = r'\d+[\.\)]\s*(.+?)(?=\n\d+[\.\)]|\Z)'
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                query = match.strip()
                if query and len(query) > 5:
                    sub_queries.append(query)

        # 方法3: 如果还是失败，按段落分割
        if not sub_queries:
            paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
            sub_queries = [p for p in paragraphs if len(p) > 10][:self.num_sub_queries]

        # 如果完全失败，返回原始文本的分段
        if not sub_queries:
            print("[Warning] Failed to parse sub-queries, using simple split")
            words = response_text.split()
            chunk_size = max(len(words) // self.num_sub_queries, 5)
            for i in range(self.num_sub_queries):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(words))
                sub_queries.append(' '.join(words[start:end]))

        # 限制数量并清理
        sub_queries = sub_queries[:self.num_sub_queries]
        sub_queries = [q.strip('.,;:!?') for q in sub_queries]

        return sub_queries

    def synthesize_answers(self, original_query: str, sub_results: list, image_inputs: list = None) -> str:
        """
        综合所有子query的答案生成最终答案
        """
        synthesis_prompt = f"""# Task: Synthesize a Comprehensive Final Answer

    ## Original Question
    {original_query}

    ## Information from Sub-questions

    """

        # 添加每个子query的结果
        for idx, (sub_query, answer, reasoning) in enumerate(sub_results):
            synthesis_prompt += f"### Sub-question {idx + 1}\n"
            synthesis_prompt += f"**Question:** {sub_query}\n"
            synthesis_prompt += f"**Answer:** {answer}\n"
            if reasoning and len(reasoning) > 0:
                synthesis_prompt += f"**Reasoning:** {reasoning}\n"
            synthesis_prompt += "\n"

        synthesis_prompt += """## Your Task
    Please synthesize all the information above to provide a comprehensive answer to the original question.

    **Format Requirements:**
    1. First, provide your reasoning inside <reason>...</reason> tags
    2. Then, provide the final answer inside <answer>...</answer> tags
    3. The final answer should be clear, concise, and directly address the original question

    Your response:"""

        # 构建messages
        messages = []
        content_items = [{"type": "text", "text": synthesis_prompt}]

        # 添加原始图像
        if image_inputs:
            for img in image_inputs:
                content_items.append({"type": "image", "image": img})

        messages.append({"role": "user", "content": content_items})

        # 使用processor的apply_chat_template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 使用tokenizer编码
        prompt_token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # 准备multi_modal_data
        multi_modal_data = {}
        if image_inputs:
            multi_modal_data['image'] = image_inputs

        vllm_input = {
            'prompt_token_ids': prompt_token_ids,
            'multi_modal_data': multi_modal_data
        }

        # 生成最终答案
        original_sampling_params = copy.deepcopy(self.sampling_params)
        self.sampling_params.temperature = 0.3
        self.sampling_params.max_tokens = 1024
        self.sampling_params.n = 1

        outputs = self.inference_engine.generate(
            prompts=[vllm_input],
            sampling_params=self.sampling_params,
            use_tqdm=False
        )

        # 恢复原sampling params
        self.sampling_params = original_sampling_params

        final_response = outputs[0].outputs[0].text

        print(f"\n[Answer Synthesis]")
        print(f"Final Response: {final_response[:200]}...")

        return final_response

    def _create_sub_query_dataproto(self, sub_query: str, image_inputs: list,
                                    original_prompts: DataProto, batch_idx: int) -> DataProto:
        """
        为子query创建新的DataProto
        """
        # 从原始prompt中提取instruction部分
        original_raw_ids = original_prompts.non_tensor_batch['raw_prompt_ids'][batch_idx]
        original_text = self.tokenizer.decode(original_raw_ids, skip_special_tokens=True)

        # 提取instruction（question之前的内容）
        instruction_match = re.search(r'^(.*?)Question:', original_text, re.DOTALL | re.IGNORECASE)
        if instruction_match:
            instruction = instruction_match.group(1).strip()
        else:
            instruction = "Answer the user's question based on the provided image."

        # 构建新prompt
        new_prompt_text = f"{instruction}\nQuestion: {sub_query}"

        # 构建messages
        messages = []
        content_items = [{"type": "text", "text": new_prompt_text}]

        # 添加图像
        if image_inputs:
            for img in image_inputs:
                content_items.append({"type": "image", "image": img})

        messages.append({"role": "user", "content": content_items})

        # 使用processor的apply_chat_template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 使用tokenizer编码
        input_ids_list = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long)

        # 创建attention_mask
        attention_mask = torch.ones_like(input_ids)

        # 构建position_ids
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long)

        # 对于Qwen2-VL的mrope，需要3D position_ids
        if hasattr(self.config, 'model') and 'qwen2' in str(self.config.model.get('path', '')).lower():
            position_ids = position_ids.unsqueeze(0).expand(3, -1).unsqueeze(0)
        else:
            position_ids = position_ids.unsqueeze(0)

        # 构建DataProto
        new_dataproto = DataProto(
            batch=TensorDict({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            }, batch_size=1),
            non_tensor_batch={
                'raw_prompt_ids': np.array([input_ids_list], dtype=object),
                'multi_modal_data': [{'image': image_inputs}] if image_inputs else [{}],
            },
            meta_info=original_prompts.meta_info.copy()
        )

        return new_dataproto

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        print(f">>> vllm_rollout_spmd Rollout Starts ...")

        # 新增Multi - Query入口
        if self.enable_query_decomposition and not prompts.meta_info.get('validate', False):
            print(f">>> [Multi-Query Mode] Enabled with {self.num_sub_queries} sub-queries")
            return self._multi_query_generate_sequences(prompts, **kwargs)
        # === 新增结束 ===

        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        # All for 1st USER prompt
        idx = prompts.batch['input_ids']  # (B'*R, max_prompt_length), left padding with |end_of_text|
        batch_size = idx.size(0)  # B'
        # for logit_log_prob & loss computation
        attention_mask = prompts.batch['attention_mask']  # (B'*R, max_prompt_length), left padding 0
        position_ids = prompts.batch['position_ids']  # (B'*R, max_prompt_length), left padding 0
        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']  # [151645, 151643] -> ｜im_end｜, |end_of_text|
        input_prompt_generation_mask = torch.zeros_like(
            idx, dtype=attention_mask.dtype, device=attention_mask.device
        )  # (B'*R, max_prompt_length), all 0

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1,  # if greedy, only 1 response
            }

        n = 1 if prompts.meta_info.get('validate', False) else self.config.n  # TODO: for validate, do_sample=False

        ##### Initialization #####
        vllm_inputs = (
            []
        )  # B*R, list of dict, into -> vllm.engine, each dict with keys: 'prompt_token_ids', 'multi_modal_data', the values are 'raw_prompt_ids' and [PIL.Image]
        multi_turn_response_mask = []  # B*R, list of list of Tensor, for distinguish 'USER tokens' & 'ASSISTANT tokens'
        prefix_prompt_lengths = []  # B*R, list of int, record first round prompt of all trajs
        search_tool_return_images = []

        # We manually repeart trajs for rollout, since some trajs need multi-round self.inference_engine.generate() with `sampling_n=1`
        if 'multi_modal_data' in non_tensor_batch:
            _multi_modal_data_list = non_tensor_batch['multi_modal_data']
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'), _multi_modal_data_list):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    # NOTE: use deepcopy to seperate variables
                    vllm_inputs.append(
                        {
                            'prompt_token_ids': deepcopy(raw_prompt_ids),
                            'multi_modal_data': deepcopy(multi_modal_data),
                        }  # raw_prompt_ids: list
                    )
                    multi_turn_response_mask.append(
                        [
                            torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                        ]  # USER, Mark as 0
                    )  # [torch.Tensor(prefix_length,)]
                    prefix_prompt_lengths.append(prefix_length)
                    search_tool_return_images.append([])  # init as empty lists

        # We need 'image_urls' for search, the shape should be aligned with B*R
        if 'image_urls' in non_tensor_batch.keys() and not prompts.meta_info.get('validate', False):
            non_tensor_batch['image_urls'] = _repeat_interleave(non_tensor_batch['image_urls'], self.config.n)

        ##### Loop Setting #####
        to_generate = list(range(batch_size * n))  # B*R, all trajs' index
        worker_trajs_count = len(to_generate)
        id_image_gen_cnt = [0] * (batch_size * n)
        max_image_gen_round = self.config.search.image_search_limit # Image Search Constraint
        id_text_gen_cnt = [0] * (batch_size * n)
        max_text_gen_round = self.config.search.text_search_limit # Text Search Constraint
        # Add pbar for better monitoring
        with tqdm(total=worker_trajs_count, desc="Worker Rollout Progress", unit="task") as pbar:
            current_iteration = 0
            max_iterations = self.config.max_gen_round
            while current_iteration < max_iterations and len(to_generate) > 0:
                # Prepare prompts to generation
                idx_to_gen = []  # list of vllm_inputs, at first the length is B'*R
                for i in to_generate:
                    idx_to_gen.append(vllm_inputs[i])

                print(
                    f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ..."
                )

                # users can customize different sampling_params at different run
                with self.update_sampling_params(n=1):  # TODO: for validate, do_sample=False
                    outputs = self.inference_engine.generate(
                        prompts=idx_to_gen, sampling_params=self.sampling_params, use_tqdm=False  # list of dict
                    )

                response = []  # list of tuple, B'*R, valid(no-pad) response_ids with unequal length
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        # HACK: filter > (voc_size+specidal_token_num) token_ids, 151664 for qwen model
                        _token_ids = output.outputs[sample_id].token_ids
                        filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
                        if 151645 not in filtered_token_ids:
                            # replace the last token with <|im_end|> if no <|im_end|> in response,
                            # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                            filtered_token_ids[-1] = 151645

                        response.append(filtered_token_ids)

                # attach model responses to vllm_inputs
                assert len(to_generate) == len(response)

                idx_to_remove = []
                id_search_query_mapping = {}
                for i_gen, response_ in zip(to_generate, response):
                    # update conversation
                    response_ = list(response_)
                    vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(
                        torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)
                    )  # ASSISTANT, Mark as 1

                    # [SEARCH TRIGGER] We check model's last turn response, if not any <xxx_search>, then remove this traj from to_generate
                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
                    # Need to call image search
                    if re.search(r'<search><img></search>$', decoded_resp_):
                        assert str(i_gen) not in id_search_query_mapping.keys()
                        if (
                            id_image_gen_cnt[i_gen] >= max_image_gen_round or current_iteration == max_iterations - 1
                        ):  # Text Search Constraint
                            idx_to_remove.append(i_gen)
                            print(f"{i_gen} has reached max_image_gen_round {max_image_gen_round}")
                            continue
                        img_to_search = non_tensor_batch["image_urls"][i_gen]
                        id_search_query_mapping[str(i_gen)] = {"type": "image", "content": img_to_search}
                        id_image_gen_cnt[i_gen] += 1  # Text Gen Constraint
                    # Need to call text search
                    elif re.search(r'<text_search>.*</text_search>$', decoded_resp_):
                        assert str(i_gen) not in id_search_query_mapping.keys()
                        if (
                            id_text_gen_cnt[i_gen] >= max_text_gen_round or current_iteration == max_iterations - 1
                        ):  # Text Search Constraint
                            idx_to_remove.append(i_gen)
                            print(f"{i_gen} has reached max_text_gen_round {max_text_gen_round}")
                            continue
                        # find last
                        text_to_search = None
                        for match in re.finditer(r'<text_search>(.*?)</text_search>', decoded_resp_):
                            text_to_search = match.group(1)
                        if text_to_search:
                            id_search_query_mapping[str(i_gen)] = {"type": "text", "content": text_to_search}
                            id_text_gen_cnt[i_gen] += 1  # Text Gen Constraint
                        else:
                            print(
                                "[Round #{current_iteration} Rollout ERROR] No text search query found!!! traj {i_gen} will be removed!!!"
                            )
                            idx_to_remove.append(i_gen)
                    # Direct Answer
                    else:
                        # remove this traj from to_generate
                        idx_to_remove.append(i_gen)
                        # NOTE: to_generate.remove(i_gen) # DO NOT .remove() in for loop

                print(
                    f"[Round #{current_iteration} Rollout Search Trigger] For THIS round, we need to conduct search for: {id_search_query_mapping} ..."
                )

                # update 'to_generate'
                for x in idx_to_remove:
                    to_generate.remove(x)

                print(
                    f"[Round #{current_iteration} Rollout END] For THIS round, We hava completed {len(idx_to_remove)} trajs ..."
                )
                print(
                    f"[Round #{current_iteration} Rollout END] For NEXT round, We hava {len(to_generate)} trajs to complete ..."
                )

                # [Call Search Tool] Conduct Search as-needed
                search_result = []

                if not self.config.search.parallel_tool_call:
                    ########################################## sequential implementation #############################################
                    for i_todo in tqdm(to_generate, desc=f"[Round #{current_iteration} Searching Progress]"):
                        tool_returned_images = []
                        assert str(i_todo) in id_search_query_mapping.keys()
                        _type = id_search_query_mapping[str(i_todo)]["type"]
                        _content = id_search_query_mapping[str(i_todo)]["content"]
                        # print(f"[Round #{current_iteration} Search START] Call search tool | Type: {_type} | Content: {_content} ...")
                        if _type == "text":
                            tool_returned_str, tool_stat = call_text_search(
                                text_query=_content,
                            )
                        elif _type == "image":
                            tool_returned_str, tool_returned_images, tool_stat = call_image_search(
                                image_url=_content,
                            )
                        else:
                            raise ValueError(f"[Round #{current_iteration} Search ERROR] Unknown Search Type: {_type}")
                        # print(f"[Round #{current_iteration} Search END] Search tool return:\n {tool_returned_str} ...")
                        search_result.append((tool_returned_str, tool_returned_images, tool_stat))
                    ########################################## sequential implementation #############################################
                else:
                    ############################################## parallel implementation #############################################
                    def tool_helper(i_todo):
                        tool_returned_images = []
                        assert str(i_todo) in id_search_query_mapping.keys()
                        _type = id_search_query_mapping[str(i_todo)]["type"]
                        _content = id_search_query_mapping[str(i_todo)]["content"]
                        thread_id = threading.current_thread().ident
                        print(
                            f"[Round #{current_iteration} Search START][Thread{thread_id}] Call search tool | Type: {_type} | Content: {_content} ..."
                        )
                        if _type == "text":
                            tool_returned_str, tool_stat = call_text_search(
                                text_query=_content,
                            )
                        elif _type == "image":
                            tool_returned_str, tool_returned_images, tool_stat = call_image_search(
                                image_url=_content,
                            )
                        else:
                            raise ValueError(
                                f"[Round #{current_iteration} Search ERROR][Thread{thread_id}] Unknown Search Type: {_type}"
                            )
                        print(
                            f"[Round #{current_iteration} Search END][Thread{thread_id}] Search tool return:\n {tool_returned_str} ..."
                        )
                        return (tool_returned_str, tool_returned_images, tool_stat)

                    search_call_futures = []
                    with ThreadPoolExecutor(self.config.search.parallel_tool_call_threads) as pool:
                        for i_todo in to_generate:
                            assert str(i_todo) in id_search_query_mapping.keys()
                            search_call_futures.append(pool.submit(tool_helper, i_todo))
                        for _ in tqdm(
                            as_completed(search_call_futures),
                            desc=f"[MT][Round #{current_iteration} Searching Progress]",
                        ):
                            pass
                    search_result = [f.result() for f in search_call_futures]
                    ############################################## parallel implementation #############################################

                # [Process Search Results]
                to_generate_ = to_generate.copy()  # make a copy since we will be modifying to_generate
                assert len(to_generate_) == len(
                    search_result
                ), f"Current Itr: {current_iteration} | len(to_generate_): {len(to_generate_)} | len(search_result): {len(search_result)}"
                for i_gen_, search_result_ in zip(to_generate_, search_result):

                    search_result_txt, search_result_img, tool_stat = search_result_

                    # init search_result_message
                    search_result_message = search_result_txt

                    # Construct Next Round Prompt
                    # Use after_image_search_prompt and after_text_search_prompt to differentiate the two cases
                    if (
                        "[Text Search Results]" in search_result_txt
                        and "[Text Search Results] There is an error encountered" not in search_result_txt
                    ):
                        # Text Search Performed and No error encountered
                        if self.user_prompt_after_text_search is not None:
                            all_context = self.tokenizer.decode(
                                vllm_inputs[i_gen_]['prompt_token_ids'], skip_special_tokens=True
                            )
                            org_query = (
                                all_context.split("Here is the image and the question:\n ")[1]
                                .split("assistant")[0]
                                .strip()
                            )
                            # text_query = all_context.split("<text_search>")[-1].split("</text_search>")[0]
                            search_result_message = (
                                "Searched results: <information>"
                                + search_result_txt
                                + "</information>\n"
                                + f"Original user's question: {org_query}\n"
                                + self.user_prompt_after_text_search
                            )
                    if (
                        "[Image Search Results]" in search_result_txt
                        and "[Image Search Results] There is an error encountered" not in search_result_txt
                    ):
                        # Image Search Performed and No error encountered
                        if self.user_prompt_after_image_search is not None:
                            all_context = self.tokenizer.decode(
                                vllm_inputs[i_gen_]['prompt_token_ids'], skip_special_tokens=True
                            )
                            org_query = (
                                all_context.split("Here is the image and the question:\n ")[1]
                                .split("assistant")[0]
                                .strip()
                            )
                            search_result_message = (
                                "Searched results: <information>"
                                + search_result_txt
                                + "</information>\n"
                                + f"Original user's question: {org_query}\n"
                                + self.user_prompt_after_image_search
                            )

                    search_result_message = (
                        "<|im_start|>user\n" + search_result_message + "<|im_end|>\n<|im_start|>assistant\n"
                    )
                    next_turn_prompt_ids = self.tokenizer.encode(search_result_message)

                    # update conversation
                    vllm_inputs[i_gen_][
                        'prompt_token_ids'
                    ] += next_turn_prompt_ids  # this might go over response length, but we will cut it later by 'max_response_length_total'
                    if search_result_img:
                        vllm_inputs[i_gen_]['multi_modal_data']['image'] += search_result_img
                        search_tool_return_images[
                            i_gen_
                        ] += search_result_img  # save images that returned by search tool
                    multi_turn_response_mask[i_gen_].append(
                        torch.zeros(len(next_turn_prompt_ids), dtype=attention_mask.dtype, device=attention_mask.device)
                    )  # USER, Mark as 0

                # update pbar
                pbar.update(worker_trajs_count - len(to_generate))

                # update iteration count
                current_iteration += 1

        # re-build response
        response = []  # B'*R, torch.Tensors with unequal lengths
        response_generation_mask = []  # B'*R, torch.Tensors with unequal lengths but align with 'response'
        # process search tool returned images
        for i_ in range(batch_size * n):  # 0~15, 4*4
            # for each traj, we skip first-round prompt_ids/attention_mask
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
            resp_mask_device = all_response_masks.device

            first_round_prompt_length = prefix_prompt_lengths[i_]
            response_after_prompt = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]

            # NOTE: [For Multi-Image] Update response_after_prompt(list of token_ids) and all_response_masks if search tool returned images
            if search_tool_return_images[i_]:
                # process PIL.Images to get 'pixel_values' and 'image_grid_thw'
                searched_image_inputs = self.processor.image_processor(
                    search_tool_return_images[i_], return_tensors='pt'
                )  # dict_keys(['pixel_values', 'image_grid_thw'])
                searched_image_grid_thw = searched_image_inputs['image_grid_thw']
                # print(f"searched_image_grid_thw shape: {searched_image_grid_thw.shape}")
                # print(f"searched_image_grid_thw: {searched_image_grid_thw}")
                if searched_image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index, image_pad_token, magic_num = 0, 151655, 654321
                    all_response_masks = all_response_masks.tolist()  # for convenient modification
                    while image_pad_token in response_after_prompt:
                        # find pos of <|image_pad|>
                        pos = response_after_prompt.index(image_pad_token)
                        replicate_count = searched_image_grid_thw[index].prod() // merge_length
                        # update response_after_prompt
                        response_after_prompt[pos : pos + 1] = [magic_num] * replicate_count
                        # update all_response_masks
                        all_response_masks[pos : pos + 1] = [0] * replicate_count
                        index += 1
                    response_after_prompt = [image_pad_token if x == magic_num else x for x in response_after_prompt]
                    all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=resp_mask_device)

            response_generation_mask.append(all_response_masks)  # at least we have single-turn conversation
            all_response = torch.tensor(response_after_prompt, device=idx.device, dtype=idx.dtype)
            response.append(all_response)
            assert (
                response[i_].shape[0] == response_generation_mask[i_].shape[0]
            ), f"shape mismatched | response[i_]: {response[i_].shape[0]} | response_generation_mask[i_]: {response_generation_mask[i_].shape[0]}"
        assert len(response) == len(
            response_generation_mask
        ), "length mismatched between response and response_generation_mask!"

        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(
            response, self.pad_token_id, dim=0
        )  # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0)  # Tensor, (B'*R, padded_length)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.response_length)
        if response.shape[1] > self.config.response_length_total:
            response = response[:, : self.config.response_length_total]
            response_generation_mask = response_generation_mask[:, : self.config.response_length_total]
        elif response.shape[1] < self.config.response_length_total:
            response = pad_sequence_to_length(response, self.config.response_length_total, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(
                response_generation_mask, self.config.response_length_total, 0
            )

        # All for 1st USER prompt
        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n)  # (B, max_prompt_length) -> (B*R, max_prompt_length)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
            # NOTE: We repeat 'multi_modal_data'
            if 'multi_modal_data' in non_tensor_batch.keys():
                repeated = []
                _index_br = 0
                for item in non_tensor_batch['multi_modal_data']:
                    for _ in range(self.config.n):
                        new_item = copy.deepcopy(item)
                        if search_tool_return_images[_index_br]:
                            new_item['image'] += search_tool_return_images[_index_br]
                        repeated.append(new_item)
                        _index_br += 1
                non_tensor_batch['multi_modal_data'] = np.array(repeated)
            # we also need to repeat 'input_prompt_generation_mask'
            input_prompt_generation_mask = _repeat_interleave(
                input_prompt_generation_mask, self.config.n
            )  # (B, max_prompt_length) -> (B*R, max_prompt_length), all 0

        seq = torch.cat([idx, response], dim=-1)  # (B*R, max_prompt_length+max_response_length_total)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_final_eos_mask(
            response_id=response, eos_token=[151645], dtype=attention_mask.dtype
        )  # HACK: for qwen, |im_end| is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        # NOTE: .contiguous() for broadcast
        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous(),
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        print(f">>> vllm_rollout_spmd Rollout Ends ...")
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _multi_query_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Multi-Query模式的核心实现
        """
        # rebuild cache
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        batch_size = prompts.batch['input_ids'].size(0)
        non_tensor_batch = prompts.non_tensor_batch

        # 存储所有样本的最终结果
        all_final_results = []

        # 对batch中的每个样本进行处理
        for batch_idx in range(batch_size):
            print(f"\n{'=' * 80}")
            print(f"Processing Sample {batch_idx + 1}/{batch_size}")
            print(f"{'=' * 80}")

            # Step 1: 提取原始query和图像
            original_query = self._extract_query_from_batch(batch_idx, prompts)
            image_inputs = self._extract_images_from_batch(batch_idx, non_tensor_batch)

            print(f"\nOriginal Query: {original_query}")

            # Step 2: 拆分query
            sub_queries = self.decompose_query(original_query, image_inputs)

            # Step 3: 对每个子query应用原MMSearch-R1流程
            sub_results = []
            for sub_idx, sub_query in enumerate(sub_queries):
                print(f"\n{'-' * 60}")
                print(f"Processing Sub-query {sub_idx + 1}/{len(sub_queries)}")
                print(f"Query: {sub_query}")
                print(f"{'-' * 60}")

                # 为子query构建DataProto
                sub_prompt = self._create_sub_query_dataproto(
                    sub_query,
                    image_inputs,
                    prompts,
                    batch_idx
                )

                # 临时禁用query拆分，调用原流程
                original_flag = self.enable_query_decomposition
                self.enable_query_decomposition = False

                try:
                    # 调用原generate_sequences
                    sub_result = self.generate_sequences(sub_prompt, **kwargs)

                    # 提取答案
                    answer, reasoning = self._extract_answer_from_result(sub_result, 0)
                    sub_results.append((sub_query, answer, reasoning))

                    print(f"Sub-answer: {answer[:150]}...")

                except Exception as e:
                    print(f"[Error] Failed to process sub-query: {e}")
                    sub_results.append((sub_query, "", f"Error: {str(e)}"))

                finally:
                    self.enable_query_decomposition = original_flag

            # Step 4: 综合答案
            print(f"\n{'=' * 60}")
            print(f"Synthesizing Final Answer")
            print(f"{'=' * 60}")

            final_response = self.synthesize_answers(original_query, sub_results, image_inputs)

            # 存储结果
            all_final_results.append({
                'batch_idx': batch_idx,
                'original_query': original_query,
                'sub_queries': sub_queries,
                'sub_results': sub_results,
                'final_response': final_response,
                'image_inputs': image_inputs
            })

        # Step 5: 构建最终的DataProto
        final_dataproto = self._build_final_dataproto(prompts, all_final_results)

        # free cache
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        print(f"\n>>> [Multi-Query Mode] Rollout Complete <<<")
        return final_dataproto

    def _extract_query_from_batch(self, batch_idx: int, prompts: DataProto) -> str:
        """从batch中提取原始query"""
        try:
            raw_prompt_ids = prompts.non_tensor_batch['raw_prompt_ids'][batch_idx]
            prompt_text = self.tokenizer.decode(raw_prompt_ids, skip_special_tokens=True)

            # 尝试多种格式提取question
            patterns = [
                r'Question:\s*(.+?)(?:\n|Image:|<\|vision_start\||<\|im_end\||$)',
                r'question:\s*(.+?)(?:\n|$)',
                r'Query:\s*(.+?)(?:\n|$)',
                r'query:\s*(.+?)(?:\n|$)',
            ]

            for pattern in patterns:
                match = re.search(pattern, prompt_text, re.IGNORECASE | re.DOTALL)
                if match:
                    query = match.group(1).strip()
                    # 清理可能的格式标记
                    query = re.sub(r'<\|.*?\|>', '', query)
                    query = query.strip()
                    if len(query) > 3:
                        return query

            # 如果都失败，尝试提取最后一个用户输入
            user_content_match = re.search(r'<\|im_start\|>user\n(.+?)(?:<\|im_end\||$)', prompt_text, re.DOTALL)
            if user_content_match:
                content = user_content_match.group(1).strip()
                # 移除Image标记
                content = re.sub(r'Image:.*', '', content, flags=re.DOTALL)
                content = content.strip()
                if len(content) > 3:
                    return content

            print(f"[Warning] Could not extract query from prompt, using truncated text")
            # 返回前200个字符作为fallback
            return prompt_text[:200].strip()

        except Exception as e:
            print(f"[Error] Failed to extract query: {e}")
            return "Unable to extract query"

    def _extract_images_from_batch(self, batch_idx: int, non_tensor_batch: dict) -> list:
        """从batch中提取图像数据"""
        images = []
        try:
            if 'multi_modal_data' in non_tensor_batch:
                modal_data = non_tensor_batch['multi_modal_data'][batch_idx]
                if isinstance(modal_data, dict) and 'image' in modal_data:
                    image_data = modal_data['image']
                    if isinstance(image_data, list):
                        images = image_data
                    else:
                        images = [image_data]
        except Exception as e:
            print(f"[Warning] Failed to extract images: {e}")

        return images

    def _create_sub_query_dataproto(self, sub_query: str, image_inputs: list,
                                    original_prompts: DataProto, batch_idx: int) -> DataProto:
        """为子query创建新的DataProto"""

        # 构建新的prompt text
        # 从原始prompt中提取system/instruction部分
        original_raw_ids = original_prompts.non_tensor_batch['raw_prompt_ids'][batch_idx]
        original_text = self.tokenizer.decode(original_raw_ids, skip_special_tokens=True)

        # 提取instruction/system prompt部分（question之前的内容）
        instruction_match = re.search(r'^(.*?)Question:', original_text, re.DOTALL | re.IGNORECASE)
        if instruction_match:
            instruction = instruction_match.group(1).strip()
        else:
            # 如果找不到，使用默认instruction
            instruction = "Answer the user's question based on the provided image."

        # 构建新prompt
        new_prompt_text = f"{instruction}\nQuestion: {sub_query}"

        # 构建messages
        messages = []
        content_items = [{"type": "text", "text": new_prompt_text}]

        # 添加图像
        if image_inputs:
            for img in image_inputs:
                content_items.append({"type": "image", "image": img})

        messages.append({"role": "user", "content": content_items})

        # 使用processor处理
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        from qwen_vl_utils import process_vision_info
        image_data, video_data = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_data,
            videos=video_data,
            padding=True,
            return_tensors="pt"
        )

        # 构建position_ids
        seq_length = inputs['input_ids'].size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs['input_ids'].device)

        # 对于Qwen2-VL，需要3D position_ids (mrope)
        if hasattr(self.config, 'model') and 'qwen2' in str(self.config.model.get('path', '')).lower():
            position_ids = position_ids.unsqueeze(0).expand(3, -1).unsqueeze(0)
        else:
            position_ids = position_ids.unsqueeze(0)

        # 构建DataProto
        new_dataproto = DataProto(
            batch=TensorDict({
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'position_ids': position_ids,
            }, batch_size=1),
            non_tensor_batch={
                'raw_prompt_ids': np.array([inputs['input_ids'][0].tolist()], dtype=object),
                'multi_modal_data': [{'image': image_data}] if image_data else [{}],
            },
            meta_info=original_prompts.meta_info.copy()
        )

        return new_dataproto

    def _extract_answer_from_result(self, result: DataProto, batch_idx: int = 0) -> tuple:
        """从rollout结果中提取答案和推理"""
        try:
            # 尝试从responses中提取
            if 'responses' in result.batch:
                response_ids = result.batch['responses'][batch_idx]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            elif 'input_ids' in result.batch:
                # 从完整的input_ids中提取response部分
                full_ids = result.batch['input_ids'][batch_idx]
                full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)

                # 尝试找到assistant的回复
                assistant_match = re.search(r'<\|im_start\|>assistant\n(.+?)(?:<\|im_end\||$)', full_text, re.DOTALL)
                if assistant_match:
                    response_text = assistant_match.group(1).strip()
                else:
                    response_text = full_text
            else:
                return "", ""

            # 提取<answer>标签内容
            answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # 如果没有标签，尝试提取最后几行
                lines = [l.strip() for l in response_text.strip().split('\n') if l.strip()]
                answer = lines[-1] if lines else response_text[:100]

            # 提取<reason>标签内容
            reason_match = re.search(r'<reason>(.*?)</reason>', response_text, re.DOTALL)
            reasoning = reason_match.group(1).strip() if reason_match else ""

            # 清理答案
            answer = answer.strip('.,;:!? \n')

            return answer, reasoning

        except Exception as e:
            print(f"[Error] Failed to extract answer: {e}")
            return "", ""

    def _build_final_dataproto(self, original_prompts: DataProto, results: list) -> DataProto:
        """构建最终的DataProto返回格式"""
        batch_size = len(results)

        # 将最终答案转换为token ids
        final_response_ids = []
        max_response_length = 0

        for result in results:
            response_text = result['final_response']
            response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            final_response_ids.append(response_ids)
            max_response_length = max(max_response_length, len(response_ids))

        # Padding responses
        padded_responses = []
        response_masks = []

        for response_ids in final_response_ids:
            pad_length = max_response_length - len(response_ids)

            # Padding
            padded = response_ids + [self.pad_token_id] * pad_length
            mask = [1] * len(response_ids) + [0] * pad_length

            padded_responses.append(torch.tensor(padded, dtype=torch.long))
            response_masks.append(torch.tensor(mask, dtype=torch.float))

        responses = torch.stack(padded_responses)
        response_attention = torch.stack(response_masks)

        # 构建完整的sequences
        prompts_input_ids = original_prompts.batch['input_ids']
        prompts_attention = original_prompts.batch['attention_mask']
        prompts_position = original_prompts.batch['position_ids']

        # 移动到同一设备
        device = prompts_input_ids.device
        responses = responses.to(device)
        response_attention = response_attention.to(device)

        # 拼接
        full_input_ids = torch.cat([prompts_input_ids, responses], dim=1)
        full_attention_mask = torch.cat([prompts_attention, response_attention], dim=1)

        # 构建response的position_ids
        response_length = responses.size(1)
        response_position_list = []

        for i in range(batch_size):
            last_pos = prompts_position[i, -1] if prompts_position.dim() == 2 else prompts_position[i, :, -1]

            if prompts_position.dim() == 3:  # Qwen2-VL mrope (B, 3, L)
                # 为每个维度创建递增的position
                resp_pos = torch.arange(1, response_length + 1, device=device, dtype=torch.long)
                resp_pos = resp_pos.unsqueeze(0).expand(3, -1)  # (3, response_length)
                # 加上last_pos
                for d in range(3):
                    resp_pos[d] = resp_pos[d] + last_pos[d]
                response_position_list.append(resp_pos)
            else:  # 普通2D position_ids (B, L)
                resp_pos = torch.arange(1, response_length + 1, device=device, dtype=torch.long)
                resp_pos = resp_pos + last_pos
                response_position_list.append(resp_pos)

        response_position_ids = torch.stack(response_position_list)
        full_position_ids = torch.cat([prompts_position, response_position_ids], dim=-1)

        # 构建multi_turn_response_mask
        # prompt部分为0，response部分为1
        multi_turn_mask = torch.cat([
            torch.zeros_like(prompts_input_ids, dtype=torch.float),
            torch.ones_like(responses, dtype=torch.float)
        ], dim=1)

        # 构建最终的TensorDict
        final_batch = TensorDict({
            'prompts': prompts_input_ids,
            'responses': responses,
            'input_ids': full_input_ids,
            'attention_mask': full_attention_mask,
            'position_ids': full_position_ids,
            'multi_turn_response_mask': multi_turn_mask,
        }, batch_size=batch_size)

        # 构建non_tensor_batch
        final_non_tensor_batch = {
            'raw_prompt_ids': original_prompts.non_tensor_batch.get('raw_prompt_ids', None),
            'multi_modal_data': original_prompts.non_tensor_batch.get('multi_modal_data', None),
        }

        return DataProto(
            batch=final_batch,
            non_tensor_batch=final_non_tensor_batch,
            meta_info=original_prompts.meta_info
        )
