# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string


# adapted from search-r1
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    '''
    prediction: string
    golden_answers: list or string, support multi candidate answers
    '''
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    exactly_match = False
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            exactly_match = True
            break
    return exactly_match


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(prediction):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, prediction, re.DOTALL)
    matches = list(match)

    if not matches:
        return None
    else:
        return matches[-1].group(1).strip()


def is_valid_direct_answer(response, direct_answer_format) -> bool:
    """
    Check Direct Answer: <reason>...</reason><answer>...</answer>
      1) Structure Matching
      2) Pattern Count: <reason>...</reason>, <answer>...</answer>
      3) No any search actions included
    """
    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). <search><img></search> or <text_search> is not allowed!
    if '<search><img></search>' in response:
        return False
    if '<text_search>' in response or '</text_search>' in response:
        return False
    return True


def is_valid_image_search(response, call_image_search_format) -> bool:
    """
    Check Image Search: <reason>...</reason>...<search><img></search>
      1) Structure Matching
      2) Pattern Count: <reason>...</reason>
      3) Pattern Count: <search><img></search>
      4) No <answer> or </answer> or <text_search> or </text_search> included
    """
    pattern = call_image_search_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <reason> Count
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    # 3). <search><img></search> Count
    if response.count('<search><img></search>') != 1:
        return False
    # 4). <answer> or <text_search> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    if '<text_search>' in response or '</text_search>' in response:
        return False
    return True


def is_valid_text_search(response, call_text_search_format) -> bool:
    """
    Check Text Search: <reason>...</reason>...<text_search>...</text_search>
      1) Structure Matching
      2) Pattern Count: <reason>...</reason> 
      3) Pattern Count: <text_search>...</text_search> 
      4) No <answer> or </answer> or <search><img></search> included
    """
    pattern = call_text_search_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <reason> Count
    if response.count('<reason>') != 1 or response.count('</reason>') != 1:
        return False
    # 3). <text_search> and </text_search> Count
    if response.count('<text_search>') != 1 or response.count('</text_search>') != 1:
        return False
    # 4). <answer> or <search><img></search> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    if '<search><img></search>' in response:
        return False
    return True


def format_reward(input_string: list):
    """
    Check if the model's response follows the required formats and return a reward.
    [1-turn]:
        - Direct Answer
    [2-turn]:
        - Call Image Search + Answer
        - Call Text Search + Answer
    [3-turn]:
        - Call Image Search + Call Text Search + Answer
    Args:
    - input_string (list): A list of responses, currently, max length of `input_string` is 3 (3-turn).
    Returns:
    - format_score: float, 1.0 for right format, 0.0 for wrong
    - search_count: int, times of search tools called
    """
    conv_rounds = len(input_string)
    format_score, search_count = 0, 0
    # All allowed formats
    direct_answer_format = r'^<reason>.*</reason>.*<answer>.*</answer>$'
    call_image_search_format = r'^<reason>.*</reason>.*<search><img></search>$'
    call_text_search_format = r'^<reason>.*</reason>.*<text_search>.*</text_search>$'
    # HACK/FIXME: We need more flexible judge in the future
    # 1-turn
    if conv_rounds == 1:
        response_1 = input_string[0].strip()
        if (
            ("<search><img></search>" in response_1)
            or ("<text_search>" in response_1 and "</text_search>" in response_1)
        ):
            search_count += 1
        # Direct Answer
        if is_valid_direct_answer(response_1, direct_answer_format):
            format_score = 1
    # 2-turn
    elif conv_rounds == 2:
        response_1, response_2 = input_string[0].strip(), input_string[1].strip()
        if (
            ("<search><img></search>" in response_1)
            or ("<text_search>" in response_1 and "</text_search>" in response_1)
        ):
            search_count += 1
        # Call Image Search + Answer
        if is_valid_image_search(response_1, call_image_search_format) and is_valid_direct_answer(response_2, direct_answer_format):
            format_score = 1
        # Call Text Search + Answer
        elif is_valid_text_search(response_1, call_text_search_format) and is_valid_direct_answer(
            response_2, direct_answer_format
        ):
            format_score = 1
    # 3-turn
    elif conv_rounds == 3:
        response_1, response_2, response_3 = input_string[0].strip(), input_string[1].strip(), input_string[2].strip()
        if (
            ("<search><img></search>" in response_1)
            or ("<text_search>" in response_1 and "</text_search>" in response_1)
        ):
            search_count += 1
        if (
            ("<search><img></search>" in response_2)
            or ("<text_search>" in response_2 and "</text_search>" in response_2)
        ):
            search_count += 1
        # Call Image Search + Call Text Search + Answer
        if (
            is_valid_image_search(response_1, call_image_search_format)
            and is_valid_text_search(response_2, call_text_search_format)
            and is_valid_direct_answer(response_3, direct_answer_format)
        ):
            format_score = 1
    else:
        raise ValueError(f"[Error Occured] Number of responses is {conv_rounds}, which is not supported currently!")

    return format_score, search_count


def compute_score(prediction: list, ground_truth: list, extra_info=None):
    # Exactly Match Scorer
    search_penalty, format_penalty = 0.1, 0.1
    reward_mode = 'EM'
    if extra_info is not None and 'search_penalty' in extra_info:
        search_penalty = extra_info.get('search_penalty', 0.1)
    if extra_info is not None and 'format_penalty' in extra_info:
        format_penalty = extra_info.get('format_penalty', 0.1)
    if extra_info is not None and 'reward_mode' in extra_info:
        reward_mode = extra_info.get('reward_mode', 'EM')
        assert reward_mode in ['EM', 'SubEM'], f'reward mode {reward_mode} passed in extra_info but not recognized'

    # Extract Answer
    assert len(prediction) > 0, "[Error Occured] Model Responses are empty!"
    answer = extract_solution(prediction=prediction[-1])

    score = 0
    # Correctness Check: EM/SubEM
    if answer is not None:
        if reward_mode == "EM" and em_check(answer, ground_truth):
            score = 1
        elif reward_mode == 'SubEM' and subem_check(answer, ground_truth):
            score = 1

    # Format Check
    format_score, search_count = format_reward(prediction)

    # Search Penalty, 0.99 is added here because we only want to punish correct answers
    if search_count > 0 and score > 0.99:
        use_search_count_penalty = extra_info.get('use_search_count_penalty', False)
        if use_search_count_penalty:
            # penalty w/ search count
            for _ in range(search_count):
                score *= 1 - search_penalty
        else:
            # penalty w/o search count
            score *= 1 - search_penalty  # no penalty when not correct

    # Weighted Score: (1-FP) * Score + FP * Format_Score
    return (1 - format_penalty) * score + format_penalty * format_score
