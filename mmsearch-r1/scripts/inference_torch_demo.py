import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from mmsearch_r1.utils.tools.image_search import call_image_search
from mmsearch_r1.utils.tools.text_search import call_text_search
from io import BytesIO
import pickle
import base64
import re
import os
import requests

###
### python3 mmsearch_r1/scripts/inference_torch_demo.py --model_path ${MODEL_PATH} --image ${IMAGE_URL} --question ${QUESTION}
###


MAX_PIXELS=672*672

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='The HuggingFace model repo or local path, e.g. Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--image', type=str, required=True, help='User image, e.g. URL of local path')
    parser.add_argument('--question', type=str, required=True, help='User question')
    return parser.parse_args()


def load_model_and_processor(model_path: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def generate_response(model, processor, messages):

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

def load_image_as_base64(image_source):
    try:
        if image_source.startswith("http://") or image_source.startswith("https://"):
            response = requests.get(image_source, stream=True, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
        else:
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"File not found: {image_source}")
            with open(image_source, "rb") as f:
                image_bytes = f.read()

        return base64.b64encode(image_bytes).decode("utf-8")

    except Exception as e:
        raise RuntimeError(f"Failed to load image from {image_source}: {e}")

def pil_images_to_base64(images, image_format="PNG"):
    base64_list = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format=image_format if img.format is None else img.format)
        img_bytes = buffered.getvalue()
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
        base64_list.append(base64_str)
    return base64_list

def main():

    args = parse_args()
    model_path = args.model_path
    
    # Load Image to Base64
    base64Frames_query_image = load_image_as_base64(args.image)

    # Load Model
    model, processor = load_model_and_processor(model_path)

    # Load Prompt
    with open("mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl", "rb") as f:
        round_1_prompt = pickle.load(f)
        round_1_prompt = round_1_prompt.replace("<image>", "").strip()
    with open("mmsearch_r1/prompts/after_image_search_prompt_qwenvl.pkl", "rb") as f:
        after_image_search_prompt = pickle.load(f)
    with open("mmsearch_r1/prompts/after_text_search_prompt_qwenvl.pkl", "rb") as f:
        after_text_search_prompt = pickle.load(f)

    # Init Messages
    messages = []

    # 1st-User Message
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": f"{round_1_prompt}\nQuestion: {args.question}\nImage: "},
            {"type": "image", "image": f"data:image/png;base64,{base64Frames_query_image}", "max_pixels": MAX_PIXELS},
        ]
    })

    print(f">>> Question: {args.question}")

    # 1st-Round Response
    assistant_first_reply = generate_response(model, processor, messages)
    print(f">>> 1st Response: {assistant_first_reply}")

    # Call Image Search at 1st Round Response
    if re.search(r'<search><img></search>$', assistant_first_reply):
        # 1st-Assistant Message
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_first_reply}
            ]
        })

        # Call Image Search
        tool_returned_str, tool_returned_images, tool_stat = call_image_search(image_url=args.image)
        
        # ⚠️fake webpage_title_list, it should be parsed from `tool_returned_str`
        img_tool_returned_web_title_list = [f"Webpage Title {i+1}" for i in range(len(tool_returned_images))]

        base64Frames_image_search_results_list = pil_images_to_base64(tool_returned_images)
        # 2nd-User Response
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Search Results: <information> [Image search results] The result of the image search consists of web page information related to the image from the user's original question. Each result includes the main image from the web page and its title, ranked in descending order of search relevance, as demonstrated below:"},
                *[
                    elem
                    for image_b64, title in zip(base64Frames_image_search_results_list, img_tool_returned_web_title_list)
                    for elem in [
                        {"type": "image", "image": f"data:image/png;base64,{image_b64}", "max_pixels": MAX_PIXELS},
                        {"type": "text", "text": f"Title: {title}"}
                    ]
                ],
                {"type": "text", "text": f"</information> Original user's question: {args.question}\n{after_image_search_prompt}"},
            ],
        })

        # 2nd-Round Response Generation
        assistant_second_reply = generate_response(model, processor, messages)
        print(f">>> 2nd Response: {assistant_second_reply}")
        
        if re.search(r'<text_search>.*</text_search>$', assistant_second_reply):
            text_to_search = None
            for match in re.finditer(r'<text_search>(.*?)</text_search>', assistant_second_reply):
                text_to_search = match.group(1)
            txt_tool_returned_str, _ = call_text_search(text_to_search)
            messages.append(
                {"role": "user", "content": [
                {"type": "text", "text": "Search Results: <information>" + txt_tool_returned_str + f"</information> Original question: {args.question}\n{after_text_search_prompt}"}
            ]})
            # 3rd-Round Response Generation
            assistant_third_reply = generate_response(model, processor, messages)
            print(f">>> 3rd Response: {assistant_third_reply}")

    # Call Text Search at 1st Round Response
    elif re.search(r'<text_search>.*</text_search>$', assistant_first_reply):
        text_to_search = None
        for match in re.finditer(r'<text_search>(.*?)</text_search>', assistant_first_reply):
            text_to_search = match.group(1)
        txt_tool_returned_str, _ = call_text_search(text_to_search)
        messages.append(
            {"role": "user", "content": [
            {"type": "text", "text": "Search Results: <information>" + txt_tool_returned_str + f"</information> Original question: {args.question}\n{after_text_search_prompt}"}
        ]})
        assistant_second_reply = generate_response(model, processor, messages)
        print(f">>> 2nd Response: {assistant_second_reply}")

    # Give the final answer
    else:
        pass


if __name__ == "__main__":
    main()