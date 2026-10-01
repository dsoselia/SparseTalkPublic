import argparse
import os
import json
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import time

torch.set_warn_always(False)
import numpy as np
import gaussian_utils as gaussian_utils
from preselected_gaussians import blind_audit, valid_completed_scene, write_json_atomic
from gaussian_feature_loader import prepare_single_payload

parser = argparse.ArgumentParser(description='LLaVA inference from Gaussian features.')
parser.add_argument('--scene_dir', default='.', help='Gaussian scene root')
parser.add_argument('--model_base', default='lmms-lab/llava-onevision-qwen2-7b-ov', help='base model path or ID')
parser.add_argument('--model_path', default='', help='optional adapter path')
parser.add_argument('--prompt', default='', help='single question')
parser.add_argument('--json_path', default='', help='questions JSON')
parser.add_argument('--gt_feature_path', default='.')
parser.add_argument('--scene_name', default='.', help='scene ID or all')
parser.add_argument('--json_save_path', default='', help='prediction output directory')
parser.add_argument('--language_feats_dir', default='', help='scene feature subdirectory')
parser.add_argument('--mode', default='scanqa', help='annotation format')
parser.add_argument('--ntokens', default=44, type=int, help='maximum 729-row blocks for entropy selection')
parser.add_argument('--max-new-tokens', default=1024, type=int)
parser.add_argument('--device_map', default='auto', help='device map passed to the LLaVA loader')
parser.add_argument('--gaussian-selection', choices=['entropy', 'preselected', 'text_only'], default='entropy')
parser.add_argument('--selection-audit-dir', default='')
parser.add_argument('--inference-timing-dir', default='')
parser.add_argument('--scanrefer', default=False, type=bool)






args = parser.parse_args()

scene_dir = args.scene_dir
model_base = args.model_base
model_path = args.model_path

if len(model_path) == 0:
    model_path = model_base
    model_base = None

model_name = get_model_name_from_path(model_base or model_path)
print(model_name)
device = "cuda"
device_map = args.device_map

all_scenes = sorted(os.listdir(scene_dir))



def init_model(pretrained, model_name, finetune):
    tokenizer, model, image_processor, max_length = load_pretrained_model(finetune, pretrained, model_name, device_map=device_map, attn_implementation=None)
    model.eval()
    model.tie_weights()
    return tokenizer, model, image_processor, max_length

def get_input_ids(conv, tokenizer):
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    return input_ids


def get_conv_mode(model_name):
    conv_mode = 'qwen_1_5'
    return conv_mode

def init_conv(conv_template, inp):
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    return conv

def generate_response(model, input_ids, image_features, image_sizes, stop_str, tokenizer):
    with torch.inference_mode():
        
        output_ids = model.generate_with_features(input_ids, None, image_features, \
                            image_sizes, do_sample=False, temperature=0.9,
                            max_new_tokens=args.max_new_tokens, use_cache=False)
    
    outputs = tokenizer.decode(output_ids[0]).strip()[:-len(stop_str)]
    return outputs

def get_data_by_scene(data, scene, mode='scanqa'):
    scene_q = [d for d in data if d['scene_id']== scene]
    scene_questions = [s['question'] for s in scene_q]
    scene_answers = {ind: s['answers'] for ind, s in enumerate(scene_q)}
    if mode == 'sqa':
        scene_situations = [s['situation'] for s in scene_q]
        scene_questions = [situation + ' ' + question  for (situation, question) in zip(scene_situations, scene_questions)]


    return scene_q, scene_questions, scene_answers

def postprocess(outputs):
    outputs = outputs.replace(',', '')
    return outputs

def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data

def get_features(scene, ov=False):
    if args.gaussian_selection == 'text_only':
        return None, blind_audit(scene)
    scene_dir = os.path.join(args.scene_dir, scene, args.language_feats_dir)

    
    torch.manual_seed(0)
    all_feature_paths = [f for f in sorted(os.listdir(scene_dir)) if f.endswith('.pt')]


    if len(all_feature_paths) == 1 or len(all_feature_paths) == 2:
        feature_path = os.path.join(scene_dir, all_feature_paths[0])
        image_features1 = torch.load(feature_path, map_location='cpu')
        return prepare_single_payload(
            image_features1, scene, args.ntokens, args.gaussian_selection
        )
    else:
        all_image_features = []
        for feature_path in all_feature_paths:
            feature_path = os.path.join(scene_dir, feature_path)
            image_feature = torch.load(feature_path, map_location='cpu')

            all_image_features.append(image_feature)
        image_features = torch.stack(all_image_features, dim=0)
        print(image_features.shape)
        ind = torch.randperm(len(image_features))[:44]

        image_features = image_features[ind]
        print(all_feature_paths[0])
    return image_features, None



def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run LLaVA inference on CPU")
    torch.manual_seed(0)

    model_load_start = time.perf_counter()
    tokenizer, model, image_processor, max_length = init_model(model_base, model_name, model_path)
    model_load_seconds = time.perf_counter() - model_load_start
    if args.inference_timing_dir:
        model_attempts_path = os.path.join(args.inference_timing_dir, 'model_load_attempts.json')
        try:
            with open(model_attempts_path, 'r') as f:
                model_attempts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            model_attempts = []
        model_attempts.append({'elapsed_seconds': model_load_seconds, 'completed_at': time.time()})
        write_json_atomic(model_attempts_path, model_attempts)
    conv_template = get_conv_mode(model_name)


    if len(args.json_save_path) > 0:
        save_data = []
    
    gt_data = []
    image_sizes = [365, 494]

    if args.scanrefer:
        pass


    if len(args.json_path) > 0:
        path = args.json_path
        mode = args.mode

        with open(path, 'r') as f:
            conv_data = json.load(f)

        if args.scene_name == 'all':
            all_scenes = set([d['scene_id'] for d in conv_data])
            all_scenes = sorted(list(all_scenes))
        else:
            all_scenes = [args.scene_name]

        print(all_scenes)

        for scene_name in all_scenes:
            print(scene_name)

            save_data = []
            gt_data = []
            scene_q, scene_questions, scene_answers = get_data_by_scene(conv_data, scene_name, mode=mode)
            if args.gaussian_selection in {'preselected', 'text_only', 'entropy'}:
                if not args.selection_audit_dir:
                    raise ValueError('--selection-audit-dir is required in preselected mode')
                pred_path = os.path.join(args.json_save_path, scene_name + '.json')
                gt_path = os.path.join(args.json_save_path, scene_name + '_gt.json')
                audit_path = os.path.join(args.selection_audit_dir, scene_name + '.json')
                timing_path = (
                    os.path.join(args.inference_timing_dir, scene_name + '.json')
                    if args.inference_timing_dir else None
                )
                if valid_completed_scene(pred_path, gt_path, audit_path, scene_q, timing_path):
                    print('Skipping validated completed scene:', scene_name)
                    continue
            scene_started = time.perf_counter()
            image_features1, selection_audit = get_features(scene_name)
            image_features = image_features1
            response = {}

            for sq_id, sq in enumerate(scene_questions):

                image_count = 0 if image_features is None else image_features.shape[0]
                inp = (DEFAULT_IMAGE_TOKEN * image_count) + "\n" + sq


                conv = init_conv(conv_template, inp)
                input_ids = get_input_ids(conv, tokenizer)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2


                model_features = None if image_features is None else image_features.half()
                outputs = generate_response(model, input_ids, model_features, image_sizes, stop_str, tokenizer)
                outputs = postprocess(outputs)

                response[sq_id] = [outputs]
                print('Q: ', sq)
                print('A: ', outputs)
                print('Ref: ', scene_answers[sq_id])
                print('------------')
                if len(args.json_save_path) > 0:
                    question_id = scene_q[sq_id].get('question_id', f'{scene_name}_{sq_id}')
                    save_data.append({'question_id': question_id, 'scene_id': scene_name, 'question': sq, 'text': outputs})
                    gt_data.append({'question_id': question_id, 'scene_id': scene_name, 'question': sq, 'text': scene_answers[sq_id]})


                torch.cuda.empty_cache()
            if len(args.json_save_path) > 0:
                os.makedirs(args.json_save_path, exist_ok=True)
                write_json_atomic(os.path.join(args.json_save_path, scene_name + '.json'), save_data)
                write_json_atomic(os.path.join(args.json_save_path, scene_name + '_gt.json'), gt_data)
                if args.gaussian_selection in {'preselected', 'text_only', 'entropy'}:
                    write_json_atomic(
                        os.path.join(args.selection_audit_dir, scene_name + '.json'),
                        selection_audit,
                    )
                    if not args.inference_timing_dir:
                        raise ValueError('--inference-timing-dir is required in preselected mode')
                    write_json_atomic(
                        os.path.join(args.inference_timing_dir, scene_name + '.json'),
                        {
                            'scene_id': scene_name,
                            'question_count': len(scene_questions),
                            'elapsed_seconds': time.perf_counter() - scene_started,
                        },
                    )
        

    else:
        image_features, selection_audit = get_features(args.scene_name)


        image_count = 0 if image_features is None else image_features.shape[0]
        inp = (DEFAULT_IMAGE_TOKEN * image_count) + "\n" + args.prompt
        conv = init_conv(conv_template, inp)
        input_ids = get_input_ids(conv, tokenizer)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        model_features = None if image_features is None else image_features.half()
        outputs = generate_response(model, input_ids, model_features, image_sizes, stop_str, tokenizer)
        print(outputs)

main()
