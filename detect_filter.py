import cv2
import subprocess
import json
import insightface
from tqdm import tqdm
import torch
import onnxruntime as rt
from swapperfp16 import get_model
import pdb


def get_sess_options():
    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = (
        rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    )  # rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_options.execution_order = rt.ExecutionOrder.PRIORITY_BASED
    return sess_options


def get_analyser(providers):
  global get_model
  sess_options = get_sess_options()
  print(providers)
  face_analyser = insightface.app.FaceAnalysis(name='buffalo_l',allowed_modules=["recognition", "detection"], providers=providers, session_options=sess_options)
  face_analyser.prepare(ctx_id=0, det_size=(256, 256))
  return face_analyser

def find_face(frame, face_analyser):
    
    faces = face_analyser.get(frame)
    return len(faces) > 0


def process_video(input_video, output_video, provider_list):

    face_analyser = get_analyser(provider_list)

    # Open the input video
    cap = cv2.VideoCapture(input_video)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_info = []
    frame_number = 0

    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            has_face = find_face(frame, face_analyser)  

            if has_face:
                out.write(frame)
                frame_info.append(frame_number)

            frame_number += 1
            pbar.update(1)

    cap.release()
    out.release()

    return frame_info, fps


def create_audio_cut_list(frame_info, fps):
		cut_list = []
		start_time = 0

		for i, frame in enumerate(frame_info):
				if i == 0 or frame != frame_info[i - 1] + 1:
						if i > 0:
								end_time = frame_info[i - 1] / fps
								cut_list.append((start_time, end_time))
						start_time = frame / fps

		# Add the last segment
		if frame_info:
				cut_list.append((start_time, frame_info[-1] / fps))

		return cut_list


def cut_audio_and_merge(input_video, face_swapped_video, cut_list, output_video):
    filters = []
    chunk_size =  10
    for i in range(0, len(cut_list), chunk_size):
        chunk = cut_list[i : i + chunk_size]
        cut_list_str = "|".join(
            [f"between(t,{start:.3f},{end:.3f})" for start, end in chunk]
        )
        filters.append(f"[0:a]aselect='{cut_list_str}',asetpts=N/SR/TB[aud{i}];")

    filter_complex = "".join(filters) + f"concat=n={len(filters)}:v=0:a=1[aout]"

    command = [
        "ffmpeg",
        "-i",
        input_video,
        "-i",
        face_swapped_video,
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        output_video,
    ]
    subprocess.run(command, check=True)


providers = [
    (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            "gpu_mem_limit": 12884901888,
            "gpu_external_alloc": 0,
            "gpu_external_free": 0,
            "gpu_external_empty_cache": 1,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "cudnn_conv1d_pad_to_nc1d": 1,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "do_copy_in_default_stream": 1,
            "enable_cuda_graph": 0,
            "cudnn_conv_use_max_workspace": 1,
            "tunable_op_enable": 1,
            "enable_skip_layer_norm_strict_mode": 1,
            "tunable_op_tuning_enable": 1,
        },
    ),
    "CPUExecutionProvider",
]

# Main process
input_video = "D:\\awan\\iCloudDrive\\CloudData\\Settings\\Config\\Google\\UserSettings\\Mapdata\\yShatioumnosinU\\CorpAtlantic_part3.verilog"
output_video = "D:\\ini\\UserSettings\\Google\\Config\\LandingZone\\outputs\\temp.mp4"
final_output = "D:\\ini\\UserSettings\\Google\\Config\\LandingZone\\outputs\\test_final.mp4"

frame_info, fps = process_video(input_video, output_video, providers)
cut_list = create_audio_cut_list(frame_info, fps)
cut_audio_and_merge(input_video, output_video, cut_list, final_output)
