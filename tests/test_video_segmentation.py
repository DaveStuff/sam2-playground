import pytest
from typing import Dict

from test_config import *
import numpy as np
from modules.paths import *
from modules.constants import *
from modules.sam_inference import SamInference


@pytest.mark.skipif(
    not is_cuda_available(),
    reason="Skipping because this test only works in GPU"
)
@pytest.mark.parametrize(
    "model_name,video_path,points,labels,box",
    [
        (TEST_MODEL, TEST_VIDEO_PATH, TEST_POINTS, TEST_LABELS, TEST_BOX)
    ]
)
def test_video_segmentation(
    model_name: str,
    video_path: str,
    points: np.ndarray,
    labels: np.ndarray,
    box: np.ndarray
):
    download_test_files()

    inferencer = SamInference()
    print("Device:", inferencer.device)
    inferencer.init_video_inference_state(
        vid_input=video_path,
        model_type=model_name,
    )

    inferencer.add_prediction_to_frame(
        frame_idx=0,
        obj_id=0,
        points=TEST_POINTS,
        labels=TEST_LABELS,
    )

    inferencer.add_prediction_to_frame(
        frame_idx=1,
        obj_id=1,
        box=TEST_BOX,
    )

    video_segments = inferencer.propagate_in_video()

    assert video_segments and isinstance(video_segments, Dict)


@pytest.mark.skipif(
    not is_cuda_available(),
    reason="Skipping because this test only works in GPU"
)
@pytest.mark.parametrize(
<<<<<<< HEAD
    "model_name,video_path,gradio_prompt",
    [
        (TEST_MODEL, TEST_VIDEO_PATH, TEST_GRADIO_PROMPT_BOX)
=======
    "model_name,video_path,filter_mode,gradio_prompt",
    [
        (TEST_MODEL, TEST_VIDEO_PATH, COLOR_FILTER, TEST_GRADIO_PROMPT_BOX),
        (TEST_MODEL, TEST_VIDEO_PATH, PIXELIZE_FILTER, TEST_GRADIO_PROMPT_BOX),
        (TEST_MODEL, TEST_VIDEO_PATH, TRANSPARENT_COLOR_FILTER, TEST_GRADIO_PROMPT_BOX),
>>>>>>> de28ecaff3782f8754a9ca331be603b2f5bc1b92
    ]
)
def test_filtered_video_creation_pipeline(
    model_name: str,
    video_path: str,
<<<<<<< HEAD
=======
    filter_mode: str,
>>>>>>> de28ecaff3782f8754a9ca331be603b2f5bc1b92
    gradio_prompt: np.ndarray,
):
    download_test_files()

    inferencer = SamInference()
    print("Device:", inferencer.device)
    inferencer.init_video_inference_state(
        vid_input=video_path,
        model_type=model_name,
    )
    prompt_data = {
        "points": gradio_prompt
    }

    out_path, out_path = inferencer.create_filtered_video(
        image_prompt_input_data=prompt_data,
<<<<<<< HEAD
        filter_mode=COLOR_FILTER,
        frame_idx=0,
        color_hex=DEFAULT_COLOR,
=======
        filter_mode=filter_mode,
        frame_idx=0,
        color_hex=DEFAULT_COLOR,
        pixel_size=DEFAULT_PIXEL_SIZE,
>>>>>>> de28ecaff3782f8754a9ca331be603b2f5bc1b92
        invert_mask=True
    )

    assert os.path.exists(out_path)
