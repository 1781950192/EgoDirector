import base64
import sys
import os
sys.path.append(os.path.dirname(__file__))
from demo import run


# 工具函数：将图片转换为 Base64 格式
def image_to_base64(file_path):
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None

def vis_hos(images):
    task = 'active'  # 或 'active'
    out_dir = 'output/box'  # 当前目录
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if task == 'hos':
        pointrend_cfg = "VISOR_HOS/configs/hos/hos_pointrend_rcnn_R_50_FPN_1x.yaml"
        epick_model = f'VISOR_HOS/checkpoints/model_final_hos.pth'
        run(task, images, out_dir, pointrend_cfg, epick_model, use_postprocess=False)
    elif task == 'active':
        pointrend_cfg = "VISOR_HOS/configs/active/active_pointrend_rcnn_R_50_FPN_1x.yaml"
        epick_model = f'VISOR_HOS/checkpoints/model_final_active.pth'
        run(task, images, out_dir, pointrend_cfg, epick_model, use_postprocess=False)
    base64_images = []
    for image in images:
        image = image.split("/")[-1]
        image = image_to_base64(os.path.join(out_dir,image))
        base64_images.append(image)
    return base64_images


if __name__ == '__main__':
    vis_hos(['/home/will/Mycode/MCP_action/output/visor/obb_P01_01_frame_0000000298.jpg'])