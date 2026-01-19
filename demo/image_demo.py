# Copyright (c) OpenMMLab. All rights reserved.
"""Image Demo.

This script adopts a new infenence class, currently supports image path,
np.array and folder input formats, and will support video and webcam
in the future.

Example:
    Save visualizations and predictions results::

        python demo/image_demo.py demo/demo.jpg rtmdet-s

        python demo/image_demo.py demo/demo.jpg \
        configs/rtmdet/rtmdet_s_8xb32-300e_coco.py \
        --weights rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 --texts bench

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 --texts 'bench . car .'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365
        --texts 'bench . car .' -c

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts 'There are a lot of cars here.'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts '$: coco'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts '$: lvis' --pred-score-thr 0.7 \
        --palette random --chunked-size 80

        python demo/image_demo.py demo/demo.jpg \
        grounding_dino_swin-t_pretrain_obj365_goldg_cap4m \
        --texts '$: lvis' --pred-score-thr 0.4 \
        --palette random --chunked-size 80

        python demo/image_demo.py demo/demo.jpg \
        grounding_dino_swin-t_pretrain_obj365_goldg_cap4m \
        --texts "a red car in the upper right corner" \
        --tokens-positive -1

    Visualize prediction results::

        python demo/image_demo.py demo/demo.jpg rtmdet-ins-s --show

        python demo/image_demo.py demo/demo.jpg rtmdet-ins_s_8xb32-300e_coco \
        --show
"""

import ast
import os
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes


def batch_detection_with_config():
    """
    批量检测函数，所有参数在代码中配置
    """

    # ================================
    # 配置参数 - 在这里修改你的设置
    # ================================

    # 输入路径 - 可以是单张图片或文件夹
    inputs = '/data/home/qr/mmdetection-main/data/HRSSD/val/'  # 或者 'path/to/single_image.jpg'

    # 模型配置 - 三种方式选择一种：
    # 方式1: 使用配置文件和权重文件
    # model = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
    model = 'configs/deformable_detr/deformable-detr-refine_r50_16xb2-50e_HR.py'
    weights = 'checkpoints/hr_de.pth'

    # 方式2: 直接使用权重文件（会自动找配置）
    # model = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    # weights = None

    # 方式3: 使用预定义模型名称
    # model = 'rtmdet-s'
    # weights = None

    # 输出设置
    out_dir = './out_HR/'  # 输出文件夹

    # 推理设置
    device = 'cuda:0'  # 设备: 'cuda:0', 'cuda:1', 'cpu' 等
    pred_score_thr = 0.3  # 置信度阈值
    batch_size = 1  # 批处理大小

    # 显示和保存设置
    show = False  # 是否在弹窗中显示结果
    no_save_vis = False  # 是否不保存可视化结果
    no_save_pred = False  # 是否不保存预测JSON结果
    print_result = True  # 是否打印结果

    # 可视化设置
    palette = 'coco'  # 颜色方案: 'coco', 'voc', 'citys', 'random', 'none'

    # 文本提示设置（仅用于支持文本提示的模型，如GLIP, Grounding DINO）
    texts = None  # 例如: "bench . car ." 或 "$: coco"
    custom_entities = False  # 是否自定义实体名称
    chunked_size = -1  # 分块大小
    tokens_positive = None  # token位置

    # ================================
    # 开始处理
    # ================================

    print("=" * 60)
    print("MMDetection 批量检测开始")
    print("=" * 60)
    print(f"输入路径: {inputs}")
    print(f"模型配置: {model}")
    print(f"权重文件: {weights}")
    print(f"输出目录: {out_dir}")
    print(f"设备: {device}")
    print(f"置信度阈值: {pred_score_thr}")
    print("=" * 60)

    # 检查输入路径
    if not os.path.exists(inputs):
        print(f"错误: 输入路径 {inputs} 不存在!")
        return

    # 处理权重文件路径
    if model and model.endswith('.pth'):
        print_log('检测到权重文件，自动分配到 weights 参数')
        weights = model
        model = None

    # 处理文本提示
    if texts is not None:
        if texts.startswith('$:'):
            dataset_name = texts[3:].strip()
            class_names = get_classes(dataset_name)
            texts = [tuple(class_names)]

    # 处理tokens_positive
    if tokens_positive is not None:
        tokens_positive = ast.literal_eval(tokens_positive)

    # 处理输出目录
    if no_save_vis and no_save_pred:
        out_dir = ''
    elif out_dir:
        os.makedirs(out_dir, exist_ok=True)
        print(f"输出目录已创建: {out_dir}")

    # 初始化推理器
    print("正在初始化推理器...")
    try:
        inferencer = DetInferencer(
            model=model,
            weights=weights,
            device=device,
            palette=palette
        )
        print("推理器初始化成功!")
    except Exception as e:
        print(f"推理器初始化失败: {e}")
        return

    # 设置分块大小
    if hasattr(inferencer.model, 'test_cfg'):
        inferencer.model.test_cfg.chunked_size = chunked_size

    # 执行推理
    print("开始批量推理...")
    try:
        inferencer(
            inputs=inputs,
            out_dir=out_dir,
            texts=texts,
            pred_score_thr=pred_score_thr,
            batch_size=batch_size,
            show=show,
            no_save_vis=no_save_vis,
            no_save_pred=no_save_pred,
            print_result=print_result,
            custom_entities=custom_entities
        )

        if out_dir != '' and not (no_save_vis and no_save_pred):
            print_log(f'结果已保存到: {out_dir}')

    except Exception as e:
        print(f"推理过程出错: {e}")
        return

    print("=" * 60)
    print("批量检测完成!")
    print("=" * 60)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='/data/home/qr/mmdetection-main/outputimages/',
        help='Output directory of images or prediction results.')
    # Once you input a format similar to $: xxx, it indicates that
    # the prompt is based on the dataset class name.
    # support $: coco, $: voc, $: cityscapes, $: lvis, $: imagenet_det.
    # detail to `mmdet/evaluation/functional/class_names.py`
    parser.add_argument(
        '--texts', help='text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument(
        '--device', default='cuda:6', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection vis results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection json results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    # only for GLIP and Grounding DINO
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    parser.add_argument(
        '--chunked-size',
        '-s',
        type=int,
        default=-1,
        help='If the number of categories is very large, '
        'you can specify this parameter to truncate multiple predictions.')
    # only for Grounding DINO
    parser.add_argument(
        '--tokens-positive',
        '-p',
        type=str,
        help='Used to specify which locations in the input text are of '
        'interest to the user. -1 indicates that no area is of interest, '
        'None indicates ignoring this parameter. '
        'The two-dimensional array represents the start and end positions.')

    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    if call_args['texts'] is not None:
        if call_args['texts'].startswith('$:'):
            dataset_name = call_args['texts'][3:].strip()
            class_names = get_classes(dataset_name)
            call_args['texts'] = [tuple(class_names)]

    if call_args['tokens_positive'] is not None:
        call_args['tokens_positive'] = ast.literal_eval(
            call_args['tokens_positive'])

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    """
    主函数 - 你可以选择使用配置文件方式或命令行方式
    """
    import sys

    # 如果没有命令行参数，使用配置文件方式
    if len(sys.argv) == 1:
        print("使用代码配置模式...")
        batch_detection_with_config()
    else:
        print("使用命令行参数模式...")
        # 使用原始的命令行参数方式
        init_args, call_args = parse_args()
        inferencer = DetInferencer(**init_args)

        chunked_size = call_args.pop('chunked_size')
        inferencer.model.test_cfg.chunked_size = chunked_size

        inferencer(**call_args)

        if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                               and call_args['no_save_pred']):
            print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()
