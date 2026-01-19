# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
"""Image Demo with Anomaly Detection.

增加了异常检测功能，可以输出可能存在漏检或检测错误的图片名字
"""

import ast
import os
import json
from argparse import ArgumentParser
from pathlib import Path

from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes


class AnomalyDetector:
    """异常检测器类"""

    def __init__(self,
                 enable=True,
                 min_detections=0,
                 max_detections=999,
                 min_avg_score=0.0,
                 output_file='anomaly_images.txt'):
        """
        Args:
            enable: 是否启用异常检测
            min_detections: 最少检测数量阈值（低于此值视为漏检）
            max_detections: 最多检测数量阈值（高于此值视为过度检测）
            min_avg_score: 最低平均置信度阈值（低于此值视为检测不可靠）
            output_file: 异常图片列表输出文件
        """
        self.enable = enable
        self.min_detections = min_detections
        self.max_detections = max_detections
        self.min_avg_score = min_avg_score
        self.output_file = output_file
        self.anomaly_images = []

    def check_result(self, image_path, predictions):
        """
        检查单张图片的检测结果是否异常

        Args:
            image_path: 图片路径
            predictions: 检测结果（可以是字典或列表）

        Returns:
            bool: 是否异常
            str: 异常原因
        """
        if not self.enable:
            return False, ""

        image_name = os.path.basename(image_path)

        # 解析检测结果 - 适配多种返回格式
        pred_list = []
        scores = []

        # 格式1: {'predictions': [...]}
        if isinstance(predictions, dict) and 'predictions' in predictions:
            pred_list = predictions['predictions']
            if pred_list and isinstance(pred_list[0], dict):
                scores = [p.get('score', p.get('scores', 0)) for p in pred_list]

        # 格式2: {'bboxes': [...], 'scores': [...]}
        elif isinstance(predictions, dict):
            if 'bboxes' in predictions:
                pred_list = predictions['bboxes']
            if 'scores' in predictions:
                scores = predictions['scores']
            elif 'score' in predictions:
                scores = predictions['score']

        # 格式3: 直接是列表
        elif isinstance(predictions, list):
            pred_list = predictions
            if pred_list and isinstance(pred_list[0], dict):
                scores = [p.get('score', p.get('scores', 0)) for p in pred_list]

        # 确保 scores 是列表
        if not isinstance(scores, list):
            if hasattr(scores, 'tolist'):  # numpy array
                scores = scores.tolist()
            else:
                scores = []

        num_detections = len(pred_list) if pred_list else 0

        # 调试信息
        print(f"  检测到 {num_detections} 个目标", end="")
        if scores:
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f", 平均分: {avg_score:.3f}")
        else:
            print()

        # 规则1: 检查是否无检测结果（可能漏检）
        if num_detections == 0:
            return True, f"无检测结果（完全漏检）"

        # 规则2: 检查检测数量是否过少（可能漏检）
        if num_detections < self.min_detections:
            return True, f"检测数量过少 ({num_detections} < {self.min_detections})"

        # 规则3: 检查检测数量是否过多（可能误检）
        if num_detections > self.max_detections:
            return True, f"检测数量过多 ({num_detections} > {self.max_detections})"

        # 规则4: 检查平均置信度是否过低（可能检测不可靠）
        if num_detections > 0 and scores:
            avg_score = sum(scores) / len(scores)
            if avg_score < self.min_avg_score:
                return True, f"平均置信度过低 ({avg_score:.3f} < {self.min_avg_score})"

        return False, ""

    def add_anomaly(self, image_path, reason):
        """添加异常图片记录"""
        image_name = os.path.basename(image_path)
        self.anomaly_images.append({
            'image': image_name,
            'path': image_path,
            'reason': reason
        })
        print_log(f"⚠️  异常图片: {image_name} - {reason}", logger='current')

    def save_anomaly_list(self, output_dir):
        """保存异常图片列表"""
        if not self.anomaly_images:
            print_log("✅ 未发现异常图片", logger='current')
            return

        output_path = os.path.join(output_dir, self.output_file)

        # 保存为文本文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"异常图片列表 (共 {len(self.anomaly_images)} 张)\n")
            f.write("=" * 80 + "\n\n")
            for item in self.anomaly_images:
                f.write(f"图片: {item['image']}\n")
                f.write(f"路径: {item['path']}\n")
                f.write(f"原因: {item['reason']}\n")
                f.write("-" * 80 + "\n")

        # 同时保存为JSON格式
        json_path = os.path.join(output_dir, self.output_file.replace('.txt', '.json'))
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.anomaly_images, f, ensure_ascii=False, indent=2)

        print_log(f"\n⚠️  发现 {len(self.anomaly_images)} 张异常图片", logger='current')
        print_log(f"异常列表已保存到: {output_path}", logger='current')
        print_log(f"JSON格式已保存到: {json_path}", logger='current')

        # 打印异常统计
        reasons = {}
        for item in self.anomaly_images:
            reason = item['reason'].split('(')[0].strip()
            reasons[reason] = reasons.get(reason, 0) + 1

        print_log("\n异常类型统计:", logger='current')
        for reason, count in reasons.items():
            print_log(f"  - {reason}: {count} 张", logger='current')


def batch_detection_with_config():
    """
    批量检测函数，所有参数在代码中配置
    """

    # ================================
    # 配置参数 - 在这里修改你的设置
    # ================================

    # 输入路径 - 可以是单张图片或文件夹
    inputs = '/data/home/qr/mmdetection-main/data/NWPU/val/'  # 或者 'path/to/single_image.jpg'

    # 模型配置
    model = 'configs/dino/sa_dino_sr_swin_NWPU.py'
    weights = 'checkpoints/ours.pth'

    # 输出设置
    out_dir = './output_images/'  # 输出文件夹

    # 推理设置
    device = 'cuda:0'
    pred_score_thr = 0.3
    batch_size = 1

    # 显示和保存设置
    show = False
    no_save_vis = False
    no_save_pred = False
    print_result = True

    # 可视化设置
    palette = 'coco'

    # 文本提示设置
    texts = None
    custom_entities = False
    chunked_size = -1
    tokens_positive = None

    # ================================
    # 🆕 异常检测配置
    # ================================
    enable_anomaly_check = True  # 是否启用异常检测

    # 异常判断规则（根据你的需求调整）:
    min_detections = 1      # 最少检测数量（0表示必须有检测结果）
    max_detections = 100    # 最多检测数量（防止过度检测）
    min_avg_score = 0.4     # 最低平均置信度（低于此值可能不可靠）

    anomaly_output_file = 'anomaly_images.txt'  # 异常图片列表文件名

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

    if enable_anomaly_check:
        print("\n🔍 异常检测: 已启用")
        print(f"  - 最少检测数: {min_detections}")
        print(f"  - 最多检测数: {max_detections}")
        print(f"  - 最低平均分: {min_avg_score}")

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

    # 初始化异常检测器
    anomaly_detector = AnomalyDetector(
        enable=enable_anomaly_check,
        min_detections=min_detections,
        max_detections=max_detections,
        min_avg_score=min_avg_score,
        output_file=anomaly_output_file
    )

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

    # 获取输入图片列表
    if os.path.isdir(inputs):
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(Path(inputs).glob(f'*{ext}'))
            image_files.extend(Path(inputs).glob(f'*{ext.upper()}'))
        image_files = sorted([str(f) for f in image_files])
    else:
        image_files = [inputs]

    print(f"\n找到 {len(image_files)} 张图片")
    print("开始批量推理...\n")

    # 逐张处理（用于异常检测）
    if enable_anomaly_check:
        # 创建临时目录保存单张图片的结果
        temp_pred_dir = os.path.join(out_dir, 'temp_predictions') if out_dir else './temp_predictions'
        os.makedirs(temp_pred_dir, exist_ok=True)

        for idx, img_path in enumerate(image_files, 1):
            img_name = os.path.basename(img_path)
            print(f"处理 [{idx}/{len(image_files)}]: {img_name}", end=" ")

            try:
                # 执行推理，保存到临时目录
                inferencer(
                    inputs=img_path,
                    out_dir=temp_pred_dir,
                    texts=texts,
                    pred_score_thr=pred_score_thr,
                    batch_size=1,
                    show=False,
                    no_save_vis=True,  # 不保存可视化（提高速度）
                    no_save_pred=False,  # 必须保存预测结果
                    print_result=False,
                    custom_entities=custom_entities
                )

                # 读取刚生成的预测JSON文件
                pred_json_path = os.path.join(temp_pred_dir, 'predictions.json')
                if os.path.exists(pred_json_path):
                    with open(pred_json_path, 'r') as f:
                        pred_data = json.load(f)

                    # 找到当前图片的预测结果
                    img_prediction = None
                    if isinstance(pred_data, list):
                        # 列表格式：[{img1}, {img2}, ...]
                        for item in pred_data:
                            if img_name in item.get('img_path', '') or img_name == os.path.basename(item.get('img_path', '')):
                                img_prediction = item
                                break
                        # 如果是单张图片，取最后一个
                        if not img_prediction and len(pred_data) > 0:
                            img_prediction = pred_data[-1]
                    elif isinstance(pred_data, dict):
                        img_prediction = pred_data

                    if img_prediction:
                        # 检查结果
                        is_anomaly, reason = anomaly_detector.check_result(img_path, img_prediction)

                        if is_anomaly:
                            anomaly_detector.add_anomaly(img_path, reason)
                        else:
                            print("  ✓")

                    # 删除临时JSON文件
                    os.remove(pred_json_path)
                else:
                    print("  ⚠️  未生成预测文件")

            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                anomaly_detector.add_anomaly(img_path, f"处理异常: {str(e)}")

        # 删除临时目录
        try:
            os.rmdir(temp_pred_dir)
        except:
            pass

        # 最后再统一进行一次完整推理（如果需要保存可视化结果）
        if not no_save_vis:
            print("\n正在生成可视化结果...")
            inferencer(
                inputs=inputs,
                out_dir=out_dir,
                texts=texts,
                pred_score_thr=pred_score_thr,
                batch_size=batch_size,
                show=show,
                no_save_vis=False,
                no_save_pred=not no_save_pred,
                print_result=print_result,
                custom_entities=custom_entities
            )

        # 保存异常列表
        if out_dir:
            anomaly_detector.save_anomaly_list(out_dir)

    else:
        # 不启用异常检测时，使用原始批量推理
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
        except Exception as e:
            print(f"推理过程出错: {e}")
            return

    if out_dir != '' and not (no_save_vis and no_save_pred):
        print_log(f'\n结果已保存到: {out_dir}')

    print("\n" + "=" * 60)
    print("批量检测完成!")
    print("=" * 60)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='/data/home/qr/mmdetection-main/outputimages/',
        help='Output directory of images or prediction results.')
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
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names?')
    parser.add_argument(
        '--chunked-size',
        '-s',
        type=int,
        default=-1,
        help='Chunked size for large number of categories.')
    parser.add_argument(
        '--tokens-positive',
        '-p',
        type=str,
        help='Token positions for Grounding DINO.')

    # 新增异常检测参数
    parser.add_argument(
        '--enable-anomaly-check',
        action='store_true',
        help='Enable anomaly detection for missing or incorrect detections')
    parser.add_argument(
        '--min-detections',
        type=int,
        default=1,
        help='Minimum number of detections (below this may indicate missing objects)')
    parser.add_argument(
        '--max-detections',
        type=int,
        default=100,
        help='Maximum number of detections (above this may indicate over-detection)')
    parser.add_argument(
        '--min-avg-score',
        type=float,
        default=0.4,
        help='Minimum average confidence score')

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
        init_args, call_args = parse_args()

        # 提取异常检测参数
        enable_anomaly = call_args.pop('enable_anomaly_check')
        min_dets = call_args.pop('min_detections')
        max_dets = call_args.pop('max_detections')
        min_score = call_args.pop('min_avg_score')

        inferencer = DetInferencer(**init_args)

        chunked_size = call_args.pop('chunked_size')
        inferencer.model.test_cfg.chunked_size = chunked_size

        # 如果启用异常检测，需要特殊处理
        if enable_anomaly:
            anomaly_detector = AnomalyDetector(
                enable=True,
                min_detections=min_dets,
                max_detections=max_dets,
                min_avg_score=min_score
            )

            inputs = call_args['inputs']
            out_dir = call_args['out_dir']

            # 获取图片列表
            if os.path.isdir(inputs):
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.extend(Path(inputs).glob(f'*{ext}'))
                    image_files.extend(Path(inputs).glob(f'*{ext.upper()}'))
                image_files = sorted([str(f) for f in image_files])
            else:
                image_files = [inputs]

            # 创建临时目录
            temp_pred_dir = os.path.join(out_dir, 'temp_predictions') if out_dir else './temp_predictions'
            os.makedirs(temp_pred_dir, exist_ok=True)

            # 逐张处理
            for img_path in image_files:
                img_name = os.path.basename(img_path)

                # 临时修改输出目录
                temp_call_args = call_args.copy()
                temp_call_args['inputs'] = img_path
                temp_call_args['out_dir'] = temp_pred_dir
                temp_call_args['no_save_vis'] = True
                temp_call_args['no_save_pred'] = False

                inferencer(**temp_call_args)

                # 读取预测结果
                pred_json_path = os.path.join(temp_pred_dir, 'predictions.json')
                if os.path.exists(pred_json_path):
                    with open(pred_json_path, 'r') as f:
                        pred_data = json.load(f)

                    img_prediction = None
                    if isinstance(pred_data, list) and len(pred_data) > 0:
                        img_prediction = pred_data[-1]
                    elif isinstance(pred_data, dict):
                        img_prediction = pred_data

                    if img_prediction:
                        is_anomaly, reason = anomaly_detector.check_result(img_path, img_prediction)
                        if is_anomaly:
                            anomaly_detector.add_anomaly(img_path, reason)

                    os.remove(pred_json_path)

            # 清理临时目录
            try:
                os.rmdir(temp_pred_dir)
            except:
                pass

            # 保存完整结果和异常列表
            if out_dir:
                inferencer(**call_args)
                anomaly_detector.save_anomaly_list(out_dir)
        else:
            inferencer(**call_args)

        if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                               and call_args['no_save_pred']):
            print_log(f'results have been saved at {call_args["out_dir"]}')


if __name__ == '__main__':
    main()