from ultralytics import YOLO
import cv2


def main():
    # 加载训练好的模型
    model = YOLO('runs/train/exp/weights/best.pt')  # 加载最佳权重

    # 进行预测
    results = model.predict(
        source='path/to/your/image.jpg',  # 图像路径
        conf=0.25,                        # 置信度阈值
        save=True,                        # 保存结果
        project='runs/predict',           # 保存预测结果的目录
        name='exp'                        # 实验名称
    )

    # 处理预测结果
    for result in results:
        boxes = result.boxes  # 获取检测框
        for box in boxes:
            # 获取坐标
            x1, y1, x2, y2 = box.xyxy[0]
            # 获取置信度
            conf = box.conf[0]
            # 获取类别
            cls = box.cls[0]
            print(
                f'检测到目标: 类别={cls}, 置信度={conf:.2f}, 位置=({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})')


if __name__ == '__main__':
    main()
