from ultralytics import YOLO


def main():
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 加载预训练的YOLOv8n模型

    # 训练模型
    results = model.train(
        data='data.yaml',        # 数据配置文件路径
        epochs=100,              # 训练轮数
        imgsz=640,              # 图像大小
        batch=16,               # GPU训练可以使用更大的batch size
        device=0,               # 使用第一块GPU
        workers=8,              # GPU训练可以使用更多workers
        save=True,              # 保存训练结果
        project='runs/train',   # 保存训练结果的目录
        name='exp',             # 实验名称
    )


if __name__ == '__main__':
    main()
