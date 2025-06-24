import wandb
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import os

WANDB_PROJECT_NAME = "huggingface_example"
WANDB_ENTITY_NAME = None 

# --- 2. 初始化 WandB ---
def initialize_wandb():
    """初始化 WandB run 并记录超参数。"""
    print(f"初始化 WandB run 到项目: {WANDB_PROJECT_NAME}")
    wandb.init(
        project=WANDB_PROJECT_NAME,
        entity=WANDB_ENTITY_NAME,
        config={
            "learning_rate": 0.01 + random.random() * 0.01,
            "epochs": 10,
            "batch_size": 32,
            "optimizer": random.choice(["Adam", "SGD", "RMSprop"]),
            "dropout_rate": 0.2 + random.random() * 0.3
        },
        name=f"test-run-1",
    )
    print("WandB run 已成功初始化！")
    print(f"本次运行的超参数: {wandb.config}")

# --- 3. 模拟训练过程并记录指标 ---
def simulate_training():
    """模拟训练循环并记录指标。"""
    print("开始模拟训练过程...")
    for epoch in range(wandb.config.epochs):
        # 模拟训练损失和准确率
        train_loss = 1.5 - (epoch * 0.1) + (random.random() * 0.1)
        train_accuracy = 0.5 + (epoch * 0.04) + (random.random() * 0.02)

        # 模拟验证损失和准确率
        val_loss = 1.2 - (epoch * 0.09) + (random.random() * 0.1)
        val_accuracy = 0.6 + (epoch * 0.03) + (random.random() * 0.03)

        # 使用 wandb.log() 记录指标
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "val/loss": val_loss,
            "val/accuracy": val_accuracy
        })
        print(f"Epoch {epoch+1}/{wandb.config.epochs}: Train Loss={train_loss:.4f}, Val Accuracy={val_accuracy:.4f}")
        time.sleep(0.5) # 模拟每个 epoch 的耗时

    print("模拟训练完成。")

# --- 4. 记录可视化对象 ---
def log_visualizations():
    """记录图片和表格等可视化对象。"""
    print("记录可视化对象...")

    # 记录一张随机生成的图片
    plt.figure(figsize=(6, 4))
    plt.plot(np.random.rand(50).cumsum(), label="Random Walk")
    plt.title("Sample Plot")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()
    image_path = "sample_plot.png"
    plt.savefig(image_path)
    plt.close()
    wandb.log({"sample_image": wandb.Image(image_path)})
    print(f"已记录图片: {image_path}")

    # 记录一个表格
    data = [
        ["Model A", 0.92, "Good"],
        ["Model B", 0.88, "Fair"],
        ["Model C", 0.95, "Excellent"]
    ]
    columns = ["Model Name", "Accuracy", "Performance"]
    table = wandb.Table(data=data, columns=columns)
    wandb.log({"model_comparison_table": table})
    print("已记录表格数据。")

# --- 5. 模拟保存模型并记录文件 ---
def log_model_artifact():
    """模拟保存模型文件并上传到 WandB artifacts。"""
    print("模拟保存模型文件...")
    model_filename = "my_dummy_model.pt"
    with open(model_filename, "w") as f:
        f.write("This is a placeholder for your trained model weights.\n")
        f.write(f"Learning rate used: {wandb.config.learning_rate}\n")
        f.write(f"Optimizer used: {wandb.config.optimizer}\n")

    # 使用 wandb.save() 上传文件
    wandb.save(model_filename)
    print(f"已将 '{model_filename}' 文件上传到 WandB。")

# --- 主执行函数 ---
def main():
    try:
        # 步骤 1: 初始化 WandB
        initialize_wandb()

        # 步骤 2: 模拟训练并记录指标
        simulate_training()

        # 步骤 3: 记录可视化对象
        log_visualizations()

        # 步骤 4: 记录模型文件
        log_model_artifact()

    except Exception as e:
        print(f"运行过程中发生错误: {e}")
    finally:
        # 步骤 5: 结束 WandB run
        # 确保所有数据被上传
        print("结束 WandB run...")
        wandb.finish()
        print("WandB run 已结束。请访问 WandB UI 查看您的实验结果。")

if __name__ == "__main__":
    # 在运行此脚本之前，请确保您已经通过 'wandb login' 命令登录了 WandB。
    # 如果没有登录，WandB 会在第一次运行 wandb.init() 时提示您登录。
    main()