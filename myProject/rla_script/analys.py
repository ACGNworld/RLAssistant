import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取 CSV 文件
csv_path = ".\\log\\quadrotor-RLA-PPO\\2025\\04\\26\\14-14-28-230599_192.168.31.159_&info=default exp info&seed=18&env=CrazyFile\\progress.csv"
df = pd.read_csv(csv_path)

# 获取CSV文件所在目录
save_dir = os.path.dirname(csv_path)

# 查看列名（确认数据字段）
print(df.columns)

# 创建图表
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("Training Metrics Overview")

# 子图 1: KL 散度
axes[0, 0].plot(df["time/total_timesteps"], df["train/approx_kl"], color="blue")
axes[0, 0].set_title("Approx KL Divergence")
axes[0, 0].grid()

# 子图 2: 价值损失
axes[0, 1].plot(df["time/total_timesteps"], df["train/value_loss"], color="red")
axes[0, 1].set_title("Value Loss")
axes[0, 1].grid()

# 子图 3: 策略损失
axes[1, 0].plot(df["time/total_timesteps"], df["train/policy_gradient_loss"], color="green")
axes[1, 0].set_title("Policy Loss")
axes[1, 0].grid()

# 子图 4: 熵
axes[1, 1].plot(df["time/total_timesteps"], df["train/entropy_loss"], color="orange")
axes[1, 1].set_title("Entropy")
axes[1, 1].grid()

# 子图 5: 裁剪比例
axes[2, 0].plot(df["time/total_timesteps"], df["train/clip_fraction"], color="purple")
axes[2, 0].set_title("Clip Fraction")
axes[2, 0].grid()

# 子图 6: 解释方差
axes[2, 1].plot(df["time/total_timesteps"], df["train/explained_variance"], color="brown")
axes[2, 1].set_title("Explained Variance")
axes[2, 1].grid()

plt.tight_layout()

# 构建图片保存路径
img_path = os.path.join(save_dir, "training_metrics.png")

# 保存图表到CSV所在目录
plt.savefig(img_path, dpi=300, bbox_inches='tight')
print(f"图表已保存到: {img_path}")

plt.show()