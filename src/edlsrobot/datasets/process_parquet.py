""" 保存时action=state，将所有的action的改变成state的后一帧。action保存为下一时刻的state """



# from datasets import load_dataset
#
# dataset = load_dataset("parquet", data_files="/home/arx/All_program/hkk/arx_model/mobile_aloha/datas/lerobotdatasets/grasp_bo/data/chunk-000-new/episode_000000.parquet")["train"]
# print(dataset)
# # print(dataset["action"])
# for i in range(5):
#     print(dataset[i])  # 查看第一个样本
#
# # import pandas as pd
# df = pd.read_parquet("/home/arx/All_program/hkk/arx_model/mobile_aloha/datas/lerobotdatasets/grasp_bo/data/chunk-000/episode_000000.parquet")
# print(df.head())
# print(df.columns)
# print(df.dtypes)





"---------------------------批量转换----------------------------"
# import pandas as pd
# import numpy as np
# from pathlib import Path
#
# def shift_action_from_state(state_array):
#     """将 state 平移一帧生成 action"""
#     actions = np.copy(state_array)
#     actions[:-1] = state_array[1:]
#     actions[-1] = state_array[-1]
#     return actions
#
# def verify_timestamps(timestamps, fps, tolerance_ratio=0.01):
#     """检查时间戳间隔是否符合 fps 容差"""
#     timestamps = np.array(timestamps, dtype=np.float64)
#     delta_ts = np.diff(timestamps)
#     expected_dt = 1.0 / fps
#     tolerance = expected_dt * tolerance_ratio
#     if np.any(np.abs(delta_ts - expected_dt) > tolerance):
#         indices = np.where(np.abs(delta_ts - expected_dt) > tolerance)[0]
#         raise ValueError(
#             f"时间戳间隔超出容差: indices={indices}, delta={delta_ts[indices]}, "
#             f"expected_dt={expected_dt}, tolerance={tolerance}"
#         )
#
# def process_parquet(input_path: Path, output_dir: Path, fps: float):
#     df = pd.read_parquet(input_path)
#
#     if "observation.state" not in df.columns:
#         print(f"{input_path.name} 没有 'state' 列，跳过。")
#         return
#
#     # # 保证 timestamp 类型不变
#     # if "timestamp" in df.columns:
#     #     df["timestamp"] = df["timestamp"].astype(np.float32)
#
#     # 检查 timestamp 是否升序且 delta 在容差内
#     if "timestamp" in df.columns:
#         verify_timestamps(df["timestamp"], fps)
#
#     # 将 state 列展开成 numpy 数组
#     states = np.stack(df["observation.state"].to_numpy())
#     shifted_actions = shift_action_from_state(states)
#
#     # 更新 action 列
#     df["action"] = list(shifted_actions)
#
#     # 输出路径
#     output_path = output_dir / input_path.name
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#
#     # 保存 parquet，不保存 DataFrame index
#     df.to_parquet(output_path, index=False)
#
#     print(f"✅ 已处理 {input_path.name} → {output_path}")
#
# def batch_shift_actions(root_dir: str, output_dir: str, fps: float):
#     root = Path(root_dir)
#     out = Path(output_dir)
#     out.mkdir(parents=True, exist_ok=True)
#
#     parquet_files = sorted(root.glob("**/*.parquet"))
#     if not parquet_files:
#         print("❌ 未找到任何 .parquet 文件。")
#         return
#
#     for f in parquet_files:
#         process_parquet(f, out, fps)
#
#     print(f"\n 全部处理完成，共 {len(parquet_files)} 个文件。")
#
# if __name__ == "__main__":
#     # 修改为你的数据目录
#     batch_shift_actions(
#         root_dir="/home/arx/All_program/hkk/arx_model/mobile_aloha/datas/lerobotdatasets/grasp_bo/data/chunk-000",             # 原始数据目录
#         output_dir="/home/arx/All_program/hkk/arx_model/mobile_aloha/datas/lerobotdatasets/grasp_bo/data/chunk-000-new",   # 对齐后的输出目录
#         fps=30.0                                       # 数据采集 FPS，确保和原数据一致
#     )





"""-----------state状态量分析---------"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ 读取 Parquet 文件
# -------------------------------
file_path = "/home/ubuntu/edlsrobot/repos/ROS2_AC-one_Play/All_datas/pour_teav21/data/chunk-000/episode_000009.parquet"  # 修改为你的文件路径
# file_path = "/home/arx/All_program/hkk/arx_model/All_datas/lerobotdatasets/state_action_plot/grasp_tttt_25ok/data/chunk-000/episode_000000.parquet"  # 修改为你的文件路径

df = pd.read_parquet(file_path)

# 查看列名，确认 state 列名
print("Columns in dataframe:", df.columns)

# 假设 state 列叫 'state'
if 'observation.state' not in df.columns:
    raise ValueError("Parquet 文件中没有 'state' 列，请确认列名")


# -------------------------------
# 2️⃣ 提取每个维度的数据
# -------------------------------
# 假设每个 state 是长度为 N 的列表或 np.array
states = df['observation.state'].to_list()  # list of lists
# states = df['action'].to_list()  # list of lists
# states = df['observation.velocity'].to_list()
# states = df['observation.effort'].to_list()

# states = states[:50]   ##截取前100帧
states_array = np.array(states)  # shape: [num_samples, N]

num_dims = states_array.shape[1]
print(f"state 的维度为 {num_dims}")

# -------------------------------
# 3️⃣ 选择要画的维度
# -------------------------------
#
# selected_dims = [0, 1, 2, 3, 4, 5, 6, ]  # left
# selected_dims = [7, 8, 9, 10, 11, 12, 13, ]    #right arm
selected_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

for dim in selected_dims:
    if dim >= num_dims:
        raise ValueError(f"选择的维度 {dim} 超出 state 最大维度 {num_dims-1}")

# 拆分选中的维度
dim_values = [states_array[:, i] for i in selected_dims]
left_gripper = dim_values[6]
right_gripper = dim_values[13]
print("------> left --------", left_gripper.min())
print("------> right --------", right_gripper.min())


# -------------------------------
# 4️⃣ 绘制线条图
# -------------------------------
plt.figure(figsize=(12, 6))
x_values = np.arange(len(states_array))  # 横坐标：索引
# x_values = np.arange(50)

# 使用 tab10 调色板，足够 10 条线
colors = plt.cm.get_cmap('tab10', len(selected_dims)).colors
labels = [f"motor_{i}" for i in selected_dims]

for i in range(len(selected_dims)):
    plt.plot(x_values, dim_values[i], color=colors[i], label=labels[i], linewidth=2)

plt.title("Selected Dimensions of observation.state Over Index")
plt.xlabel("Index / Range")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 可选：保存图表
# plt.savefig("selected_state_dimensions_plot.png", dpi=300)



