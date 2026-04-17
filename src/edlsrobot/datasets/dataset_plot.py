import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_parquet_column(file_path, column_name="observation.state", dtype=np.float32):
    """
    从 parquet 文件中读取指定列，返回 shape=[T, D] 的 numpy 数组
    """
    df = pd.read_parquet(file_path)

    print("Columns in dataframe:", df.columns)

    if column_name not in df.columns:
        raise ValueError(f"Parquet 文件中没有 '{column_name}' 列，请确认列名")

    data = df[column_name].to_list()
    data_array = np.asarray(data, dtype=dtype)

    if data_array.ndim != 2:
        raise ValueError(f"{column_name} 数据维度异常: {data_array.shape}")

    print(f"{column_name} shape: {data_array.shape}")
    return data_array


def plot_selected_dims_from_file(
    file_path,
    column_name="observation.state",
    selected_dims=None,
    save_path="selected_dims_plot.png",
    title=None,
):
    """
    方法一：
    将选中的多个维度画到同一张图里
    """
    data_array = load_parquet_column(file_path, column_name)

    num_dims = data_array.shape[1]
    if selected_dims is None:
        selected_dims = list(range(num_dims))

    for dim in selected_dims:
        if dim >= num_dims:
            raise ValueError(f"选择的维度 {dim} 超出最大维度 {num_dims - 1}")

    dim_values = [data_array[:, i] for i in selected_dims]
    x_values = np.arange(len(data_array))

    plt.figure(figsize=(12, 6))
    colors = plt.cm.get_cmap("tab10", len(selected_dims)).colors
    labels = [f"motor_{i}" for i in selected_dims]

    for i in range(len(selected_dims)):
        plt.plot(x_values, dim_values[i], color=colors[i], label=labels[i], linewidth=2)

    if title is None:
        title = f"Selected Dimensions of {column_name} Over Index"

    plt.title(title)
    plt.xlabel("Index / Range")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved selected dims figure to: {save_path}")


def plot_joint_curves_from_file(
    file_path,
    column_name="observation.state",
    save_path="joint_curves.png",
):
    """
    方法二：
    默认按 14 维数据处理，画成 7x2 子图（左臂7个、右臂7个）
    """
    data_array = load_parquet_column(file_path, column_name)

    if data_array.shape[1] != 14:
        raise ValueError(f"{column_name} 维度不是14，当前 shape={data_array.shape}")

    left_names = ["L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6", "L_Gripper"]
    right_names = ["R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6", "R_Gripper"]

    fig, axes = plt.subplots(7, 2, figsize=(14, 20), sharex=True)

    for i in range(7):
        axes[i, 0].plot(data_array[:, i])
        axes[i, 0].set_title(left_names[i])
        axes[i, 0].grid(True)

        axes[i, 1].plot(data_array[:, i + 7])
        axes[i, 1].set_title(right_names[i])
        axes[i, 1].grid(True)

    axes[-1, 0].set_xlabel("Timestep")
    axes[-1, 1].set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved joint curve figure to: {save_path}")


def plot_joint_diff_curves_from_file(
    file_path,
    column_name="observation.state",
    save_path="joint_diff_curves.png",
):
    """
    方法三：
    读取文件后，画相邻两步的差分曲线
    """
    data_array = load_parquet_column(file_path, column_name)

    if data_array.shape[1] != 14:
        raise ValueError(f"{column_name} 维度不是14，当前 shape={data_array.shape}")

    diffs = np.diff(data_array, axis=0)

    joint_names = [
        "L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_J6", "L_Gripper",
        "R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_J6", "R_Gripper",
    ]

    fig, axes = plt.subplots(14, 1, figsize=(12, 28), sharex=True)

    for i in range(14):
        axes[i].plot(diffs[:, i])
        axes[i].set_ylabel(joint_names[i], rotation=0, labelpad=35)
        axes[i].grid(True)

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved joint diff figure to: {save_path}")


if __name__ == "__main__":
    file_path = "/home/ubuntu/edlsrobot/repos/ROS2_AC-one_Play/All_datas/pour_tea/data/chunk-000/episode_000200.parquet"

    # 可选列:
    # "observation.state"
    # "action"
    # "observation.velocity"
    # "observation.effort"
    column_name = "observation.state"

    # 方法一：多维同图
    plot_selected_dims_from_file(
        file_path=file_path,
        column_name=column_name,
        selected_dims=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        save_path="selected_dims_plot.png",
    )

    # 方法二：左右臂子图
    plot_joint_curves_from_file(
        file_path=file_path,
        column_name=column_name,
        save_path="joint_curves.png",
    )

    # 方法三：相邻步差分图
    plot_joint_diff_curves_from_file(
        file_path=file_path,
        column_name=column_name,
        save_path="joint_diff_curves.png",
    )