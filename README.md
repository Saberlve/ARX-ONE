ACone robot,整体架构,数采,训练,推理部署.



1.qucik start
ACone项目文件环境配置,参考文档:https://alidocs.dingtalk.com/i/nodes/ZQYprEoWon7RKy9GiqkZZgoZ81waOeDk?utm_scene=team_space

1.1 collect dataset
(1)首先启动机械臂
	bash ./tools/01_collect.sh
启动机械臂,按下2回到数据采集初始状态.

(2)启动数据采集文件
	cd ROS2_AC-one_Play
	python ./src/edlsrobot/datasets/collect_ledatav21.py
	
(3)数据转换lerobot, v21 -> v30
	python ./src/edlsrobot/datasets/convert_dataset_v21_to_v30.py
	
(4) 回放
	先关掉01_collect.sh,然后启动下面推理模式.
	bash ./tools/04_reply.sh,
	最后在执行下面程序.
	python ./src/edlsrobot/datasets/replay.py

参数可在代码中修改,主要修改参数:
	--root_path	# 数据保存路径.
	--episode_nums	# 采集样本数量.
	--frame_rate	# 帧率.
	--resume	# 是否回复训练.

1.2 train
位于第三方库,也可以其他版本库.

	/ROS2_AC-one_Play/third_lib/lerobot/src/lerobot/scripts/lerobot_train.py


1.3 inference
目前三种算法推理模型.pi05,smolvla, act,可在infer.sh中修改参数.
	cd ./src/edlsrobot/scripts
	bash infer.sh



2.update ...


