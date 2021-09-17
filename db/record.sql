# 动作状态记录
CREATE TABLE action_state_record(
id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
player_name VARCHAR(32) NOT NULL COMMENT '用来区分当前使用的机器人',
table_uid int(11) NOT NULL DEFAULT 1111 COMMENT '桌号',
round_count int(11) NOT NULL DEFAULT 0 COMMENT '轮次',
step VARCHAR(32) NOT NULL COMMENT '所在轮次中的阶段',
stack int(11) NOT NULL DEFAULT 0 COMMENT '当前筹码量',
hole_card VARCHAR(32) COMMENT '手牌',
community_card VARCHAR(32) COMMENT '公共牌',
valid_actions VARCHAR(125) COMMENT '有效动作',
pot VARCHAR(32) COMMENT '场上筹码',
seats VARCHAR(512) COMMENT '场上选手信息',
action int(11) COMMENT '当前状态采取的动作',
amount int(11) COMMENT '当前动作的筹码量'
);

# 结果记录
CREATE TABLE result_record(
id BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
round_count int(11) NOT NULL DEFAULT 0 COMMENT '轮次',
player_name VARCHAR(32) NOT NULL  COMMENT '用来区分当前使用的机器人',
seats VARCHAR(512) COMMENT '场上选手信息',
);