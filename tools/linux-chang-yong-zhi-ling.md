# Linux常用指令

## OpenCV

### Ubuntu查看已安装的OpenCV的版本和库

`pkg-config --modversion opencv` `pkg-config --libs opencv`

## APT

### Could not get lock /var/lib/dpkg/lock-frontend

[解决方案](https://blog.csdn.net/lun55423/article/details/108907779)

## ROS

### rosbag

#### 查看数据包中的信息&#x20;

`rosbag info ***.bag`&#x20;

#### 回放数据包中包含的topic内容（要在另一个终端内先启动ROS master，即roscore）&#x20;

`rosbag play ***.bag`&#x20;

#### N倍速回放&#x20;

`rosbag play -r N ***.bag`&#x20;

#### 循环播放&#x20;

`rosbag play ***.bag -l [--loop]`&#x20;

#### 暂停&#x20;

`rosbag play --pause ***.bag`&#x20;

#### 常用命令：&#x20;

![](<../.gitbook/assets/image (1026).png>)



## 命令

### tmux

#### 启动新的会话&#x20;

`tmux new -s name`&#x20;

#### 退出会话&#x20;

`ctrl B + D`&#x20;

#### 在会话中新建窗口&#x20;

`ctrl B + C`&#x20;

#### 在会话中切换窗口&#x20;

`ctrl B + id`&#x20;

#### 查看所有会话&#x20;

`tmux ls`&#x20;

#### 回到某会话中&#x20;

`tmux a -t name`&#x20;

#### 重命名会话&#x20;

`tmux rename -t source target`&#x20;

#### 关闭某会话&#x20;

`tmux kill-session -t name`
