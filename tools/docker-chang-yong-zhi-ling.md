# Docker常用指令

搜索镜像 `docker search <image name>`&#x20;

拉去镜像(tag 默认为lastest) `docker pull <image name>:<tag>`&#x20;

删除某个镜像 `docker rmi <image name>:<tag>`&#x20;

查看所有下载镜像的状态 `docker images`&#x20;

导出镜像 `docker export <image name> -o <file name>`&#x20;

重命名镜像 `docker tag <image1 name>:<tag> <image2 name>:<tag>`

查看所有容器的状态 `docker ps -a` / `docker container ls --all`&#x20;

查看正在运行的容器 `docker ps` / `docker container ls`&#x20;

创建一个新的容器 `docker run [OPTIONS] <image name>:<tag>`&#x20;

终止容器运行 `docker kill <container ID>`&#x20;

在容器中运行单个命令(--rm 容器运行结束后自动删除) `docker run --rm <image name> <command>`&#x20;

交互式运行容器 `docker run --name=<window name> -it <image name>`&#x20;

进入退出的容器 `docker start <window name> & docker attach <window name>`&#x20;

删除某几个容器 `docker rm <container1 ID> <container2 ID>`&#x20;

后台运行镜像，只显示ID(-d 指定后台运行) `docker run -d --name=<window name> <image name>`&#x20;

查看运行的详细参数 `docker run --help`&#x20;

可视化运行容器 `docker run --name=<window name> -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE <image name>`（需要在容器外 `xhost +`）&#x20;

挂载文件夹 `docker run --name=<window name> -v <source fold path>:<target fold path> <image name>`
