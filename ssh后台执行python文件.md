使用 nohup
SSH 连接到服务器：

bash
复制
编辑
ssh your_username@your_server_ip
进入代码所在目录：

bash
复制
编辑
cd /home/jlhao/Proj/RadioMap/RM_部分条件DM/基于diffmae的实现/
运行 Python 代码并使其在后台运行：

bash
复制
编辑
nohup python train1.py > train1.log 2>&1 &
nohup：防止进程因 SSH 断开而终止。
>：将标准输出重定向到 train1.log。
2>&1：将错误信息也重定向到 train1.log。
&：在后台运行。
检查进程是否在运行：

bash
复制
编辑
ps aux | grep train1.py
