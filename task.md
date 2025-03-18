物理矛盾：信号在空旷区域遵循平滑衰减（遵循平方反比定律），而建筑物边缘则表现为突变 
表示困难：卷积网络天然偏向平滑表示，难以同时捕获平滑变化和锐利边缘 
采样稀疏性：15%的采样率在建筑物边缘可能不足以准确重建边界效应
超分辨率重建技术，从稀疏采样中恢复高分辨率信号。
动态调整采样密度，在建筑物边缘增加采样点
使用主动学习等技术，选择信息量最大的采样点。
解决问题的思路：
边缘感知机制： 
设计专门的边缘检测分支，使用类似Canny或Sobel的边缘检测器识别建筑物边缘 
将边缘信息作为额外特征通道融入预测过程 
在边缘区域应用特殊的注意力机制，增强模型对边缘的敏感性 
物理导向的多尺度融合： 
设计自适应融合机制，在空旷区域偏向大尺度平滑特征，在建筑物边缘偏向小尺度锐利特征 
引入物理先验，根据电磁波传播理论，在不同区域应用不同的传播模型 
边缘增强损失函数： 
设计区域自适应损失：在建筑物边缘区域使用更高权重的梯度损失 
引入方向性梯度约束：沿建筑物边缘方向保持梯度一致性，垂直于边缘方向允许梯度变化 
添加高阶导数损失：不仅考虑一阶梯度，还考虑二阶导数（曲率），更好地捕获边缘特性 
建筑物边缘特殊处理： 
在预处理阶段，扩展建筑物掩码创建"边缘区域掩码" 
在这些区域应用特殊的边缘增强卷积，类似于图像处理中的锐化操作 
设计边缘感知注意力机制，在建筑物边缘区域增强特征表示
![Pasted Graphic](https://github.com/user-attachments/assets/47e98e50-4344-481a-91c5-bda7402f029f)

授权码 7a01c2222f654e4b87fc96268b2dfe60

3. 相似问题与可借鉴项目
相似领域项目：
主动学习(Active Learning)
Deep Bayesian Active Learning - 基于不确定性的批量采样
相似点：同样是选择最有信息量的样本进行标注/采样
传感器网络优化
Sensor Placement Optimization - 基于高斯过程的传感器放置
相似点：优化传感器位置以最大化信息覆盖
计算机视觉中的关键点采样
Learning to Sample - 学习点云采样策略
相似点：通过稀疏采样重建复杂分布
压缩感知(Compressive Sensing)
CS-based Indoor Localization - 用于室内无线信号定位
相似点：利用信号稀疏性实现重建
学术论文参考：
"Active Learning for Wireless Communications: From Signal Reconstruction to Channel Estimation" - 探讨在无线通信中应用主动学习进行信号采样
"Efficient Radio Map Construction Based on Matrix Completion for Indoor Positioning" - 使用矩阵补全方法从稀疏测量重建RadioMap
"DeepSense: Deep Learning for Unsupervised Sensor Selection in Mobile Sensing" - 使用深度学习选择传感器位置

