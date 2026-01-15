```mermaid
graph TD
    %% 定义全局样式：黑白线条，无阴影，符合专利风格
    classDef default fill:#fff,stroke:#000,stroke-width:2px;
    classDef cluster fill:#fff,stroke:#000,stroke-width:3px,stroke-dasharray: 5 5;

    %% 第一层：物理硬件层
    subgraph Layer1 [100 物理硬件层]
        direction LR
        A[110 离子阱装置<br>线性链拓扑] --> B[120 探测/冷却<br>激光系统]
        B --> C[130 高分辨率成像模组<br>物镜 + sCMOS/EMCCD]
    end

    %% 连接层级
    C == 原始荧光信号流 ==> D

    %% 第二层：计算处理流水线层
    subgraph Layer2 [200 计算处理流水线层]
        direction TB
        D[210 图像张量<br>构建模块] --> E[220 鲁棒主成分分析<br>RPCA 处理器]
        
        E -- 稀疏信号矩阵 --> F[230 拓扑约束<br>弹性网格配准引擎]
        E -. 低秩背景噪声 .-> Trash((丢弃))
        
        F -- 配准坐标 --> G[240 盲源分离与<br>串扰解耦单元]
        E -- 原始稀疏信号 --> G
        
        G -- 解耦光子数 --> H[250 自适应 GMM<br>聚类判决器]
    end

    %% 连接层级
    H --> I

    %% 第三层：输出与反馈层
    subgraph Layer3 [300 输出与反馈层]
        I[310 最终量子态读出接口<br>二进制状态向量]
    end

    %% 样式调整
    style Layer1 fill:#fff,stroke:#000,stroke-width:2px
    style Layer2 fill:#fff,stroke:#000,stroke-width:2px
    style Layer3 fill:#fff,stroke:#000,stroke-width:2px
