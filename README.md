graph TD
    %% --- 样式定义 ---
    classDef existing fill:#fff,stroke:#000,stroke-width:1px;
    classDef novel fill:#fff,stroke:#000,stroke-width:4px;
    classDef cluster fill:#fff,stroke:#000,stroke-width:2px,stroke-dasharray: 5 5;

    %% --- 第一层：物理硬件层 ---
    %% 修复点：给标题加上双引号 "..."
    subgraph Layer1 ["100 物理硬件层 (现有基础)"]
        direction LR
        A["110 离子阱装置<br>线性链拓扑"]:::existing --> B["120 探测/冷却<br>激光系统"]:::existing
        B --> C["130 高分辨率成像模组<br>物镜 + sCMOS/EMCCD"]:::existing
    end

    %% 连接
    C == 原始荧光信号流 ==> D

    %% --- 第二层：计算处理流水线层 ---
    %% 修复点：给标题加上双引号
    subgraph Layer2 ["200 核心创新：计算处理流水线层"]
        direction TB
        D["210 图像张量<br>构建模块"]:::novel --> E["220 鲁棒主成分分析<br>RPCA 处理器"]:::novel
        
        E -- 稀疏信号矩阵 --> F["230 拓扑约束<br>弹性网格配准引擎"]:::novel
        E -. 低秩背景噪声 .-> Trash(("丢弃")):::existing
        
        F -- 配准坐标 --> G["240 盲源分离与<br>串扰解耦单元"]:::novel
        E -- 原始稀疏信号 --> G
        
        G -- 解耦光子数 --> H["250 自适应 GMM<br>聚类判决器"]:::novel
    end

    %% 连接
    H --> I

    %% --- 第三层：输出与反馈层 ---
    subgraph Layer3 ["300 输出与反馈层"]
        I["310 最终量子态读出接口<br>二进制状态向量"]:::novel
    end

    %% --- 图例说明 ---
    subgraph Legend ["图例说明"]
        direction LR
        L1[现有硬件/组件]:::existing
        L2[本提案新增/改进模块]:::novel
    end

    %% 样式应用
    style Layer1 fill:#fafafa,stroke:#666,stroke-width:1px,stroke-dasharray: 5 5
    style Layer2 fill:#fff,stroke:#000,stroke-width:2px
    style Layer3 fill:#fafafa,stroke:#666,stroke-width:1px,stroke-dasharray: 5 5
