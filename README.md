graph TD
    %% --- 样式定义 ---
    %% 现有技术：灰色虚线框，浅灰背景
    classDef existing fill:#f9f9f9,stroke:#666,stroke-width:2px,stroke-dasharray: 5 5;
    %% 本提案新增：蓝色实线粗框，淡蓝背景
    classDef novel fill:#e3f2fd,stroke:#1565c0,stroke-width:3px;
    
    %% --- 图例说明 (置于顶部以便查阅) ---
    subgraph Legend ["图例说明"]
        direction LR
        L1["现有技术 / 通用硬件"]:::existing
        L2["本提案新增核心组件"]:::novel
    end

    %% --- 第一层：物理硬件层 (现有基础) ---
    subgraph Layer1 ["100 物理硬件层 (现有基础)"]
        direction LR
        A["110 离子阱装置<br>线性链拓扑"]:::existing --> B["120 探测/冷却<br>激光系统"]:::existing
        B --> C["130 高分辨率成像模组<br>物镜 + sCMOS/EMCCD"]:::existing
    end

    %% 连接
    C == 原始荧光信号流 ==> D

    %% --- 第二层：计算处理流水线层 (核心创新) ---
    subgraph Layer2 ["200 核心创新：计算处理流水线层"]
        direction TB
        D["210 图像张量<br>构建模块"]:::novel --> E["220 鲁棒主成分分析<br>(RPCA) 处理器"]:::novel
        
        E -- "稀疏信号矩阵 (前景)" --> F["230 拓扑约束<br>弹性网格配准引擎"]:::novel
        E -. "低秩结构化背景" .-> Trash(("丢弃/忽略")):::existing
        
        F -- 配准坐标 --> G["240 盲源分离与<br>串扰解耦单元 (NMF)"]:::novel
        E -- 原始稀疏信号 --> G
        
        G -- 解耦后的纯净光子数 --> H["250 自适应 GMM<br>聚类判决器"]:::novel
    end

    %% 连接
    H == 判决结果 ==> I

    %% --- 第三层：输出与反馈层 ---
    subgraph Layer3 ["300 输出与反馈层"]
        I["310 最终量子态读出接口<br>二进制状态向量"]:::novel
    end

    %% 整体层级背景微调
    style Layer1 fill:#f0f0f0,stroke:none
    style Layer2 fill:#fff,stroke:#1565c0,stroke-width:2px
    style Layer3 fill:#f0f0f0,stroke:none
