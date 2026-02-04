graph LR
    %% ==========================================
    %% 1. 全局样式定义
    %% ==========================================
    classDef existing fill:#f5f5f5,stroke:#666,stroke-width:1px,stroke-dasharray: 5 5;
    classDef novel fill:#e3f2fd,stroke:#1565c0,stroke-width:3px;
    
    %% ==========================================
    %% 2. 硬件层 (左侧输入端)
    %% ==========================================
    subgraph Layer1 ["100 物理硬件层 (现有基础)"]
        direction TB
        A["110 离子阱装置"]:::existing --> B["120 激光系统"]:::existing
        B --> C["130 成像模组<br>(sCMOS/EMCCD)"]:::existing
    end

    %% 连接
    C == "原始荧光信号流" ==> D

    %% ==========================================
    %% 3. 核心算法层 (中间核心处理)
    %% ==========================================
    subgraph Layer2 ["200 核心创新：计算处理流水线"]
        direction LR
        %% 排列逻辑：从左到右流转
        D["210 张量构建"]:::novel --> E["220 RPCA<br>背景分离"]:::novel
        
        E -- "稀疏信号" --> F["230 拓扑约束<br>网格配准"]:::novel
        E -. "低秩背景" .-> Trash(("丢弃噪声")):::existing
        
        F --> G["240 NMF<br>串扰解耦"]:::novel
        E -- "原始信号" --> G
        
        G -- "纯净光子数" --> H["250 GMM<br>自适应判决"]:::novel
    end

    %% 连接
    H == "判决结果" ==> I

    %% ==========================================
    %% 4. 输出层 (右侧输出端)
    %% ==========================================
    subgraph Layer3 ["300 输出层"]
        I["310 量子态接口<br>(0/1 状态)"]:::novel
    end

    %% ==========================================
    %% 5. 图例 (底部)
    %% ==========================================
    subgraph Legend ["图例说明"]
        L1["现有技术/硬件"]:::existing
        L2["本提案创新点"]:::novel
    end

    %% 样式微调
    style Layer1 fill:#fafafa,stroke:#999,stroke-dasharray: 5 5
    style Layer2 fill:#fff,stroke:#1565c0,stroke-width:2px
    style Layer3 fill:#fafafa,stroke:#999,stroke-dasharray: 5 5
