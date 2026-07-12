---
title: 大世界行军优化：从寻路算法到服务器架构的全面实践
published: 2026-07-12
description: "深入解析无尽冬日、Dark Survival、重返帝国等 SLG 游戏在大世界行军系统上的核心技术优化，涵盖寻路算法、视野管理、网络同步与服务器架构。"
image: "/images/ai-industry-2025-header.png"
tags: ["游戏开发", "SLG", "寻路算法", "服务器架构", "性能优化", "行军系统", "视野管理", "网络同步"]
category: 游戏开发
draft: false
---

# 大世界行军优化：从寻路算法到服务器架构的全面实践

## 引言

如果你玩过《无尽冬日》（Whiteout Survival）、《Dark Survival》或《重返帝国》这类 SLG 手游，你一定对"行军"这个操作不陌生——点击地图上的某个目标，你的部队就会沿着一条路线自动走过去。看起来简单，但当这个操作发生在**250 万个六边形格子**的大地图上，同时有**上万个玩家在线**、**千人同屏团战**时，背后的技术挑战就完全不同了。

本文将深入探讨大世界行军系统在**寻路、网络同步、视野管理、服务器架构**四个维度的核心优化，结合《重返帝国》等实际项目的公开技术分享，以及笔者在相关项目中的实践经验。

---

## 第一部分：大世界行军的核心挑战

### 1.1 问题规模

先看一组数据，感受一下 SLG 大世界面临的挑战：

| 维度 | 数据规模 |
|------|---------|
| 地图大小 | 5000m × 5000m，250 万+ 连续六边形网格 |
| 静态地图数据 | 约 70MB（纯障碍物和地形） |
| 可见对象 | 超过 50 万个（资源点、建筑、NPC、部队） |
| 同时在线 | 10000+ 玩家 |
| 团战规模 | 千人同屏，万人观战 |
| 行军单位 | 每玩家最多 5 支部队，皇城战可达 3000+ 部队 |

### 1.2 核心挑战拆解

```
大世界行军系统面临的挑战
├── 寻路挑战
│   ├── 长距离寻路（跨州郡，路径长度可达数千格）
│   ├── 动态阻挡（玩家主城、要塞、公会关系）
│   ├── 高并发寻路请求（每秒数千次）
│   └── 路径质量 vs 性能的平衡
├── 网络同步挑战
│   ├── 行军状态同步（位置、速度、方向）
│   ├── 视野内海量对象同步
│   ├── 带宽限制（移动网络）
│   └── 延迟与卡顿处理
├── 视野管理挑战
│   ├── 视野内对象筛选
│   ├── 流量洪峰（视野切换时）
│   └── 优先级裁减
└── 服务器架构挑战
    ├── 单核无法承载
    ├── 多线程/多进程拆分
    └── 开发效率 vs 性能的平衡
```

---

## 第二部分：寻路系统的优化

寻路是行军系统的核心。在 SLG 大世界中，寻路面临的挑战比传统游戏大得多。

### 2.1 六边形网格 vs 正方形网格

SLG 大世界通常采用六边形网格（Hex Grid），这是因为：

**六边形网格的优势**：
- 6 个方向等距移动（正方形网格对角线距离不一致）
- 更自然的空间划分
- 与 SLG 地格占用的玩法天然契合

**但带来的挑战**：
- A\* 等传统寻路算法的启发式函数需要调整
- 六边形坐标系统更复杂（立方坐标 / 轴向坐标 / 偏移坐标）

```
六边形网格坐标系统（立方坐标）：

           (0,-1,+1)  (+1,-1,0)
               / \     / \
              /   \   /   \
    (-1,0,+1) \   / \   / (+1,0,-1)
               \ /   \ /
    (-1,+1,0) --+-- (0,0,0) --+-- (+1,+1,-1)
               / \   / \
    (0,+1,-1) \   / \   / (0,-1,+1)
               \ /   \ /
                +-----+
```

### 2.2 寻路架构：前端还是后端？

这是 SLG 行军寻路最核心的架构决策。主要考虑因素：

**方案一：纯前端寻路**

```csharp
// 客户端计算路径，发送给服务器校验
public class ClientPathfinding
{
    List<HexCell> FindPath(HexCell start, HexCell end)
    {
        // 使用 A* 在客户端计算
        var path = AStarPathfinding.Find(start, end);
        // 发送路径给服务器
        SendPathToServer(path);
        return path;
    }
}
```

**问题**：
- 远距离寻路需要完整地图数据，客户端内存压力大
- 动态阻挡（其他玩家主城、公会关系）需要实时同步，数据量大
- 客户端 CPU 和发热问题严重

**方案二：纯后端寻路**

```go
// 服务器计算路径，返回给客户端
func (s *PathfindingService) FindPath(req *PathRequest) (*PathResponse, error) {
    // 服务器拥有完整静态+动态数据
    path := s.engine.FindPath(req.Start, req.End, req.PlayerId)
    return &PathResponse{Path: path}, nil
}
```

**问题**：
- 寻路是 CPU 密集型操作，高并发下服务器压力大
- 复杂地形下长距离寻路 TPS 通常只有几百

**方案三（推荐）：混合架构**

实际项目中，往往采用混合方案：

```
前端寻路（短距离/预览）  +  后端寻路（长距离/权威）
         │                        │
         ▼                        ▼
   客户端展示路径            服务器计算权威路径
         │                        │
         └────────┬───────────────┘
                  ▼
         路径校验 + 行军模拟
```

**关键设计**：

1. **客户端预计算**：玩家点击目标时，客户端先用 A\* 快速预览路径（用于 UI 展示）
2. **服务器权威计算**：确认行军后，服务器使用相同算法计算权威路径
3. **路径校验**：服务器校验客户端路径是否合法（防止作弊）
4. **行军模拟**：服务器按路径和时间戳模拟行军位置，客户端做插值表现

### 2.3 长距离寻路优化：分层寻路（Hierarchical Pathfinding）

对于跨州郡的长距离行军（路径可达数千格），直接使用 A\* 是不可行的。分层寻路是业界标准方案。

**核心思想**：将地图抽象为多个层次

```
层次结构：

Level 2 (大区域层): 州 → 郡 → 县
    ┌───────┬───────┐
    │ 州 A  │ 州 B  │
    │   ────┼───    │
    │ 州 C  │ 州 D  │
    └───────┴───────┘

Level 1 (关隘层): 区域之间的连接点
    州A ──[关隘A]── 州B
    州A ──[码头B]── 州C

Level 0 (地格层): 精细的六边形网格
    每个格子是一个节点
```

**算法流程**：

```
1. 高层抽象寻路（Level 2）
   - 将起点和目标点映射到对应的大区域
   - 在大区域图上用 A* 找到区域序列
   - 得到：州A → 关隘X → 州B → 码头Y → 州C

2. 中层关隘寻路（Level 1）
   - 在每个区域内部，找到到达关隘的路径
   - 区域入口 → 关隘 → 区域出口

3. 低层精细寻路（Level 0）
   - 仅在起点和终点所在的局部区域做精细寻路
   - 起点 → 区域出口 （短距离）
   - 区域入口 → 终点 （短距离）

4. 路径拼接
   - 将三段路径拼接为完整路径
```

**伪代码实现**：

```python
class HierarchicalPathfinder:
    def __init__(self, map_data):
        self.region_graph = self.build_region_graph()  # 大区域图
        self.gateway_graph = self.build_gateway_graph()  # 关隘图
        self.local_grid = map_data  # 精细网格

    def find_path(self, start, end):
        # Step 1: 高层抽象
        start_region = self.get_region(start)
        end_region = self.get_region(end)

        if start_region == end_region:
            # 同区域，直接精细寻路
            return self.local_astar(start, end)

        region_path = self.region_astar(start_region, end_region)

        # Step 2: 拼接路径
        full_path = []
        current_pos = start

        for i, region in enumerate(region_path):
            if i == len(region_path) - 1:
                # 最后一个区域，寻路到终点
                entry = self.get_region_entry(region, region_path[i-1])
                segment = self.local_astar(entry, end)
            else:
                # 找到去下一个区域的出口
                exit_gateway = self.get_gateway(region, region_path[i+1])
                segment = self.local_astar(current_pos, exit_gateway)
                current_pos = exit_gateway

            full_path.extend(segment)

        return full_path
```

**性能对比**：

| 方法 | 路径长度 | 搜索节点数 | 耗时 |
|------|---------|-----------|------|
| 直接 A\* | 2000 格 | ~500,000 | 2000ms |
| 分层寻路 | 2000 格 | ~5,000 | 20ms |
| **性能提升** | - | **100x** | **100x** |

### 2.4 动态阻挡处理

SLG 大世界的阻挡不是静态的——玩家的主城、要塞、联盟关系都会影响通行。

**核心挑战**：
- 阻挡是**动态变化**的（玩家建造/拆除建筑、公会关系变化）
- 阻挡有**关联关系**（同公会可通行，敌对不可通行）
- 阻挡变化后，已发出的行军需要**重新寻路**

**解决方案：分离静态数据与动态数据**

```go
// 数据结构设计
type MapData struct {
    // 静态数据（不变，可以预加载到内存）
    StaticObstacles  []HexCoord   // 山、水等永久障碍
    GatewayGraph     *GatewayGraph // 关隘、码头等固定连接点

    // 动态数据（频繁变化）
    DynamicObstacles sync.Map     // 玩家建筑等动态阻挡
    AllianceRelations map[int64]int64 // 公会关系
}

// 阻挡查询时合并静态和动态数据
func (m *MapData) IsBlocked(coord HexCoord, playerID int64) bool {
    // 静态阻挡
    if m.StaticObstacles.Contains(coord) {
        return true
    }

    // 动态阻挡 - 检查是否有玩家建筑
    if building, ok := m.DynamicObstacles.Load(coord); ok {
        b := building.(*Building)
        // 同公会/联盟可通行
        if m.AllianceRelations[playerID] == b.AllianceID {
            return false
        }
        return true
    }

    return false
}
```

**路径重算策略**：

```
行军途中遇到新建阻挡：
┌──────────────────────────────────────────┐
│  原路径: A ──→ B ──→ C ──→ D ──→ E      │
│                     ↑                    │
│              在此处出现新阻挡             │
│                                          │
│  处理策略：                               │
│  1. 从当前位置到 C 的路径不变             │
│  2. 从 C 开始重新寻路到终点 E             │
│  3. 拼接新路径：A→B→C→F→G→E              │
│  4. 通知客户端路径更新                    │
└──────────────────────────────────────────┘
```

### 2.5 寻路算法选择：A\* vs JPS vs 预计算

| 算法 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **A\*** | 通用场景 | 保证最优，实现简单 | 大空间搜索慢 |
| **JPS（Jump Point Search）** | 均匀网格，局部寻路 | 搜索节点少，速度快 | 仅适用于均匀代价网格 |
| **预计算路径表** | 长距离行军 | O(1) 查询，极快 | 内存占用大，不支持动态阻挡 |
| **双向 A\*** | 已知起点终点 | 搜索空间减半 | 需要两个方向都可达 |

**实际项目中的组合策略**：

```
短距离寻路（< 100 格）: JPS / A*
中距离寻路（100-500 格）: 双向 A*
长距离寻路（> 500 格）: 分层寻路 + 预计算关键路径
```

---

## 第三部分：行军网络同步

行军同步是 SLG 大世界中最复杂的网络问题之一。当玩家视野中有数百个行军单位同时移动时，如何高效同步？

### 3.1 行军数据模型

**设计原则**：行军本身是轻量的事件，不应与部队数据耦合。

```go
// 行军数据结构（服务器端）
type MarchEvent struct {
    MarchID     int64          // 行军唯一 ID
    OwnerID     int64          // 所属玩家
    ArmyID      int64          // 关联的部队 ID
    Path        []HexCoord     // 完整路径（坐标序列）
    StartTime   time.Time      // 出发时间戳
    Speed       float64        // 行军速度（格/秒）
    TargetType  MarchTargetType // 目标类型：攻击/采集/回城
    TargetID    int64          // 目标 ID
    CurrentPos  HexCoord       // 当前位置（服务器权威）
    State       MarchState     // 行军/战斗/返回/完成
}

// 行军状态
type MarchState int
const (
    MarchMoving   MarchState = iota // 正在行军
    MarchFighting                    // 战斗中
    MarchReturning                   // 返回中
    MarchCompleted                   // 已完成
)
```

**关键设计：行军与部队分离**

```
行军 (March)  ←→  部队 (Army)
  轻量对象           重量对象
  - 路径              - 英雄/士兵
  - 时间戳            - 属性/装备
  - 速度              - 技能
  - 状态              - 战斗数据

到达目的地后，March 对象被删除
Army 对象进入战斗/采集状态
```

### 3.2 位置同步：服务器权威 + 客户端预测

SLG 行军对实时性要求不如 FPS/MOBA 那么高，但仍需要流畅的视觉表现。

**方案：时间戳驱动的线性插值**

```
服务器端逻辑：
┌─────────────────────────────────────────────────────┐
│ 1. 行军开始时，服务器计算完整路径和时间表           │
│    Path: [A, B, C, D, E, F]                        │
│    Timeline:                                        │
│      t=0s    到达 A (起点)                          │
│      t=5s    到达 B                                │
│      t=12s   到达 C                                │
│      t=18s   到达 D                                │
│      t=25s   到达 E                                │
│      t=30s   到达 F (终点)                          │
│                                                     │
│ 2. 服务器只同步：                                    │
│    - 当前路径段索引                                 │
│    - 段内进度 (0.0 ~ 1.0)                           │
│    - 行军速度                                        │
│                                                     │
│ 3. 客户端根据时间戳推算当前位置：                    │
│    currentPos = lerp(path[i], path[i+1], progress)  │
└─────────────────────────────────────────────────────┘
```

**客户端插值实现**：

```csharp
public class MarchVisualizer : MonoBehaviour
{
    private MarchData marchData;
    private float lastSyncTime;

    // 服务器同步的数据
    public void OnServerSync(int segmentIndex, float segmentProgress, long serverTimestamp)
    {
        marchData.ServerSegmentIndex = segmentIndex;
        marchData.ServerSegmentProgress = segmentProgress;
        marchData.ServerTimestamp = serverTimestamp;
        lastSyncTime = Time.time;
    }

    // 客户端每帧插值
    void Update()
    {
        // 根据服务器时间戳推算当前应该在哪
        float elapsedSinceSync = Time.time - lastSyncTime;
        float predictedProgress = marchData.ServerSegmentProgress
            + (elapsedSinceSync * marchData.Speed / marchData.SegmentLength);

        // 平滑插值到预测位置
        int currentSegment = marchData.ServerSegmentIndex;
        Vector3 targetPos = Vector3.Lerp(
            marchData.Path[currentSegment],
            marchData.Path[currentSegment + 1],
            Mathf.Clamp01(predictedProgress)
        );

        // 平滑移动
        transform.position = Vector3.Lerp(
            transform.position,
            targetPos,
            Time.deltaTime * 10f  // 平滑系数
        );
    }
}
```

### 3.3 同步频率优化

**不要每帧同步！** 行军的同步频率应该根据实际需求动态调整：

```go
// 自适应同步频率
type SyncStrategy struct {
    BaseInterval    time.Duration  // 基础同步间隔（如 2 秒）
    MinInterval     time.Duration  // 最小同步间隔（如 0.5 秒）
    MaxInterval     time.Duration  // 最大同步间隔（如 5 秒）
}

func (s *SyncStrategy) GetSyncInterval(distance float64, isInView bool) time.Duration {
    if !isInView {
        // 不在视野内，降低同步频率
        return s.MaxInterval
    }

    if distance < 100 {
        // 近距离，提高同步频率
        return s.MinInterval
    }

    // 距离越远，同步频率越低
    return s.BaseInterval * time.Duration(distance/100)
}
```

**同步数据最小化**：

```
传统同步方式（每个部队每 2 秒）：
  FullSync: { marchId, path, speed, position, state, armyData, ... }
  大小: ~500 bytes/次
  1000 个部队: 500KB/次 × 30次/分钟 = 15MB/分钟

优化后同步方式：
  DeltaSync: { marchId, segmentIndex, progress }
  大小: ~20 bytes/次
  1000 个部队: 20KB/次 × 30次/分钟 = 600KB/分钟

带宽节省: 96%
```

---

## 第四部分：视野管理优化

视野管理是 SLG 大世界流量优化的核心。当玩家视野中有上千个对象时，不可能全部同步。

### 4.1 传统九宫格的问题

```
传统九宫格视野管理：

┌───┬───┬───┐
│ 1 │ 2 │ 3 │   玩家在中心格子 5
├───┼───┼───┤   同步格子 1-9 内的所有对象
│ 4 │ 5 │ 6 │
├───┼───┼───┤   问题1: 玩家跨越格子时出现流量洪峰
│ 7 │ 8 │ 9 │   问题2: 至少 60% 的同步对象玩家看不到
└───┴───┴───┘
```

**流量洪峰问题**：

```
玩家从 A 移动到 B：

A 的九宫格: [1,2,3,4,5,6,7,8,9]
B 的九宫格: [3,6,9,10,11,12,13,14,15]

退出视野: 格子 1,2,4,5,7,8  (6个格子)
进入视野: 格子 10,11,12,13,14,15 (6个格子)

瞬间需要处理 12 个格子的对象变化！
→ 可能导致客户端卡顿甚至卡死
```

### 4.2 精确视野裁减

《重返帝国》的解决方案：**基于真实视锥体的精确裁减**

```
精确视野裁减流程：

1. 客户端上报真实梯形视野范围
        ┌─────────────┐
       /│             │\
      / │   可见区域   │ \
     /  │   (梯形)    │  \
    └───┴─────────────┴───┘

2. 服务器用梯形范围初筛格子
   - 计算梯形内包含哪些格子
   - 这些格子内的对象 = 潜在可见对象

3. 精确匹配
   - 对每个潜在对象，检查是否在梯形内
   - 仅同步真正可见的对象

效果：
- 流量浪费从 60% 降至接近 0%
- 所有同步对象都是客户端可见的
```

### 4.3 优先级队列裁减

即使做了精确裁减，大规模团战时视野内仍有大量对象。需要进一步裁减。

```go
// 优先级裁减系统
type PriorityCulling struct {
    MaxObjects int  // 客户端上报的最大承载对象数
}

type ObjectPriority int
const (
    PriorityCritical  ObjectPriority = 0  // 自身部队、当前目标
    PriorityHigh      ObjectPriority = 1  // 友军、附近敌军
    PriorityMedium    ObjectPriority = 2  // 视野内其他部队
    PriorityLow       ObjectPriority = 3  // 资源点、装饰物
    PriorityNone      ObjectPriority = 4  // 可裁剪
)

func (c *PriorityCulling) Cull(objects []*WorldObject, playerID int64) []*WorldObject {
    // 1. 计算每个对象的优先级
    for _, obj := range objects {
        obj.Priority = c.calculatePriority(obj, playerID)
    }

    // 2. 按优先级排序
    sort.Slice(objects, func(i, j int) bool {
        return objects[i].Priority < objects[j].Priority
    })

    // 3. 裁减超出上限的对象
    if len(objects) > c.MaxObjects {
        return objects[:c.MaxObjects]
    }
    return objects
}

func (c *PriorityCulling) calculatePriority(obj *WorldObject, playerID int64) ObjectPriority {
    if obj.OwnerID == playerID {
        return PriorityCritical  // 自己的部队
    }
    if obj.TargetID == playerID {
        return PriorityCritical  // 正在攻击自己的部队
    }
    if c.isAllianceMember(obj.OwnerID, playerID) {
        return PriorityHigh  // 友军
    }
    if c.isNearby(obj, playerID, 50) {
        return PriorityHigh  // 附近敌军
    }
    if obj.Type == ObjectTypeArmy {
        return PriorityMedium
    }
    return PriorityLow
}
```

### 4.4 无极缩放中的数据分层（LOD 同步）

《重返帝国》支持无极缩放，玩家可以从地表层（微操）缩放到国家层（战略）。不同层级需要的对象数据不同。

```
无极缩放层级：

Layer 1 (地表层) ──── 完整数据
  ├── 部队位置、兵力、英雄详情
  ├── 战斗动画、技能特效
  └── 精确坐标

Layer 2 (战场层) ──── 中等数据
  ├── 部队位置、兵力摘要
  └── 战斗状态

Layer 3 (国家层) ──── 核心数据
  ├── 部队位置（概略）
  └── 归属信息
```

**标签化属性同步**：

```go
// 属性定义时打标签
type ArmyAttributes struct {
    // +layer1 +layer2 +layer3
    Position    HexCoord   `sync:"layer1,layer2,layer3"`

    // +layer1 +layer2
    TroopCount  int32      `sync:"layer1,layer2"`

    // +layer1
    HeroDetails *HeroData  `sync:"layer1"`

    // +layer1
    SkillEffects []*Effect `sync:"layer1"`
}

// 根据当前缩放层级，只同步对应标签的属性
func (s *SyncManager) SyncArmy(army *Army, viewerLayer int) {
    tags := getLayerTags(viewerLayer)
    delta := s.collectDirty(army, tags)  // 只收集对应标签的脏数据
    s.sendToClient(viewerID, delta)
}
```

**关键优化：层级切换时不重复同步**

```
玩家从 Layer 3 缩放到 Layer 2：
  ❌ 错误做法：同步 Layer 2 的完整数据
  ✅ 正确做法：仅同步 Layer 2 特有的数据（客户端已有 Layer 3 数据）

  客户端数据合并：
  Layer3 已有数据 + Layer2 特有增量 = Layer2 完整数据
  完全不重复同步！
```

---

## 第五部分：服务器架构优化

大世界行军的性能压力最终落在服务器端。合理的架构设计是性能的基础。

### 5.1 两种架构方案对比

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **Zoning（空间分区）** | 将地图按空间划分为多个区域，分别调度 | 理论上支持无限大地图 | 全异步编程，开发成本高 |
| **Offloading（功能拆分）** | 将独立功能拆分到独立线程 | 实现简单，开发效率高 | 承载有上限 |

**《重返帝国》的选择：Offloading**

在综合考虑开发效率和成本后，选择了 Offloading 方案。

### 5.2 线程拆分

```
大地图线程拆分架构：

┌────────────────────────────────────────────┐
│                  主线程                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 请求处理  │  │ 定时器   │  │ 状态管理  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  视野线程    │ │  寻路线程    │ │  并行定时器  │
│  (×1)       │ │  (×4)       │ │  (×N)       │
│             │ │             │ │             │
│ - 视野裁减  │ │ - 路径计算  │ │ - 战斗计算  │
│ - 优先级排序│ │ - 路径校验  │ │ - AI 逻辑   │
│ - 事件裁剪  │ │ - 路径重算  │ │ - 结构化并发 │
└─────────────┘ └─────────────┘ └─────────────┘
```

**视野线程**：
- 将视野管理从主线程拆分出来
- 做精确视野裁减、优先级队列裁减、技能事件裁减
- 拆分后不再受性能瓶颈限制，可以做到更精细的视野控制

**寻路线程**：
- 4 个独立寻路线程并行处理寻路请求
- 所有寻路请求异步化
- 不再需要牺牲寻路品质来换取性能

### 5.3 结构化并发

对于与大地图逻辑强耦合的模块（战斗、AI），无法拆分成独立线程，但可以通过**结构化并发**来利用多核：

```go
// 结构化并发 - 并行定时器
type ParallelTimer struct {
    Name     string
    Tasks    []func()  // 并行阶段的任务
    Finalize func()    // 串行收尾
}

func (pt *ParallelTimer) Execute() {
    // 阶段 1: 并行执行
    var wg sync.WaitGroup
    for _, task := range pt.Tasks {
        wg.Add(1)
        go func(t func()) {
            defer wg.Done()
            t()
        }(task)
    }
    wg.Wait()

    // 阶段 2: 串行收尾
    pt.Finalize()
}

// 使用示例：战斗结算
var battleTimer = ParallelTimer{
    Name: "BattleSettlement",
    Tasks: []func(){
        func() { processPlayerABattles() },  // 并行
        func() { processPlayerBBattles() },  // 并行
        func() { processPlayerCBattles() },  // 并行
    },
    Finalize: func() {
        // 串行合并战斗结果
        mergeBattleResults()
        notifyClients()
    },
}
```

**关键设计原则**：
- 并行阶段的任务之间**无资源竞争**
- 开发者只需保证单个处理函数内的线程安全
- 多线程范围被限定在单个处理函数内

### 5.4 行军数据的 Block 管理

一种经典的 SLG 行军数据管理方案——基于 Block 的数据组织：

```go
// 将地图划分为 Block
type BlockCoord struct {
    X, Y int32
}

type BlockManager struct {
    // 三个核心映射
    Player2Blocks  map[int64]map[BlockCoord]bool  // 玩家 → 视口覆盖的 Block
    Block2Players  map[BlockCoord]map[int64]bool  // Block → 关注该 Block 的玩家
    Block2Events   map[BlockCoord][]*MarchEvent   // Block → 该 Block 上的行军事件
}

// 行军创建时：挂载到途经的所有 Block
func (bm *BlockManager) RegisterMarch(march *MarchEvent) {
    for _, coord := range march.Path {
        block := bm.coordToBlock(coord)
        bm.Block2Events[block] = append(bm.Block2Events[block], march)
    }
}

// 玩家视口变化时：通知相关 Block
func (bm *BlockManager) OnPlayerViewChange(playerID int64, newBlocks []BlockCoord) {
    oldBlocks := bm.Player2Blocks[playerID]

    // 进入新 Block
    for _, block := range newBlocks {
        if !oldBlocks[block] {
            bm.Block2Players[block][playerID] = true
            // 推送该 Block 的现有行军事件
            bm.pushEvents(playerID, bm.Block2Events[block])
        }
    }

    // 离开旧 Block
    for block := range oldBlocks {
        if !contains(newBlocks, block) {
            delete(bm.Block2Players[block], playerID)
        }
    }

    bm.Player2Blocks[playerID] = newBlocks
}
```

**优势**：
- 行军数据与静态地图数据分离，轻量化
- 基于 Block 的增量同步，避免全量推送
- 支持多视口（玩家可以同时关注多个区域）

---

## 第六部分：客户端表现优化

### 6.1 行军动画的 LOD 策略

```csharp
public enum MarchLODLevel
{
    High,    // 近距离：完整模型 + 动画
    Medium,  // 中距离：简化模型 + 简化动画
    Low,     // 远距离：图标 + 行军线
    Culled   // 视野外：不渲染
}

public class MarchLODManager : MonoBehaviour
{
    public float HighDistance = 50f;
    public float MediumDistance = 200f;
    public float LowDistance = 500f;

    void Update()
    {
        foreach (var march in activeMarches)
        {
            float distance = Vector3.Distance(Camera.main.transform.position, march.Position);
            MarchLODLevel level = CalculateLOD(distance);

            switch (level)
            {
                case MarchLODLevel.High:
                    march.ShowFullModel();
                    march.ShowDetailedAnimation();
                    break;
                case MarchLODLevel.Medium:
                    march.ShowSimplifiedModel();
                    march.ShowBasicAnimation();
                    break;
                case MarchLODLevel.Low:
                    march.ShowIcon();
                    march.ShowMarchLine();
                    break;
                case MarchLODLevel.Culled:
                    march.Hide();
                    break;
            }
        }
    }
}
```

### 6.2 对象池

行军对象频繁创建和销毁，必须使用对象池：

```csharp
public class MarchObjectPool : MonoBehaviour
{
    private Queue<MarchVisual> pool = new Queue<MarchVisual>();
    private MarchVisual prefab;

    public MarchVisual Get()
    {
        MarchVisual obj;
        if (pool.Count > 0)
        {
            obj = pool.Dequeue();
            obj.gameObject.SetActive(true);
        }
        else
        {
            obj = Instantiate(prefab);
        }
        return obj;
    }

    public void Return(MarchVisual obj)
    {
        obj.gameObject.SetActive(false);
        obj.Reset();
        pool.Enqueue(obj);
    }
}
```

### 6.3 行军线渲染优化

大量行军线同时渲染对 GPU 压力很大。优化策略：

```
行军线渲染优化：

1. 合并绘制：将多条行军线合并为一次 Draw Call
   - 使用 LineRenderer 的 GPU Instancing
   - 或使用自定义 Shader + ComputeBuffer

2. 距离裁剪：超过一定距离不渲染行军线
   - 近距离（< 100m）：显示完整行军线
   - 中距离（100-500m）：只显示已走过的线段
   - 远距离（> 500m）：不显示行军线

3. 简化路径点：远距离时减少行军线的顶点数
   - 原始路径可能有 1000 个点
   - 远距离时简化为 20 个点即可
```

---

## 第七部分：各游戏的行军系统特点对比

### 7.1 无尽冬日（Whiteout Survival）

- **冰雪主题大世界**：寒冰地形影响行军速度
- **联盟协同**：行军路线可以经过盟友领地加速
- **资源采集**：行军到资源点采集，采集完成后自动返回
- **野兽狩猎**：行军到野兽位置进行战斗

### 7.2 Dark Survival

- **黑暗生存主题**：视野受限，行军需要探索
- **夜战机制**：夜间行军有特殊规则
- **资源争夺**：多个玩家同时行军到同一资源点时的冲突处理

### 7.3 重返帝国

- **自由行军**：玩家可以任意拖动部队，不需要走格子
- **持续战斗**：战斗在大地图上实时计算，可以随时撤退
- **无极缩放**：支持从地表层到国家层的无缝缩放
- **千人同屏**：皇城战 3000+ 部队同时在场

| 特性 | 无尽冬日 | Dark Survival | 重返帝国 |
|------|---------|---------------|---------|
| 网格类型 | 六边形 | 六边形 | 连续自由移动 |
| 行军方式 | 点击目标行军 | 点击目标行军 | 自由拖动行军 |
| 战斗方式 | 回合/即时 | 回合/即时 | 实时持续战斗 |
| 视野缩放 | 固定层级 | 固定层级 | 无极缩放 |
| 团战规模 | 中 | 中 | 超大（千人同屏） |

---

## 总结

大世界行军系统看似简单，实则是 SLG 游戏中最复杂的系统之一。优化的核心思路可以总结为：

1. **寻路层面**：分层寻路 + 预计算 + 动态阻挡增量更新，100x 性能提升
2. **网络层面**：时间戳驱动插值 + 自适应同步频率 + delta 同步，96% 带宽节省
3. **视野层面**：精确视锥裁减 + 优先级队列 + LOD 属性同步，60%+ 流量浪费消除
4. **架构层面**：Offloading 线程拆分 + 结构化并发 + Block 数据管理

**核心设计原则**：

```
┌──────────────────────────────────────────────┐
│                                              │
│  1. 只同步客户端需要的                          │
│     - 精确视野裁减                            │
│     - 优先级裁减                              │
│     - LOD 属性同步                            │
│                                              │
│  2. 只同步变化的数据                            │
│     - 字段级增量同步                          │
│     - 行军位置用时间戳推算                      │
│     - 技能事件裁剪                            │
│                                              │
│  3. 计算密集型任务异步化                         │
│     - 寻路拆分为独立线程                       │
│     - 视野管理拆分为独立线程                    │
│     - 战斗/AI 使用结构化并发                    │
│                                              │
│  4. 客户端做表现，服务器做权威                    │
│     - 客户端插值/预测                          │
│     - 服务器校验/计算                          │
│     - 路径一致，表现分离                        │
│                                              │
└──────────────────────────────────────────────┘
```

---

## 参考资料

1. [天美干货分享：怎么解决大地图SLG的技术痛点？](https://news.qq.com/rain/a/20230228A09HJ700) —— IGDC 国际游戏开发者大会，《重返帝国》大地图基础架构负责人周元军分享
2. [手游 SLG 寻路是如何实现的（一）](https://juejin.cn/post/7053647818971414541) —— 超大型地图寻路在服务器上的实现
3. [开发笔记：SLG 地图](http://wudaijun.com/2015/11/erlang-server-design4-slg-map) —— 行军数据组织与 Block 管理
4. [Unity 手游实战：从 0 开始 SLG](https://www.cnblogs.com/dyf214/p/13020605.html) —— 客户端技术选型
5. [预运算地图寻路的一种方法](https://blog.codingnow.com/2021/01/path_map.html) —— 云风博客，分层寻路思路
6. [SLG 游戏服务器开发手册：常用算法与数据结构](https://zhuanlan.zhihu.com/p/1941087140550283764) —— 六边形坐标、JPS 等