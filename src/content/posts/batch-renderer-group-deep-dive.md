---
title: 深入理解 BatchRendererGroup：从渲染批处理的前世今生说起
published: 2026-03-13
description: "从早期的 Draw Call 优化到 BatchRendererGroup，全面解析渲染批处理技术的演进与优化原理。"
image: "/images/ai-industry-2025-header.png"
tags: ["Unity", "图形渲染", "GPU优化", "BatchRendererGroup", "SRP", "DOTS", "性能优化"]
category: 图形编程
draft: false
---

# 深入理解 BatchRendererGroup：从渲染批处理的前世今生说起

## 引言

如果你曾在 Unity 中尝试渲染成千上万个相同的物体——无论是草地、粒子效果还是小行星带——你一定会遇到一个核心问题：**Draw Call 过多导致的性能瓶颈**。

BatchRendererGroup（BRG）是 Unity 在 2022.1 版本中引入的底层渲染 API，它代表了 Unity 渲染优化技术的最新阶段。但要真正理解它解决了什么问题，我们需要从渲染批处理的历史说起。

本文将从以下几个方面进行事无巨细的讲解：

1. **渲染批处理的历史演变**：从最原始的 Draw Call 到 BRG 的完整技术演进
2. **每种技术的核心原理与局限性**
3. **BatchRendererGroup 的设计哲学与实现细节**
4. **结合 OpenGL Instancing 的底层原理**
5. **实际性能对比与最佳实践**

---

## 第一部分：渲染批处理的历史演变

### 1.1 问题的根源：Draw Call

在图形渲染中，**Draw Call**（绘制调用）是指 CPU 向 GPU 发送渲染命令的过程。每次调用 `glDrawArrays` 或 `glDrawElements`（在 OpenGL 中）或 `DrawInstanced`（在 DirectX 中），都会触发一系列操作：

```
CPU                          GPU
  |                            |
  |---- 设置顶点缓冲区 -------->|
  |---- 设置材质参数 ---------->|
  |---- 设置变换矩阵 ---------->|
  |---- 发送绘制命令 --------->|  ← Draw Call
  |                            |---- 执行顶点着色器
  |                            |---- 执行片元着色器
  |                            |---- 输出到帧缓冲
```

**问题在于**：CPU 和 GPU 之间的通信开销非常大。根据 LearnOpenGL 的经典论述：

> "告诉 GPU 渲染你的顶点数据通过 glDrawArrays 或 glDrawElements 这样的函数会消耗相当大的性能，因为 OpenGL 必须在进行必要的准备工作之后才能绘制顶点数据（比如告诉 GPU 从哪个缓冲区读取数据，在哪里找到顶点属性，所有这些都通过相对较慢的 CPU 到 GPU 总线传输）。所以即使渲染顶点本身非常快，给 GPU 发送渲染命令的过程却不快。"

这个问题的严重性可以用数据说明：

| 场景 | Draw Calls | 帧率影响 |
|------|-----------|---------|
| 100 个独立物体 | 100 | 轻微 |
| 10,000 个独立物体 | 10,000 | 显著下降 |
| 100,000 个独立物体 | 100,000 | 可能无法运行 |

### 1.2 第一阶段：静态批处理（Static Batching）

**出现时间**：Unity 早期版本

**核心原理**：
静态批处理是最简单的优化方式。对于标记为 "Static" 的 GameObject，Unity 在构建时或运行时将它们的网格数据合并成一个大的网格。

```
原始状态：                          批处理后：
[Mesh A] [Mesh B] [Mesh C]    →    [Combined Mesh: A+B+C]
   ↓         ↓         ↓                  ↓
Draw()   Draw()     Draw()           Draw() (一次调用)
```

**实现细节**：

```csharp
// 在 Unity 中，只需在 Inspector 中勾选 Static
// 或通过代码设置
gameObject.isStatic = true;
```

**内存布局变化**：

```
静态批处理前的顶点缓冲区：
Mesh A: [v0, v1, v2, v3]
Mesh B: [v4, v5, v6]
Mesh C: [v7, v8, v9, v10, v11]

静态批处理后的顶点缓冲区：
Combined: [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]
```

**优点**：
- 实现简单，开发者几乎无需修改代码
- 对于不移动的物体效果极佳
- 可以合并使用相同材质的不同网格

**局限性**：
1. **内存开销增加**：每个实例都需要存储完整的顶点数据副本
2. **仅适用于静态物体**：一旦物体移动，批处理就会中断
3. **运行时开销**：动态合并网格需要 CPU 时间
4. **材质限制**：只有使用相同材质的物体才能合并

### 1.3 第二阶段：动态批处理（Dynamic Batching）

**出现时间**：Unity 早期版本（与静态批处理同期）

**核心原理**：
动态批处理在每帧运行时自动检测可以合并的小型网格，将它们的顶点数据临时合并然后绘制。

```
帧 N：
检测可合并物体 → 合并顶点 → 单次绘制

帧 N+1：
物体移动 → 重新检测 → 重新合并 → 单次绘制
```

**关键限制条件**：

1. **顶点数限制**：通常不超过 900 个顶点（可配置）
2. **缩放限制**：非均匀缩放会破坏批处理
3. **光照贴图限制**：使用光照贴图的物体不能合并
4. **材质限制**：必须使用相同材质实例

**Unity 内部实现伪代码**：

```csharp
void DynamicBatching()
{
    List<MeshRenderer> batchableRenderers = FindBatchableRenderers();
    
    // 检查是否满足批处理条件
    foreach (var renderer in batchableRenderers)
    {
        if (renderer.sharedMaterial != currentMaterial) continue;
        if (renderer.mesh.vertexCount > 900) continue;
        // ... 更多检查
    }
    
    // 动态合并顶点数据
    CombineInstance[] combine = new CombineInstance[batchableRenderers.Count];
    for (int i = 0; i < batchableRenderers.Count; i++)
    {
        combine[i].mesh = batchableRenderers[i].mesh;
        combine[i].transform = batchableRenderers[i].transform.localToWorldMatrix;
    }
    
    Mesh combinedMesh = new Mesh();
    combinedMesh.CombineMeshes(combine);
    Graphics.DrawMesh(combinedMesh, ...);
}
```

**优点**：
- 自动化，无需手动标记
- 支持移动的物体

**局限性**：
- 顶点数限制严格
- CPU 开销大（每帧都要检测和合并）
- 条件苛刻，很容易意外破坏批处理

### 1.4 第三阶段：GPU Instancing

**出现时间**：Unity 5.6+，OpenGL 3.1+，DirectX 10+

**核心原理**：
GPU Instancing 是一个革命性的变化。不同于批处理将网格合并，Instancing 让 GPU 能够用**一次 Draw Call** 渲染同一个网格的多个实例，每个实例可以有不同的属性（位置、旋转、颜色等）。

这是 LearnOpenGL 中描述的核心概念：

> "Instancing 是一种技术，我们可以用单次渲染调用绘制多个（相同的网格数据）对象，节省我们每次需要渲染对象时的 CPU 到 GPU 通信。"

**OpenGL 底层实现**：

```glsl
// 顶点着色器
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aOffset;  // 实例化属性

void main()
{
    gl_Position = vec4(aPos + aOffset, 0.0, 1.0);
}
```

```cpp
// C++ 端设置实例化数组
unsigned int instanceVBO;
glGenBuffers(1, &instanceVBO);
glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 100, &translations[0], GL_STATIC_DRAW);

// 关键：glVertexAttribDivisor
glEnableVertexAttribArray(2);
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
glVertexAttribDivisor(2, 1);  // 每个实例更新一次，而不是每个顶点

// 绘制 100 个实例
glDrawArraysInstanced(GL_TRIANGLES, 0, 6, 100);
```

**`glVertexAttribDivisor` 的工作原理**：

```
普通顶点属性（divisor = 0）：
顶点 0 → 属性 A
顶点 1 → 属性 B
顶点 2 → 属性 C
...

实例化属性（divisor = 1）：
实例 0 的所有顶点 → 属性 A
实例 1 的所有顶点 → 属性 B
实例 2 的所有顶点 → 属性 C
...
```

**Unity 中的使用**：

```csharp
MaterialPropertyBlock props = new MaterialPropertyBlock();
MeshRenderer renderer;

foreach (var obj in objects)
{
    props.SetColor("_Color", obj.color);
    props.SetMatrix("_ObjectToWorld", obj.transform.localToWorldMatrix);
    renderer.SetPropertyBlock(props);
}

// 使用 Graphics.DrawMeshInstanced
Graphics.DrawMeshInstanced(mesh, 0, material, matrices, count);
```

**性能对比**（LearnOpenGL 示例）：

| 方法 | 100,000 个小行星 |
|------|-----------------|
| 逐物体绘制 | 无法运行 (~1000 个就开始卡顿) |
| GPU Instancing | 流畅运行，仅 2 次 Draw Call |

**优点**：
- 极高的性能提升
- 内存效率高（只存储一份网格数据）
- 支持每实例属性变化

**局限性**：
1. **只能使用相同网格**：不同网格不能实例化
2. **材质变体限制**：不同材质需要分开绘制
3. **属性数量限制**：每实例数据量有限
4. **Unity API 限制**：`DrawMeshInstanced` 每次最多 1023 个实例

### 1.5 第四阶段：SRP Batcher

**出现时间**：Unity 2019.3+（URP/HDRP）

**核心原理**：
SRP Batcher 不是一个传统意义上的"批处理"技术。它**不会减少 Draw Call 的数量**，而是减少了 **Render State Changes**（渲染状态切换）的开销。

这是理解 SRP Batcher 的关键：

> "传统优化 Draw Call 的方法是减少它们的数量。相反，SRP Batcher 减少了 Draw Call 之间的渲染状态切换。" —— Unity 官方文档

**渲染状态切换的问题**：

```
传统渲染：
Draw(A) → 切换材质 → Draw(B) → 切换材质 → Draw(C)
         ↑ 开销大          ↑ 开销大          ↑ 开销大

SRP Batcher：
Bind(材质A) → Draw(A) → Draw(B) → Draw(C) → Unbind
             ↑ 仅一次状态切换
```

**技术实现**：

SRP Batcher 使用**持久化 GPU 缓冲区**（Persistent GPU Buffer）来存储材质属性。当材质属性变化时，只更新缓冲区，而不重新绑定整个材质。

```
CPU 内存：                    GPU 内存：
[Material A Props]    →     [Persistent Buffer]
[Material B Props]    →     [Persistent Buffer]
[Material C Props]    →     [Persistent Buffer]

渲染时：
GPU 直接从持久化缓冲区读取数据
不需要 CPU 每帧重新上传
```

**Shader 要求**：

```hlsl
// SRP Batcher 兼容的 Shader 必须使用 CBUFFER
CBUFFER_START(UnityPerDraw)
    float4x4 unity_ObjectToWorld;
    float4x4 unity_WorldToObject;
CBUFFER_END

CBUFFER_START(UnityPerMaterial)
    float4 _BaseColor;
    float _Metallic;
    float _Smoothness;
CBUFFER_END
```

**优点**：
- 大幅减少 CPU 渲染开销
- 与现有 GameObject 工作流兼容
- 无需手动管理实例化

**局限性**：
- 仅支持 SRP（URP/HDRP）
- 需要特定的 Shader 结构
- 不减少 Draw Call 数量（只是让它们更便宜）

---

## 第二部分：BatchRendererGroup 的诞生

### 2.1 为什么需要 BRG？

在 BRG 出现之前，Unity 的渲染优化面临着几个根本性问题：

**问题 1：GameObject 开销**

传统 Unity 渲染严重依赖 GameObject 和 Transform 组件。当场景中有数万个物体时：

```csharp
// 每帧的开销
foreach (var obj in gameObjects)  // 10,000+ 迭代
{
    obj.transform.position += velocity * Time.deltaTime;  // Transform 更新开销
    renderer.material.SetColor("_Color", obj.color);       // 材质属性设置开销
}

// Unity 内部还要处理：
// - Transform 层级更新
// - 消息广播
// - 序列化
// - 内存分配/GC
```

**问题 2：主线程瓶颈**

即使使用了 Instancing 或 SRP Batcher，渲染命令的准备仍然在主线程执行：

```
主线程：
PrepareRenderCommands() → 10ms+
    ↓
渲染线程：
ExecuteRenderCommands() → 2ms
```

**问题 3：缺乏对 DOTS 的原生支持**

DOTS（Data-Oriented Technology Stack）使用 ECS（Entity-Component-System）架构，与 GameObject 模型不兼容。需要一个能够直接渲染 Entity 的系统。

### 2.2 BRG 的设计目标

BatchRendererGroup 的设计目标是：

1. **零 GameObject 开销**：直接操作 GPU 数据
2. **数据导向设计**：与 DOTS 无缝集成
3. **多线程友好**：可以在 Job System 中并行处理
4. **精确控制**：开发者完全掌控渲染流程

### 2.3 BRG 的核心概念

#### 2.3.1 Draw Command（绘制命令）

Draw Command 是 BRG 的核心抽象，包含了创建一个优化过的实例化 Draw Call 所需的所有信息：

```csharp
public struct BatchDrawCommand
{
    public BatchID batchID;           // 批次标识
    public BatchMaterialID materialID; // 材质标识
    public BatchMeshID meshID;         // 网格标识
    public int subMeshIndex;           // 子网格索引
    public uint splitVisibilityMask;   // 可见性掩码
    public uint sortFlags;             // 排序标志
    public uint visibleCount;          // 可见实例数
    public uint instanceOffset;        // 实例偏移量
}
```

#### 2.3.2 Filter Settings（过滤设置）

Filter Settings 决定何时渲染实例：

```csharp
public struct BatchFilterSettings
{
    public RenderingLayerMask renderingLayerMask;  // 渲染层
    public uint layer;                              // 层级
    public int renderingLayerMask;                  // 渲染层掩码
    // 控制阴影、运动向量等的渲染
}
```

#### 2.3.3 Draw Range（绘制范围）

Draw Range 将过滤设置应用到一组连续的 Draw Command：

```csharp
public struct BatchDrawRange
{
    public BatchFilterSettings filterSettings;  // 过滤设置
    uint drawCommandsStart;                      // 起始命令索引
    uint drawCommandsCount;                      // 命令数量
}
```

### 2.4 数据布局：GPU 友好的内存结构

BRG 使用特定的数据布局来最大化 GPU 效率。理解这个布局是使用 BRG 的关键。

**内存布局图**：

```
GraphicsBuffer 布局：
┌─────────────────────────────────────────────────────────────┐
│ 偏移量 0-63: 64 字节的零（约定俗成，方便加载默认值）           │
├─────────────────────────────────────────────────────────────┤
│ 偏移量 64-95: 32 字节未初始化（对齐用）                       │
├─────────────────────────────────────────────────────────────┤
│ 偏移量 96+: unity_ObjectToWorld 数组                         │
│   每个实例 48 字节 (float3x4)                                │
│   实例 0: [c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, ...]   │
│   实例 1: [c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, ...]   │
│   ...                                                        │
├─────────────────────────────────────────────────────────────┤
│ unity_WorldToObject 数组（逆矩阵）                           │
│   每个实例 48 字节 (float3x4)                                │
└─────────────────────────────────────────────────────────────┘
```

**float3x4 与 Matrix4x4 的区别**：

这是使用 BRG 时最容易混淆的地方：

```csharp
// Matrix4x4 按行存储
Matrix4x4 matrix = new Matrix4x4(
    m00, m01, m02, m03,  // 第一行
    m10, m11, m12, m13,  // 第二行
    m20, m21, m22, m23,  // 第三行
    m30, m31, m32, m33   // 第四行
);

// float3x4 按列存储（BRG 要求的格式）
float3x4 packedMatrix = new float3x4(
    m00, m10, m20, m01,  // 第一列
    m11, m21, m02, m12,  // 第二列
    m22, m03, m13, m23   // 第三列
);
// 注意：最后一行 (m30, m31, m32, m33) 通常是 (0, 0, 0, 1)
```

**转换代码示例**：

```csharp
public static float3x4 ToFloat3x4(Matrix4x4 m)
{
    // 从按行存储转换为按列存储
    return new float3x4(
        m.m00, m.m10, m.m20, m.m01,
        m.m11, m.m21, m.m02, m.m12,
        m.m22, m.m03, m.m13, m.m23
    );
}
```

### 2.5 完整的 BRG 示例

以下是一个最小化的 BRG 渲染示例：

```csharp
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Mathematics;

public class SimpleBRGExample : MonoBehaviour
{
    [SerializeField] private Mesh _mesh;
    [SerializeField] private Material _material;
    [SerializeField] private uint _instanceCount = 1000;
    
    private BatchRendererGroup _brg;
    private GraphicsBuffer _instanceData;
    private BatchID _batchID;
    private BatchMaterialID _materialID;
    private BatchMeshID _meshID;
    
    private const int SizeOfPackedMatrix = 48; // sizeof(float3x4)
    private const int ExtraBytes = 96; // 64 zeroes + 32 uninitialized
    
    void Start()
    {
        InitializeBRG();
    }
    
    void InitializeBRG()
    {
        // 1. 创建 BatchRendererGroup
        _brg = new BatchRendererGroup(OnPerformCulling, IntPtr.Zero);
        
        // 2. 设置全局包围盒
        var bounds = new Bounds(Vector3.zero, new Vector3(10000, 10000, 10000));
        _brg.SetGlobalBounds(bounds);
        
        // 3. 注册网格和材质
        _meshID = _brg.RegisterMesh(_mesh);
        _materialID = _brg.RegisterMaterial(_material);
        
        // 4. 创建 GPU 缓冲区
        int bufferCount = BufferCountForInstances(SizeOfPackedMatrix, (int)_instanceCount, ExtraBytes);
        _instanceData = new GraphicsBuffer(GraphicsBuffer.Target.Raw, bufferCount, 4);
        
        // 5. 准备实例数据
        float3x4[] objectToWorld = new float3x4[_instanceCount];
        float3x4[] worldToObject = new float3x4[_instanceCount];
        
        for (int i = 0; i < _instanceCount; i++)
        {
            var pos = Random.insideUnitSphere * 50;
            var matrix = Matrix4x4.TRS(pos, Quaternion.identity, Vector3.one);
            objectToWorld[i] = ToFloat3x4(matrix);
            worldToObject[i] = ToFloat3x4(matrix.inverse);
        }
        
        // 6. 上传数据到 GPU
        uint byteAddressObjectToWorld = SizeOfPackedMatrix * 2;
        uint byteAddressWorldToObject = byteAddressObjectToWorld + SizeOfPackedMatrix * _instanceCount;
        
        _instanceData.SetData(new float4[16], 0, 0, 16); // 零填充
        _instanceData.SetData(objectToWorld, 0, (int)(byteAddressObjectToWorld / SizeOfPackedMatrix), objectToWorld.Length);
        _instanceData.SetData(worldToObject, 0, (int)(byteAddressWorldToObject / SizeOfPackedMatrix), worldToObject.Length);
        
        // 7. 设置元数据
        var metadata = new NativeArray<MetadataValue>(2, Allocator.Temp);
        metadata[0] = new MetadataValue
        {
            NameID = Shader.PropertyToID("unity_ObjectToWorld"),
            Value = 0x80000000 | byteAddressObjectToWorld  // 最高位表示数组
        };
        metadata[1] = new MetadataValue
        {
            NameID = Shader.PropertyToID("unity_WorldToObject"),
            Value = 0x80000000 | byteAddressWorldToObject
        };
        
        // 8. 添加批次
        _batchID = _brg.AddBatch(metadata, _instanceData.bufferHandle);
        
        metadata.Dispose();
    }
    
    // 剔除回调
    unsafe JobHandle OnPerformCulling(BatchRendererGroup rendererGroup, BatchCullingContext cullingContext, BatchCullingOutput cullingOutput, IntPtr userContext)
    {
        var drawCommands = new NativeArray<BatchDrawCommand>(1, Allocator.TempJob);
        var drawRanges = new NativeArray<BatchDrawRange>(1, Allocator.TempJob);
        var filteringSettings = new NativeArray<BatchFilteringSettings>(1, Allocator.TempJob);
        
        drawCommands[0] = new BatchDrawCommand
        {
            batchID = _batchID,
            materialID = _materialID,
            meshID = _meshID,
            subMeshIndex = 0,
            visibleCount = (uint)_instanceCount,
            instanceOffset = 0
        };
        
        // ... 设置 drawRanges 和 filteringSettings
        
        cullingOutput.drawCommands[0] = drawCommands;
        cullingOutput.drawRanges[0] = drawRanges;
        cullingOutput.filteringSettings[0] = filteringSettings;
        
        return default;
    }
    
    void OnDestroy()
    {
        _brg.Dispose();
        _instanceData?.Dispose();
    }
}
```

---

## 第三部分：BRG 与传统方法的性能对比

### 3.1 测试场景：Boids 模拟

以下是基于实际测试的性能数据（来源：gamedev.center）：

**测试配置**：
- 2000 个实例
- 每帧更新所有位置
- 使用 Job System + Burst

| 方法 | PlayerLoop 时间 | Update 时间 | 渲染时间 |
|------|----------------|-------------|----------|
| GameObjects | 15-20ms | 0.24ms | 12-15ms |
| Graphics.DrawMeshInstanced | 2-3ms | 0.5ms | 1-2ms |
| BRG（未优化） | 7-8ms | 5ms | 2ms |
| BRG（Job 优化） | 2-3ms | 0.3ms | 2ms |

### 3.2 分析

**GameObjects 方案的问题**：

```
PostLateUpdate.UpdateAllRenderers: 8-10ms  ← Transform 更新开销
Rendering: 4-5ms                          ← 每个对象独立绘制
```

**BRG 方案的优势**：

```
Update (并行 Job): 0.3ms                  ← 数据导向更新
Rendering: 2ms                            ← 单次 Draw Call
```

### 3.3 BRG 优化技巧

**技巧 1：使用 Job System 并行更新数据**

```csharp
[BurstCompile]
public unsafe struct CopyMatricesJob : IJobParallelFor
{
    [ReadOnly] public float4x4* Source;
    [WriteOnly] public NativeArray<Vector4> DataBuffer;
    public int Size;
    
    public void Execute(int index)
    {
        int offset = 4 + index * 3;
        DataBuffer[offset + 0] = new Vector4(Source[index].c0.x, Source[index].c0.y, Source[index].c0.z, Source[index].c1.x);
        DataBuffer[offset + 1] = new Vector4(Source[index].c1.y, Source[index].c1.z, Source[index].c2.x, Source[index].c2.y);
        DataBuffer[offset + 2] = new Vector4(Source[index].c2.z, Source[index].c3.x, Source[index].c3.y, Source[index].c3.z);
        
        // 逆矩阵...
    }
}
```

**技巧 2：避免每帧重新分配**

```csharp
// 错误：每帧分配
void Update()
{
    var matrices = new float3x4[count];  // GC 压力
    // ...
}

// 正确：预分配并复用
private NativeArray<float3x4> _matrices;

void Start()
{
    _matrices = new NativeArray<float3x4>(count, Allocator.Persistent);
}
```

**技巧 3：实现自定义剔除**

BRG 不提供自动的视锥剔除，需要手动实现：

```csharp
unsafe JobHandle OnPerformCulling(...)
{
    // 使用 Job System 并行剔除
    var cullingJob = new FrustumCullingJob
    {
        CameraPlanes = cullingContext.cullingPlanes,
        InstancePositions = _positions,
        VisibleIndices = _visibleIndices,
        VisibleCount = _visibleCount
    }.Schedule(_instanceCount, 64);
    
    return cullingJob;
}
```

---

## 第四部分：BRG 与 OpenGL Instancing 的对应关系

理解 BRG 与底层图形 API 的关系有助于深入掌握这项技术。

### 4.1 概念映射

| BRG 概念 | OpenGL 概念 |
|----------|-------------|
| `GraphicsBuffer` | `GL_SHADER_STORAGE_BUFFER` 或 `GL_UNIFORM_BUFFER` |
| `MetadataValue` | `glVertexAttribDivisor` + uniform location |
| `DrawCommand` | `glDrawElementsInstanced` |
| `visibleCount` | `instancecount` 参数 |

### 4.2 数据流对比

**OpenGL Instancing 数据流**：

```
CPU                           GPU
  |                             |
  |--- glBufferData ----------->|  上传实例数据
  |--- glVertexAttribPointer -->|  设置属性指针
  |--- glVertexAttribDivisor -->|  设置实例化频率
  |--- glDraw*Instanced ------->|  绘制
```

**BRG 数据流**：

```
C# 代码                        Unity 内部                GPU
  |                              |                         |
  |-- GraphicsBuffer.SetData --->|                         |
  |-- AddBatch(metadata) ------->|                         |
  |                              |-- 注册到 SRP ----------->|
  |                              |                         |
  |-- OnPerformCulling --------->|                         |
  |                              |-- 生成 Draw Commands --->|
  |                              |-- glDraw*Instanced ----->|
```

### 4.3 Shader 差异

**传统 Instancing Shader**：

```glsl
#version 330 core
layout (location = 3) in mat4 instanceMatrix;

void main()
{
    gl_Position = projection * view * instanceMatrix * vec4(aPos, 1.0);
}
```

**BRG 兼容 Shader**：

```hlsl
// Unity 需要使用 DOTS_INSTANCING_ON
#ifdef UNITY_DOTS_INSTANCING_ENABLED
    UNITY_DOTS_INSTANCING_START(float4x4, unity_ObjectToWorld)
        UNITY_DOTS_INSTANCING_END(float4x4)
#endif

void main()
{
    // BRG 通过元数据自动填充这些值
    float4x4 objectToWorld = UNITY_DOTS_INSTANCED_PROP(float4x4, unity_ObjectToWorld);
    gl_Position = VP * objectToWorld * float4(positionOS, 1.0);
}
```

---

## 第五部分：BRG 的适用场景与选择指南

### 5.1 何时使用 BRG？

**适合**：
- 渲染大量相同网格（草地、树木、岩石、粒子）
- 使用 DOTS/ECS 架构的项目
- 需要精确控制渲染流程的高级用户
- 移动平台上的性能敏感场景

**不适合**：
- 少量物体的场景
- 网格种类繁多且各不相同
- 快速原型开发（BRG 的学习曲线较陡）
- 不需要自定义剔除的场景

### 5.2 选择决策树

```
需要渲染大量相同物体？
├── 是
│   ├── 使用 DOTS？
│   │   └── 是 → BatchRendererGroup
│   │   └── 否 → 
│   │       ├── 物体移动？
│   │       │   └── 是 → GPU Instancing / SRP Batcher
│   │       │   └── 否 → Static Batching
│   │       └── 自定义剔除需求？
│   │           └── 是 → BatchRendererGroup
│   └── 否 → SRP Batcher（如果使用 URP/HDRP）
└── 少量物体
    └── GameObjects + SRP Batcher
```

---

## 总结

BatchRendererGroup 代表了 Unity 渲染优化技术的最新阶段，它：

1. **解决了 Draw Call 瓶颈**：通过 GPU Instancing 技术
2. **解决了 CPU 瓶颈**：通过数据导向设计和 Job System 支持
3. **解决了 GameObject 开销**：直接操作 GPU 数据
4. **提供了精确控制**：自定义剔除、LOD、材质变化

从 Static Batching 到 Dynamic Batching，从 GPU Instancing 到 SRP Batcher，再到 BatchRendererGroup，这是一条不断追求更高性能的技术演进之路。理解这个演进过程，才能更好地选择和使用适合自己项目的技术。

---

## 参考资料

1. [Unity Documentation: BatchRendererGroup API](https://docs.unity3d.com/Manual/batch-renderer-group.html)
2. [LearnOpenGL: Instancing](https://learnopengl.com/Advanced-OpenGL/Instancing)
3. [Unity Blog: Achieve high frame rate on budget devices with BRG](https://unity.com/blog/engine-platform/batchrenderergroup-sample-high-frame-rate-on-budget-devices)
4. [GameDev.Center: Trying out BatchRendererGroup](https://gamedev.center/trying-out-new-unity-api-batchrenderergroup/)
5. [Unity Manual: SRP Batcher](https://docs.unity3d.com/Manual/SRPBatcher.html)
6. [Unity Manual: Optimizing draw calls](https://docs.unity.cn/Manual/optimizing-draw-calls.html)