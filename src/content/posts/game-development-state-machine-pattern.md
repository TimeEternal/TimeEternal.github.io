---
title: 游戏开发中的状态机模式：从入门到精通的完整指南
published: 2026-03-13
description: "通过实际示例一步步详解状态机模式的设计与实现，包含多种实现方式和最佳实践。"
image: "/images/ai-industry-2025-header.png"
tags: ["游戏开发", "设计模式", "状态机", "Unity", "C#", "FSM", "编程架构"]
category: 游戏编程
draft: false
---

# 游戏开发中的状态机模式：从入门到精通的完整指南

## 引言

在游戏开发中，我们经常需要处理具有多种行为状态的对象：一个角色可以站立、行走、奔跑、跳跃、攻击；一个敌人可以巡逻、追击、攻击、逃跑；一个 UI 菜单可以处于打开、关闭、过渡动画等状态。

如何优雅地管理这些状态及其转换？**状态机模式（State Pattern）** 就是答案。

本文将从事无巨细的角度，通过实际示例一步步讲解状态机模式的设计与实现，从最简单的枚举实现到高级的面向对象设计，帮助你彻底掌握这一核心设计模式。

---

## 第一部分：什么是状态机？

### 1.1 生活中的状态机

理解状态机最好的方式是从生活中找例子。考虑一盏台灯：

```
台灯状态转换图：

[关闭] --按下开关--> [弱光] --按下开关--> [强光] --按下开关--> [关闭]
  ↑                                                              |
  └──────────────────────────────────────────────────────────────┘
```

这个台灯有以下特点：
- **有限的状态集合**：关闭、弱光、强光
- **同一时间只处于一个状态**
- **特定条件下状态会转换**
- **转换规则是固定的**

这就是一个典型的**有限状态机（Finite State Machine, FSM）**。

### 1.2 状态机的正式定义

根据 Wikipedia 的定义：

> "状态模式是一种行为设计模式，也称为'对象用于状态'模式。该模式用于封装同一对象基于其内部状态的不同行为。"

一个有限状态机由以下元素组成：

| 元素 | 描述 |
|------|------|
| **状态（State）** | 对象可能的状况 |
| **事件（Event）** | 触发状态改变的条件 |
| **转换（Transition）** | 从一个状态到另一个状态的变更 |
| **动作（Action）** | 状态转换时执行的操作 |
| **初始状态** | 对象创建时的默认状态 |

### 1.3 为什么游戏开发需要状态机？

**问题场景**：假设你要实现一个游戏角色，它有以下行为：

```
角色行为：
- 当玩家按下方向键时移动
- 当玩家按下跳跃键时跳跃
- 当玩家按下攻击键时攻击
- 当角色在空中时不能再次跳跃
- 当角色攻击时不能移动
- ...
```

**不用状态机的"面条代码"**：

```csharp
void Update()
{
    if (isGrounded)
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (!isAttacking)
            {
                Jump();
            }
        }
        if (Input.GetKeyDown(KeyCode.J))
        {
            isAttacking = true;
            Attack();
        }
        if (!isAttacking && !isJumping)
        {
            Move();
        }
    }
    else
    {
        // 在空中的逻辑...
        if (isJumping && !isFalling)
        {
            // 跳跃上升阶段...
        }
        else if (isFalling)
        {
            // 下落阶段...
        }
    }
    // 更多嵌套的 if-else...
}
```

这种代码的问题：
- **难以维护**：逻辑散落在各处
- **容易出 bug**：状态之间的约束关系不清晰
- **难以扩展**：添加新状态需要修改大量代码

**使用状态机后**：

```csharp
void Update()
{
    currentState.Update();  // 当前状态自己处理逻辑
}

// 状态转换清晰明确
void OnJumpPressed()
{
    if (currentState is GroundedState)
    {
        TransitionTo(new JumpingState());
    }
}
```

---

## 第二部分：状态机的实现方式

状态机有多种实现方式，从简单到复杂各有优劣。我们将通过一个**角色控制器**的例子，一步步展示每种实现。

### 2.1 方式一：枚举 + Switch-Case（最简单）

这是最直观的实现方式，适合状态较少且逻辑简单的场景。

#### 2.1.1 定义状态枚举

```csharp
public enum PlayerState
{
    Idle,       // 站立
    Walking,    // 行走
    Running,    // 奔跑
    Jumping,    // 跳跃
    Attacking   // 攻击
}

public class Player : MonoBehaviour
{
    private PlayerState currentState = PlayerState.Idle;
    
    void Update()
    {
        HandleInput();
        UpdateState();
    }
    
    void HandleInput()
    {
        switch (currentState)
        {
            case PlayerState.Idle:
                if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.A) || 
                    Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.D))
                {
                    currentState = Input.GetKey(KeyCode.LeftShift) 
                        ? PlayerState.Running 
                        : PlayerState.Walking;
                }
                else if (Input.GetKeyDown(KeyCode.Space))
                {
                    currentState = PlayerState.Jumping;
                }
                else if (Input.GetKeyDown(KeyCode.J))
                {
                    currentState = PlayerState.Attacking;
                }
                break;
                
            case PlayerState.Walking:
            case PlayerState.Running:
                if (!IsMoving())
                {
                    currentState = PlayerState.Idle;
                }
                else if (Input.GetKeyDown(KeyCode.Space))
                {
                    currentState = PlayerState.Jumping;
                }
                break;
                
            case PlayerState.Jumping:
                if (IsGrounded())
                {
                    currentState = PlayerState.Idle;
                }
                break;
                
            case PlayerState.Attacking:
                if (AttackFinished())
                {
                    currentState = PlayerState.Idle;
                }
                break;
        }
    }
    
    void UpdateState()
    {
        switch (currentState)
        {
            case PlayerState.Idle:
                PlayIdleAnimation();
                break;
            case PlayerState.Walking:
                Move(walkSpeed);
                PlayWalkAnimation();
                break;
            case PlayerState.Running:
                Move(runSpeed);
                PlayRunAnimation();
                break;
            case PlayerState.Jumping:
                ApplyJumpPhysics();
                PlayJumpAnimation();
                break;
            case PlayerState.Attacking:
                PerformAttack();
                PlayAttackAnimation();
                break;
        }
    }
}
```

#### 2.1.2 分析

**优点**：
- 实现简单，易于理解
- 不需要额外的类结构
- 适合快速原型开发

**缺点**：
- 所有逻辑集中在一个方法中，代码量大
- 添加新状态需要修改多处代码
- 状态之间的转换规则不清晰
- 难以复用单个状态的逻辑

### 2.2 方式二：面向对象的状态模式（推荐）

这是标准的 GoF 状态模式实现，通过将每个状态封装成独立的类来解决上述问题。

#### 2.2.1 状态模式的核心结构

```
┌─────────────────────────────────────────────────────────────┐
│                      State Pattern 结构                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐       ┌─────────────────────────────┐     │
│  │   Context   │──────>│        State (抽象)          │     │
│  │  (持有状态)  │       │  + Enter()                   │     │
│  └─────────────┘       │  + Exit()                    │     │
│         │              │  + Update()                  │     │
│         │              │  + HandleInput()             │     │
│         ↓              └─────────────────────────────┘     │
│  ┌─────────────┘                    △                       │
│  │ currentState│          ┌─────────┴─────────┐            │
│  └─────────────┘          │                   │            │
│                    ┌──────┴──────┐     ┌──────┴──────┐     │
│                    │ ConcreteState│     │ ConcreteState│   │
│                    │   (Idle)     │     │  (Walking)   │   │
│                    └─────────────┘     └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2.2 基础类结构

首先，我们创建一个通用的状态基类：

```csharp
// State.cs - 状态基类
public abstract class State
{
    protected FSM fsm;  // 所属的状态机
    
    public State(FSM fsm)
    {
        this.fsm = fsm;
    }
    
    // 进入状态时调用（状态开始）
    public virtual void Enter() { }
    
    // 离开状态时调用（状态结束）
    public virtual void Exit() { }
    
    // 每帧更新
    public virtual void Update() { }
    
    // 固定时间步更新（物理相关）
    public virtual void FixedUpdate() { }
    
    // 处理输入
    public virtual void HandleInput() { }
}
```

然后，创建状态机类：

```csharp
// FSM.cs - 有限状态机
using System.Collections.Generic;

public class FSM
{
    protected Dictionary<int, State> states = new Dictionary<int, State>();
    protected State currentState;
    
    // 添加状态
    public void AddState(int key, State state)
    {
        states.Add(key, state);
    }
    
    // 获取状态
    public State GetState(int key)
    {
        return states[key];
    }
    
    // 设置当前状态
    public void SetCurrentState(State state)
    {
        // 先调用上一个状态的 Exit
        if (currentState != null)
        {
            currentState.Exit();
        }
        
        currentState = state;
        
        // 再调用新状态的 Enter
        if (currentState != null)
        {
            currentState.Enter();
        }
    }
    
    // 每帧更新
    public void Update()
    {
        if (currentState != null)
        {
            currentState.Update();
        }
    }
    
    // 物理更新
    public void FixedUpdate()
    {
        if (currentState != null)
        {
            currentState.FixedUpdate();
        }
    }
    
    // 处理输入
    public void HandleInput()
    {
        if (currentState != null)
        {
            currentState.HandleInput();
        }
    }
}
```

#### 2.2.3 实际应用：角色控制器

现在让我们用状态模式实现一个完整的角色控制器。

**步骤 1：定义状态键**

```csharp
// PlayerStateID.cs
public static class PlayerStateID
{
    public const int Idle = 0;
    public const int Walking = 1;
    public const int Running = 2;
    public const int Jumping = 3;
    public const int Attacking = 4;
}
```

**步骤 2：创建角色上下文**

```csharp
// PlayerController.cs
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [Header("移动参数")]
    public float walkSpeed = 3f;
    public float runSpeed = 6f;
    public float jumpForce = 8f;
    
    [Header("组件引用")]
    public Animator animator;
    public Rigidbody rb;
    public GroundChecker groundChecker;
    
    // 状态机
    private FSM fsm;
    
    // 公开属性供状态访问
    public bool IsGrounded => groundChecker.IsGrounded;
    public Vector3 MoveInput { get; private set; }
    public bool IsRunPressed { get; private set; }
    public bool IsJumpPressed { get; private set; }
    public bool IsAttackPressed { get; private set; }
    
    void Awake()
    {
        // 初始化状态机
        fsm = new FSM();
        
        // 注册所有状态
        fsm.AddState(PlayerStateID.Idle, new PlayerIdleState(fsm, this));
        fsm.AddState(PlayerStateID.Walking, new PlayerWalkingState(fsm, this));
        fsm.AddState(PlayerStateID.Running, new PlayerRunningState(fsm, this));
        fsm.AddState(PlayerStateID.Jumping, new PlayerJumpingState(fsm, this));
        fsm.AddState(PlayerStateID.Attacking, new PlayerAttackingState(fsm, this));
        
        // 设置初始状态
        fsm.SetCurrentState(fsm.GetState(PlayerStateID.Idle));
    }
    
    void Update()
    {
        // 读取输入
        ReadInput();
        
        // 让状态机处理更新
        fsm.HandleInput();
        fsm.Update();
    }
    
    void FixedUpdate()
    {
        fsm.FixedUpdate();
    }
    
    void ReadInput()
    {
        float h = Input.GetAxis("Horizontal");
        float v = Input.GetAxis("Vertical");
        MoveInput = new Vector3(h, 0f, v).normalized;
        
        IsRunPressed = Input.GetKey(KeyCode.LeftShift);
        IsJumpPressed = Input.GetKeyDown(KeyCode.Space);
        IsAttackPressed = Input.GetKeyDown(KeyCode.J);
    }
    
    // 公开方法供状态调用
    public void Move(float speed)
    {
        Vector3 movement = MoveInput * speed * Time.deltaTime;
        transform.Translate(movement, Space.World);
        
        // 朝向移动方向
        if (MoveInput != Vector3.zero)
        {
            transform.forward = MoveInput;
        }
    }
    
    public void Jump()
    {
        rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
    }
    
    public void PlayAnimation(string animationName)
    {
        animator.CrossFade(animationName, 0.1f);
    }
    
    public void TransitionToState(int stateID)
    {
        fsm.SetCurrentState(fsm.GetState(stateID));
    }
}
```

**步骤 3：实现具体状态类**

```csharp
// PlayerIdleState.cs - 站立状态
public class PlayerIdleState : State
{
    private PlayerController player;
    
    public PlayerIdleState(FSM fsm, PlayerController player) : base(fsm)
    {
        this.player = player;
    }
    
    public override void Enter()
    {
        // 进入站立状态：播放站立动画
        player.PlayAnimation("Idle");
    }
    
    public override void Exit()
    {
        // 离开站立状态：可以做清理工作
    }
    
    public override void HandleInput()
    {
        // 检测是否开始移动
        if (player.MoveInput != Vector3.zero)
        {
            if (player.IsRunPressed)
            {
                fsm.SetCurrentState(fsm.GetState(PlayerStateID.Running));
            }
            else
            {
                fsm.SetCurrentState(fsm.GetState(PlayerStateID.Walking));
            }
            return;
        }
        
        // 检测跳跃
        if (player.IsJumpPressed && player.IsGrounded)
        {
            fsm.SetCurrentState(fsm.GetState(PlayerStateID.Jumping));
            return;
        }
        
        // 检测攻击
        if (player.IsAttackPressed)
        {
            fsm.SetCurrentState(fsm.GetState(PlayerStateID.Attacking));
            return;
        }
    }
    
    public override void Update()
    {
        // 站立状态的持续逻辑
        // 比如：站立时恢复体力
    }
}
```

```csharp
// PlayerWalkingState.cs - 行走状态
public class PlayerWalkingState : State
{
    private PlayerController player;
    
    public PlayerWalkingState(FSM fsm, PlayerController player) : base(fsm)
    {
        this.player = player;
    }
    
    public override void Enter()
    {
        player.PlayAnimation("Walk");
    }
    
    public override void HandleInput()
    {
        // 停止移动 -> 切换到站立
        if (player.MoveInput == Vector3.zero)
        {
            fsm.SetCurrentState(fsm.GetState(PlayerStateID.Idle));
            return;
        }
        
        // 按住 Shift -> 切换到奔跑
        if (player.IsRunPressed)
        {
            fsm.SetCurrentState(fsm.GetState(PlayerStateID.Running));
            return;
        }
        
        // 跳跃
        if (player.IsJumpPressed && player.IsGrounded)
        {
            fsm.SetCurrentState(fsm.GetState(PlayerStateID.Jumping));
            return;
        }
    }
    
    public override void FixedUpdate()
    {
        player.Move(player.walkSpeed);
    }
}
```

```csharp
// PlayerJumpingState.cs - 跳跃状态
public class PlayerJumpingState : State
{
    private PlayerController player;
    private bool hasJumped;
    
    public PlayerJumpingState(FSM fsm, PlayerController player) : base(fsm)
    {
        this.player = player;
    }
    
    public override void Enter()
    {
        player.PlayAnimation("Jump");
        player.Jump();
        hasJumped = true;
    }
    
    public override void Update()
    {
        // 等待落地
        if (hasJumped && player.IsGrounded)
        {
            fsm.SetCurrentState(fsm.GetState(PlayerStateID.Idle));
        }
    }
    
    public override void FixedUpdate()
    {
        // 跳跃时也可以空中移动（可选）
        player.Move(player.walkSpeed * 0.5f);
    }
}
```

```csharp
// PlayerAttackingState.cs - 攻击状态
public class PlayerAttackingState : State
{
    private PlayerController player;
    private float attackDuration = 0.5f;
    private float attackTimer;
    
    public PlayerAttackingState(FSM fsm, PlayerController player) : base(fsm)
    {
        this.player = player;
    }
    
    public override void Enter()
    {
        player.PlayAnimation("Attack");
        attackTimer = 0f;
        
        // 触发攻击伤害检测等
        // player.PerformAttack();
    }
    
    public override void Update()
    {
        attackTimer += Time.deltaTime;
        
        // 攻击动画播放完毕
        if (attackTimer >= attackDuration)
        {
            fsm.SetCurrentState(fsm.GetState(PlayerStateID.Idle));
        }
    }
}
```

#### 2.2.4 状态转换图

```
玩家角色状态转换图：

                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
              ┌─────────┐                                      │
              │  Idle   │◄─────────────────────────────┐       │
              └────┬────┘                              │       │
                   │                                   │       │
        移动键按下  │                                   │       │
                   │    Shift未按住                    │       │
                   ▼                                   │       │
              ┌─────────┐  移动键松开  ┌─────────┐     │       │
         ┌───>│ Walking │────────────►│  Idle   │─────┘       │
         │    └────┬────┘             └─────────┘             │
         │         │                            ▲              │
         │         │ Shift按住                  │              │
         │         ▼                            │              │
         │    ┌─────────┐  移动键松开           │              │
         │    │ Running │──────────────────────┤              │
         │    └────┬────┘                       │              │
         │         │                            │              │
         │         │ Space按下 (在地面)         │              │
         │         ▼                            │              │
         │    ┌─────────┐  落地                 │              │
         │    │ Jumping │──────────────────────┤              │
         │    └─────────┘                       │              │
         │                                      │              │
         │         J按下                        │              │
         │         │                            │              │
         └─────────┴──────────►┌─────────┐     │              │
                               │Attacking│─────┘              │
                               └─────────┘  攻击结束           │
                                                              │
                                                              │
```

### 2.3 方式三：使用委托的状态机（更灵活）

委托方式提供了一种更灵活、更轻量的实现，特别适合需要频繁添加/移除状态逻辑的场景。

```csharp
// DelegateFSM.cs - 基于委托的状态机
using System;
using System.Collections.Generic;

public class DelegateState
{
    public Action OnEnter;      // 进入状态
    public Action OnExit;       // 离开状态
    public Action OnUpdate;     // 每帧更新
    public Action OnFixedUpdate; // 物理更新
    
    public DelegateState(Action onEnter = null, Action onExit = null, 
                         Action onUpdate = null, Action onFixedUpdate = null)
    {
        OnEnter = onEnter;
        OnExit = onExit;
        OnUpdate = onUpdate;
        OnFixedUpdate = onFixedUpdate;
    }
}

public class DelegateFSM
{
    private Dictionary<string, DelegateState> states = new Dictionary<string, DelegateState>();
    private DelegateState currentState;
    private string currentStateName;
    
    public void AddState(string name, DelegateState state)
    {
        states[name] = state;
    }
    
    public void ChangeState(string name)
    {
        if (currentState != null && currentState.OnExit != null)
        {
            currentState.OnExit();
        }
        
        currentStateName = name;
        currentState = states[name];
        
        if (currentState != null && currentState.OnEnter != null)
        {
            currentState.OnEnter();
        }
    }
    
    public void Update()
    {
        if (currentState != null && currentState.OnUpdate != null)
        {
            currentState.OnUpdate();
        }
    }
    
    public void FixedUpdate()
    {
        if (currentState != null && currentState.OnFixedUpdate != null)
        {
            currentState.OnFixedUpdate();
        }
    }
    
    public string GetCurrentStateName()
    {
        return currentStateName;
    }
}
```

**使用示例**：

```csharp
public class SimpleEnemy : MonoBehaviour
{
    private DelegateFSM fsm;
    
    void Start()
    {
        fsm = new DelegateFSM();
        
        // 定义巡逻状态
        fsm.AddState("Patrol", new DelegateState(
            onEnter: () => {
                Debug.Log("开始巡逻");
                // 选择一个巡逻点
            },
            onUpdate: () => {
                // 移动到巡逻点
                Patrol();
                
                // 检测玩家
                if (CanSeePlayer())
                {
                    fsm.ChangeState("Chase");
                }
            },
            onExit: () => {
                Debug.Log("停止巡逻");
            }
        ));
        
        // 定义追击状态
        fsm.AddState("Chase", new DelegateState(
            onEnter: () => {
                Debug.Log("发现敌人，开始追击！");
            },
            onUpdate: () => {
                ChasePlayer();
                
                // 距离太远，丢失目标
                if (DistanceToPlayer() > 20f)
                {
                    fsm.ChangeState("Patrol");
                }
                // 距离足够近，攻击
                else if (DistanceToPlayer() < 2f)
                {
                    fsm.ChangeState("Attack");
                }
            }
        ));
        
        // 定义攻击状态
        fsm.AddState("Attack", new DelegateState(
            onEnter: () => {
                Debug.Log("攻击！");
                PerformAttack();
            },
            onExit: () => {
                // 攻击结束
            }
        ));
        
        // 设置初始状态
        fsm.ChangeState("Patrol");
    }
    
    void Update()
    {
        fsm.Update();
    }
}
```

---

## 第三部分：进阶技巧与最佳实践

### 3.1 状态机与其他模式的结合

#### 3.1.1 状态机 + 观察者模式

当状态变化需要通知其他系统时（如 UI 更新、音效播放），可以使用观察者模式：

```csharp
public class ObservableFSM : FSM
{
    // 状态变化事件
    public event Action<State, State> OnStateChanged; // 前一个状态，新状态
    
    public override void SetCurrentState(State state)
    {
        State previousState = currentState;
        
        base.SetCurrentState(state);
        
        // 触发事件
        OnStateChanged?.Invoke(previousState, currentState);
    }
}

// 使用示例
fsm.OnStateChanged += (from, to) => {
    Debug.Log($"状态变化: {from?.GetType().Name} -> {to?.GetType().Name}");
    AudioManager.Instance.PlayStateChangeSound(to);
};
```

#### 3.1.2 状态机 + 对象池

对于频繁创建销毁的状态对象，可以使用对象池：

```csharp
public class StatePool<T> where T : State, new()
{
    private Stack<T> pool = new Stack<T>();
    
    public T Get(FSM fsm)
    {
        T state = pool.Count > 0 ? pool.Pop() : new T();
        // 初始化状态...
        return state;
    }
    
    public void Return(T state)
    {
        // 重置状态...
        pool.Push(state);
    }
}
```

### 3.2 层次状态机（Hierarchical State Machine）

当状态之间存在层次关系时，可以使用层次状态机：

```
角色状态层次结构：

PlayerState
├── Grounded (地面状态)
│   ├── Idle
│   ├── Walking
│   └── Running
├── Airborne (空中状态)
│   ├── Jumping
│   └── Falling
└── Action (动作状态)
    ├── Attacking
    └── Dodging
```

**实现方式**：

```csharp
public abstract class HierarchicalState : State
{
    protected HierarchicalState parentState;
    protected State currentSubState;
    
    public HierarchicalState(FSM fsm, HierarchicalState parent = null) : base(fsm)
    {
        parentState = parent;
    }
    
    public override void Update()
    {
        // 先更新子状态
        currentSubState?.Update();
        
        // 再更新自己
        OnUpdate();
    }
    
    protected virtual void OnUpdate() { }
}

// 地面状态（父状态）
public class GroundedState : HierarchicalState
{
    public GroundedState(FSM fsm) : base(fsm, null)
    {
        // 默认子状态
        currentSubState = new IdleState(fsm, this);
    }
    
    public override void Enter()
    {
        currentSubState?.Enter();
    }
    
    // 地面状态的公共逻辑
    public bool CheckJumpInput()
    {
        return Input.GetKeyDown(KeyCode.Space);
    }
}

// 站立状态（子状态）
public class IdleState : HierarchicalState
{
    public IdleState(FSM fsm, HierarchicalState parent) : base(fsm, parent) { }
    
    public override void HandleInput()
    {
        // 可以访问父状态的公共方法
        if ((parentState as GroundedState).CheckJumpInput())
        {
            // 转换到跳跃状态...
        }
    }
}
```

### 3.3 并行状态机

有时一个对象需要同时处于多个"独立"的状态。例如，一个角色可能同时在"移动"和"攻击"：

```csharp
public class ParallelFSM
{
    private List<FSM> stateMachines = new List<FSM>();
    
    public void AddFSM(FSM fsm)
    {
        stateMachines.Add(fsm);
    }
    
    public void Update()
    {
        foreach (var fsm in stateMachines)
        {
            fsm.Update();
        }
    }
}

// 使用示例
public class Player : MonoBehaviour
{
    private ParallelFSM parallelFSM;
    
    void Start()
    {
        parallelFSM = new ParallelFSM();
        
        // 移动状态机
        FSM movementFSM = new FSM();
        movementFSM.AddState(0, new IdleState());
        movementFSM.AddState(1, new WalkState());
        parallelFSM.AddFSM(movementFSM);
        
        // 战斗状态机（独立运行）
        FSM combatFSM = new FSM();
        combatFSM.AddState(0, new IdleCombatState());
        combatFSM.AddState(1, new AttackingState());
        parallelFSM.AddFSM(combatFSM);
    }
}
```

---

## 第四部分：实战案例 - 敌人 AI

让我们实现一个完整的敌人 AI，展示状态机在游戏 AI 中的应用。

### 4.1 敌人状态设计

```
敌人 AI 状态图：

                    ┌─────────────────────────────────────┐
                    │                                     │
                    ▼                                     │
              ┌──────────┐                               │
              │  Patrol  │◄──────────────────────────────┤
              │  (巡逻)   │                               │
              └────┬─────┘                               │
                   │ 发现玩家 (距离 < 10)                  │
                   ▼                                     │
              ┌──────────┐ 玩家逃跑 (距离 > 20) ─────────┤
              │  Chase   │                               │
              │  (追击)   │                               │
              └────┬─────┘                               │
                   │ 接近玩家 (距离 < 2)                   │
                   ▼                                     │
              ┌──────────┐ 攻击完成                       │
              │  Attack  │───────────────────────────────┘
              │  (攻击)   │
              └────┬─────┘
                   │ 生命值 < 20%
                   ▼
              ┌──────────┐
              │   Flee   │
              │  (逃跑)   │
              └──────────┘
```

### 4.2 完整实现

```csharp
// Enemy.cs - 敌人主类
using UnityEngine;

public class Enemy : MonoBehaviour
{
    [Header("属性")]
    public float maxHealth = 100f;
    public float currentHealth;
    public float patrolSpeed = 2f;
    public float chaseSpeed = 4f;
    public float attackRange = 2f;
    public float detectionRange = 10f;
    public float losePlayerRange = 20f;
    
    [Header("组件")]
    public Animator animator;
    public NavMeshAgent agent;
    public Transform player;
    
    // 状态机
    private FSM fsm;
    
    // 状态键
    private const int STATE_PATROL = 0;
    private const int STATE_CHASE = 1;
    private const int STATE_ATTACK = 2;
    private const int STATE_FLEE = 3;
    
    public float DistanceToPlayer => Vector3.Distance(transform.position, player.position);
    public bool IsHealthLow => currentHealth < maxHealth * 0.2f;
    
    void Awake()
    {
        currentHealth = maxHealth;
        
        fsm = new FSM();
        fsm.AddState(STATE_PATROL, new EnemyPatrolState(fsm, this));
        fsm.AddState(STATE_CHASE, new EnemyChaseState(fsm, this));
        fsm.AddState(STATE_ATTACK, new EnemyAttackState(fsm, this));
        fsm.AddState(STATE_FLEE, new EnemyFleeState(fsm, this));
        
        fsm.SetCurrentState(fsm.GetState(STATE_PATROL));
    }
    
    void Update()
    {
        fsm.Update();
    }
    
    public void TakeDamage(float damage)
    {
        currentHealth -= damage;
        if (currentHealth <= 0)
        {
            Die();
        }
    }
    
    void Die()
    {
        Destroy(gameObject);
    }
}
```

```csharp
// EnemyPatrolState.cs - 巡逻状态
public class EnemyPatrolState : State
{
    private Enemy enemy;
    private Vector3 patrolTarget;
    private float waitTimer;
    private bool isWaiting;
    
    public EnemyPatrolState(FSM fsm, Enemy enemy) : base(fsm)
    {
        this.enemy = enemy;
    }
    
    public override void Enter()
    {
        enemy.animator.SetBool("IsWalking", true);
        SetNewPatrolTarget();
    }
    
    public override void Update()
    {
        // 检测玩家
        if (enemy.DistanceToPlayer < enemy.detectionRange)
        {
            fsm.SetCurrentState(fsm.GetState(STATE_CHASE));
            return;
        }
        
        // 生命值过低，逃跑
        if (enemy.IsHealthLow)
        {
            fsm.SetCurrentState(fsm.GetState(STATE_FLEE));
            return;
        }
        
        // 巡逻逻辑
        if (isWaiting)
        {
            waitTimer -= Time.deltaTime;
            if (waitTimer <= 0)
            {
                isWaiting = false;
                SetNewPatrolTarget();
            }
        }
        else
        {
            // 移动到巡逻点
            enemy.agent.SetDestination(patrolTarget);
            
            if (Vector3.Distance(enemy.transform.position, patrolTarget) < 1f)
            {
                // 到达巡逻点，等待
                isWaiting = true;
                waitTimer = Random.Range(2f, 5f);
                enemy.animator.SetBool("IsWalking", false);
            }
        }
    }
    
    void SetNewPatrolTarget()
    {
        // 随机选择一个巡逻点
        Vector3 randomDirection = Random.insideUnitSphere * 10f;
        randomDirection += enemy.transform.position;
        patrolTarget = randomDirection;
        enemy.animator.SetBool("IsWalking", true);
    }
    
    public override void Exit()
    {
        enemy.animator.SetBool("IsWalking", false);
    }
}
```

```csharp
// EnemyChaseState.cs - 追击状态
public class EnemyChaseState : State
{
    private Enemy enemy;
    
    public EnemyChaseState(FSM fsm, Enemy enemy) : base(fsm)
    {
        this.enemy = enemy;
    }
    
    public override void Enter()
    {
        enemy.agent.speed = enemy.chaseSpeed;
        enemy.animator.SetBool("IsRunning", true);
    }
    
    public override void Update()
    {
        // 玩家逃跑
        if (enemy.DistanceToPlayer > enemy.losePlayerRange)
        {
            fsm.SetCurrentState(fsm.GetState(STATE_PATROL));
            return;
        }
        
        // 接近玩家，攻击
        if (enemy.DistanceToPlayer < enemy.attackRange)
        {
            fsm.SetCurrentState(fsm.GetState(STATE_ATTACK));
            return;
        }
        
        // 生命值过低
        if (enemy.IsHealthLow)
        {
            fsm.SetCurrentState(fsm.GetState(STATE_FLEE));
            return;
        }
        
        // 追击玩家
        enemy.agent.SetDestination(enemy.player.position);
    }
    
    public override void Exit()
    {
        enemy.agent.speed = enemy.patrolSpeed;
        enemy.animator.SetBool("IsRunning", false);
    }
}
```

```csharp
// EnemyAttackState.cs - 攻击状态
public class EnemyAttackState : State
{
    private Enemy enemy;
    private float attackCooldown = 1f;
    private float attackTimer;
    private bool hasAttacked;
    
    public EnemyAttackState(FSM fsm, Enemy enemy) : base(fsm)
    {
        this.enemy = enemy;
    }
    
    public override void Enter()
    {
        attackTimer = 0f;
        hasAttacked = false;
        enemy.agent.isStopped = true;
        enemy.animator.SetTrigger("Attack");
    }
    
    public override void Update()
    {
        attackTimer += Time.deltaTime;
        
        // 攻击动画中点时造成伤害
        if (!hasAttacked && attackTimer >= 0.3f)
        {
            hasAttacked = true;
            DealDamage();
        }
        
        // 攻击完成
        if (attackTimer >= attackCooldown)
        {
            // 检查玩家是否还在攻击范围内
            if (enemy.DistanceToPlayer < enemy.attackRange)
            {
                // 继续攻击
                attackTimer = 0f;
                hasAttacked = false;
                enemy.animator.SetTrigger("Attack");
            }
            else
            {
                // 玩家逃跑，追击
                fsm.SetCurrentState(fsm.GetState(STATE_CHASE));
            }
        }
    }
    
    void DealDamage()
    {
        // 对玩家造成伤害
        // Player.Instance.TakeDamage(10f);
    }
    
    public override void Exit()
    {
        enemy.agent.isStopped = false;
    }
}
```

```csharp
// EnemyFleeState.cs - 逃跑状态
public class EnemyFleeState : State
{
    private Enemy enemy;
    private Vector3 fleeTarget;
    
    public EnemyFleeState(FSM fsm, Enemy enemy) : base(fsm)
    {
        this.enemy = enemy;
    }
    
    public override void Enter()
    {
        // 计算逃跑方向（远离玩家）
        Vector3 fleeDirection = (enemy.transform.position - enemy.player.position).normalized;
        fleeTarget = enemy.transform.position + fleeDirection * 20f;
        
        enemy.agent.speed = enemy.chaseSpeed * 1.5f;
        enemy.animator.SetBool("IsRunning", true);
    }
    
    public override void Update()
    {
        enemy.agent.SetDestination(fleeTarget);
        
        // 逃跑成功
        if (Vector3.Distance(enemy.transform.position, fleeTarget) < 2f)
        {
            fsm.SetCurrentState(fsm.GetState(STATE_PATROL));
        }
    }
    
    public override void Exit()
    {
        enemy.agent.speed = enemy.patrolSpeed;
        enemy.animator.SetBool("IsRunning", false);
    }
}
```

---

## 第五部分：常见问题与解决方案

### 5.1 状态转换时的竞态条件

**问题**：在同一帧内多次触发状态转换。

```csharp
// 问题代码
void Update()
{
    if (Input.GetKeyDown(KeyCode.Space))
    {
        fsm.ChangeState("Jump");  // 第一次转换
    }
    if (Input.GetKeyDown(KeyCode.J))
    {
        fsm.ChangeState("Attack"); // 第二次转换（不应该发生）
    }
}
```

**解决方案**：使用延迟状态变更或状态变更队列。

```csharp
public class SafeFSM : FSM
{
    private State pendingState;
    private bool isTransitioning = false;
    
    public override void SetCurrentState(State state)
    {
        if (isTransitioning)
        {
            pendingState = state;
            return;
        }
        
        isTransitioning = true;
        base.SetCurrentState(state);
        isTransitioning = false;
        
        if (pendingState != null)
        {
            var temp = pendingState;
            pendingState = null;
            SetCurrentState(temp);
        }
    }
}
```

### 5.2 状态数据的持久化

**问题**：状态之间需要共享数据。

**解决方案**：使用上下文对象或黑板（Blackboard）模式。

```csharp
public class Blackboard
{
    public float lastAttackTime;
    public Vector3 lastKnownPlayerPosition;
    public int attackCount;
}

public class EnemyState : State
{
    protected Enemy enemy;
    protected Blackboard blackboard;
    
    public EnemyState(FSM fsm, Enemy enemy, Blackboard blackboard) : base(fsm)
    {
        this.enemy = enemy;
        this.blackboard = blackboard;
    }
}
```

### 5.3 调试状态机

**解决方案**：实现可视化调试工具。

```csharp
public class FSMDebugger : MonoBehaviour
{
    public FSM targetFSM;
    public bool showInGame = true;
    
    void OnGUI()
    {
        if (!showInGame || targetFSM == null) return;
        
        GUILayout.BeginArea(new Rect(10, 10, 300, 200));
        GUILayout.Label($"当前状态: {targetFSM.GetCurrentStateName()}");
        GUILayout.Label($"状态历史: {GetStateHistory()}");
        GUILayout.EndArea();
    }
    
    string GetStateHistory()
    {
        // 返回最近的状态变化历史
        return string.Join(" -> ", stateHistory);
    }
}
```

---

## 总结

状态机模式是游戏开发中最重要、最常用的设计模式之一。本文从基础概念出发，详细讲解了：

1. **什么是状态机**：有限状态机的定义、元素和用途
2. **三种实现方式**：
   - 枚举 + Switch-Case：简单直接，适合小型项目
   - 面向对象状态模式：标准 GoF 实现，推荐使用
   - 委托方式：灵活轻量，适合特定场景
3. **进阶技巧**：观察者模式结合、层次状态机、并行状态机
4. **实战案例**：完整的敌人 AI 实现
5. **常见问题**：竞态条件、数据持久化、调试方法

掌握状态机模式，将使你的游戏代码更加清晰、可维护、可扩展。

---

## 参考资料

1. [Game Programming Patterns - State Pattern](https://gameprogrammingpatterns.com/state.html)
2. [Unity Learn - Finite State Machines](https://learn.unity.com/tutorial/finite-state-machines-1)
3. [Wikipedia - State Pattern](https://en.wikipedia.org/wiki/State_pattern)
4. [Faramira - Implementing a Finite State Machine Using C# in Unity](https://faramira.com/implementing-a-finite-state-machine-using-c-in-unity-part-1/)
5. [Habrador - Game programming patterns in Unity - State Pattern](https://www.habrador.com/tutorials/programming-patterns/6-state-pattern/)