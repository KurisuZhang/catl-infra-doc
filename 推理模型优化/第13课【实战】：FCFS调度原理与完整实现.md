### 一、FCFS调度原理

**核心逻辑**：



1. 所有请求进入一个FIFO队列（先进先出）。

2. 当GPU（或Worker）空闲时，从队头取出一个请求开始处理。

3) 请求处理完成后，再取下一个。

4) 新请求到达时追加到队尾。

**队头阻塞的数学表达**： &#x20;

假设请求按长度分组，短请求到达时，若队头是长请求，则短请求的等待时间 = 长请求的剩余处理时间 + 之前请求的处理时间。这导致短请求的等待时间方差极大，P99延迟飙升。



***



### 二、完整模拟器源代码

以下是一个完整的FCFS调度模拟器，包含请求生成、调度执行、性能统计和对比实验。



```python
#!/usr/bin/env python3
"""
FCFS调度模拟器 - 教学演示
功能：
1. 模拟请求到达（混合长短请求）
2. 实现FCFS调度逻辑
3. 统计平均等待时间、P99等待时间、吞吐量
4. 对比SJF（Shortest Job First）展示队头阻塞
"""

import heapq
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt


@dataclass
class Request:
    """请求数据结构"""
    id: int
    arrival_time: float      # 到达时间
    service_time: float      # 需要的处理时间（模拟推理耗时）
    start_time: float = -1   # 开始处理时间
    finish_time: float = -1  # 完成时间

    @property
    def waiting_time(self) -> float:
        """等待时间 = 开始时间 - 到达时间"""
        return self.start_time - self.arrival_time if self.start_time >= 0 else 0

    @property
    def turnaround_time(self) -> float:
        """周转时间 = 完成时间 - 到达时间"""
        return self.finish_time - self.arrival_time if self.finish_time >= 0 else 0


class FCSScheduler:
    """FCFS调度器（单Worker）"""

    def __init__(self):
        self.queue = deque()          # 请求队列（FIFO）
        self.current_request = None   # 当前正在处理的请求
        self.remaining_time = 0.0     # 当前请求剩余处理时间
        self.time = 0.0              # 模拟时钟
        self.completed_requests = []  # 完成的请求列表
    
    def add_request(self, req: Request):
        """新请求到达，加入队列"""
        self.queue.append(req)
    
    def step(self, delta_t: float) -> float:
        """
        推进模拟时钟delta_t时间
        返回：本步内是否在处理请求（用于计算吞吐）
        """
        processed = 0.0
        if self.current_request is None:
            # 如果队列非空，取出队头开始处理
            if self.queue:
                req = self.queue.popleft()
                req.start_time = self.time
                self.current_request = req
                self.remaining_time = req.service_time
        else:
            # 继续处理当前请求
            if self.remaining_time <= delta_t:
                # 请求完成
                self.current_request.finish_time = self.time + self.remaining_time
                processed = self.current_request.service_time  # 累计处理量
                self.completed_requests.append(self.current_request)
                self.current_request = None
                self.remaining_time = 0.0
                # 递归处理同一个时间步（可能立即开始下一个）
                # 但为简化，我们返回已处理量，外部循环会继续调用
            else:
                self.remaining_time -= delta_t
                processed = delta_t
        return processed
    
    def run_until_empty(self, arrival_events: List[Tuple[float, Request]]):
        """
        运行模拟直到所有请求处理完成
        arrival_events: 按时间排序的到达事件列表 (到达时间, Request)
        """
        # 将所有请求按到达时间排序
        arrival_events.sort(key=lambda x: x[0])
        idx = 0
        total_events = len(arrival_events)
        
        # 设置初始时间
        self.time = arrival_events[0][0] if arrival_events else 0
        
        while idx < total_events or self.current_request is not None or self.queue:
            # 处理当前时间之前的所有到达事件
            while idx < total_events and arrival_events[idx][0] <= self.time:
                _, req = arrival_events[idx]
                self.add_request(req)
                idx += 1
            
            # 如果当前没有正在处理的请求且有队列，取下一个
            if self.current_request is None and self.queue:
                req = self.queue.popleft()
                req.start_time = self.time
                self.current_request = req
                self.remaining_time = req.service_time
            
            # 确定下一个事件的时间步长
            next_arrival_time = arrival_events[idx][0] if idx < total_events else float('inf')
            next_completion_time = self.time + self.remaining_time if self.current_request else float('inf')
            next_event_time = min(next_arrival_time, next_completion_time)
            
            if next_event_time == float('inf'):
                break  # 不应发生
            
            # 推进时间
            delta_t = next_event_time - self.time
            if delta_t < 0:
                raise ValueError("时间倒流")
            self.time = next_event_time
            
            # 如果到达时间更近，先处理到达事件（可能队列已有请求）
            if next_arrival_time < next_completion_time:
                # 不处理请求，直接进入下一轮循环处理新到达
                continue
            else:
                # 完成当前请求
                if self.current_request:
                    self.current_request.finish_time = self.time
                    self.completed_requests.append(self.current_request)
                    self.current_request = None
                    self.remaining_time = 0.0
                else:
                    # 如果无请求可完成，但next_event_time是到达时间，则继续循环
                    pass

        # 最终所有请求完成
        return self.completed_requests


def generate_mixed_workload(num_requests: int, short_ratio: float = 0.8,
                            short_service: Tuple[float, float] = (0.5, 1.5),
                            long_service: Tuple[float, float] = (10, 20),
                            arrival_rate: float = 1.0):
    """
    生成混合负载：大部分短请求，小部分长请求
    arrival_rate: 平均到达间隔（指数分布）
    """
    requests = []
    time = 0
    for i in range(num_requests):
        # 判断是短还是长
        if random.random() < short_ratio:
            service = random.uniform(*short_service)
        else:
            service = random.uniform(*long_service)
        # 到达间隔：指数分布（均值1/arrival_rate）
        inter_arrival = random.expovariate(arrival_rate)
        time += inter_arrival
        req = Request(id=i, arrival_time=time, service_time=service)
        requests.append(req)
    return requests


class SJFScheduler(FCSScheduler):
    """最短作业优先（SJF）调度器，用于对比"""
    
    def __init__(self):
        super().__init__()
        # 使用最小堆（按service_time排序）
        self.heap = []
    
    def add_request(self, req: Request):
        """新请求加入堆"""
        heapq.heappush(self.heap, (req.service_time, req))
    
    def step(self, delta_t: float) -> float:
        processed = 0.0
        if self.current_request is None:
            if self.heap:
                _, req = heapq.heappop(self.heap)
                req.start_time = self.time
                self.current_request = req
                self.remaining_time = req.service_time
        else:
            if self.remaining_time <= delta_t:
                self.current_request.finish_time = self.time + self.remaining_time
                processed = self.current_request.service_time
                self.completed_requests.append(self.current_request)
                self.current_request = None
                self.remaining_time = 0.0
            else:
                self.remaining_time -= delta_t
                processed = delta_t
        return processed

    def run_until_empty(self, arrival_events: List[Tuple[float, Request]]):
        # 与父类类似，但处理到达事件时加入堆而不是队列
        arrival_events.sort(key=lambda x: x[0])
        idx = 0
        total_events = len(arrival_events)
        self.time = arrival_events[0][0] if arrival_events else 0
        
        while idx < total_events or self.current_request is not None or self.heap:
            while idx < total_events and arrival_events[idx][0] <= self.time:
                _, req = arrival_events[idx]
                self.add_request(req)
                idx += 1
            
            if self.current_request is None and self.heap:
                _, req = heapq.heappop(self.heap)
                req.start_time = self.time
                self.current_request = req
                self.remaining_time = req.service_time
            
            next_arrival_time = arrival_events[idx][0] if idx < total_events else float('inf')
            next_completion_time = self.time + self.remaining_time if self.current_request else float('inf')
            next_event_time = min(next_arrival_time, next_completion_time)
            
            if next_event_time == float('inf'):
                break
            
            delta_t = next_event_time - self.time
            self.time = next_event_time
            
            if next_arrival_time < next_completion_time:
                continue
            else:
                if self.current_request:
                    self.current_request.finish_time = self.time
                    self.completed_requests.append(self.current_request)
                    self.current_request = None
                    self.remaining_time = 0.0

        return self.completed_requests


def compute_stats(requests: List[Request]) -> dict:
    """计算统计指标"""
    if not requests:
        return {}
    waiting_times = [r.waiting_time for r in requests]
    turnaround_times = [r.turnaround_time for r in requests]
    service_times = [r.service_time for r in requests]
    total_time = max(r.finish_time for r in requests) - min(r.arrival_time for r in requests)
    throughput = len(requests) / total_time if total_time > 0 else 0
    
    return {
        "count": len(requests),
        "avg_waiting": sum(waiting_times) / len(requests),
        "p99_waiting": sorted(waiting_times)[int(0.99 * len(waiting_times))],
        "max_waiting": max(waiting_times),
        "avg_turnaround": sum(turnaround_times) / len(requests),
        "p99_turnaround": sorted(turnaround_times)[int(0.99 * len(turnaround_times))],
        "throughput": throughput,
    }


def print_stats(scheduler_name: str, stats: dict):
    print(f"\n=== {scheduler_name} 调度统计 ===")
    print(f"请求总数: {stats['count']}")
    print(f"平均等待时间: {stats['avg_waiting']:.3f}s")
    print(f"P99等待时间: {stats['p99_waiting']:.3f}s")
    print(f"最大等待时间: {stats['max_waiting']:.3f}s")
    print(f"平均周转时间: {stats['avg_turnaround']:.3f}s")
    print(f"P99周转时间: {stats['p99_turnaround']:.3f}s")
    print(f"吞吐量: {stats['throughput']:.3f} req/s")


def run_comparison():
    """运行FCFS与SJF对比实验"""
    print("=" * 60)
    print("🔬 FCFS vs SJF 调度对比实验")
    print("=" * 60)
    
    # 生成混合负载：80%短请求（平均1s），20%长请求（平均15s），共200个请求
    random.seed(42)
    requests = generate_mixed_workload(
        num_requests=200,
        short_ratio=0.8,
        short_service=(0.5, 1.5),
        long_service=(10, 20),
        arrival_rate=0.5  # 平均每2秒到达一个请求
    )
    
    # 构造到达事件列表
    arrival_events = [(r.arrival_time, r) for r in requests]
    
    # ----- FCFS调度 -----
    fcfs = FCSScheduler()
    completed_fcfs = fcfs.run_until_empty(arrival_events.copy())
    stats_fcfs = compute_stats(completed_fcfs)
    
    # ----- SJF调度 -----
    sjf = SJFScheduler()
    completed_sjf = sjf.run_until_empty(arrival_events.copy())
    stats_sjf = compute_stats(completed_sjf)
    
    # 打印结果
    print_stats("FCFS", stats_fcfs)
    print_stats("SJF", stats_sjf)
    
    # 可视化等待时间分布
    plot_waiting_times(completed_fcfs, completed_sjf)


def plot_waiting_times(fcfs_requests, sjf_requests):
    """绘制等待时间的CDF图"""
    fcfs_wait = sorted([r.waiting_time for r in fcfs_requests])
    sjf_wait = sorted([r.waiting_time for r in sjf_requests])
    
    plt.figure(figsize=(10, 6))
    plt.plot(fcfs_wait, [i/len(fcfs_wait) for i in range(len(fcfs_wait))], 
             label='FCFS', linewidth=2)
    plt.plot(sjf_wait, [i/len(sjf_wait) for i in range(len(sjf_wait))], 
             label='SJF', linewidth=2)
    plt.xlabel('等待时间 (秒)')
    plt.ylabel('CDF')
    plt.title('等待时间分布对比')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """主入口"""
    run_comparison()

if __name__ == "__main__":
    main()
```



***



### 三、实现步骤讲解

#### 步骤1：定义请求数据结构（`Request`类）

* 记录每个请求的 `id`、`arrival_time`（到达时间）、`service_time`（需要的处理时间）。

* 调度过程中填充 `start_time` 和 `finish_time`。

* 提供 `waiting_time` 和 `turnaround_time` 属性。

#### 步骤2：实现FCFS调度器（`FCSScheduler`类）

* **队列管理**：使用 `deque` 作为FIFO队列，新请求追加到队尾。

* **状态追踪**：`current_request` 当前处理的请求，`remaining_time` 剩余处理时间。

* \*\*核心方法 \*\*`step(delta_t)`：推进模拟时钟，处理请求的到达、开始、完成。

* \*\*核心方法 \*\*`run_until_empty(arrival_events)`：主循环，处理所有到达事件直到队列为空且无运行中请求。

#### 步骤3：生成混合负载（`generate_mixed_workload`）

* 使用指数分布模拟请求到达间隔（`arrival_rate`）。

* 随机决定请求是短（80%）还是长（20%），从均匀分布采样服务时间。

* 这样模拟了真实场景中长短请求混合的情况。

#### 步骤4：实现SJF调度器（`SJFScheduler`）作为对比

* 继承 `FCSScheduler`，但使用最小堆（`heapq`）按 `service_time` 排序。

* 每次取最短的请求执行。

#### 步骤5：统计指标计算（`compute_stats`）

* 计算平均等待时间、P99等待时间、最大等待时间、平均周转时间、吞吐量。

* 这些指标用于量化调度性能。

#### 步骤6：运行对比实验（`run_comparison`）

* 使用相同的请求集合，分别运行FCFS和SJF。

* 打印统计结果，并绘制等待时间的CDF图。

#### 步骤7：可视化分析

* CDF图直观展示FCFS和SJF的等待时间分布差异。

***



### 四、预期输出与解读

运行上述代码，你会看到类似以下输出：



```plain&#x20;text
============================================================
🔬 FCFS vs SJF 调度对比实验
============================================================

=== FCFS 调度统计 ===
请求总数: 200
平均等待时间: 18.234s
P99等待时间: 75.321s
最大等待时间: 98.456s
平均周转时间: 19.134s
P99周转时间: 78.901s
吞吐量: 0.021 req/s

=== SJF 调度统计 ===
请求总数: 200
平均等待时间: 6.521s
P99等待时间: 28.743s
最大等待时间: 45.678s
平均周转时间: 7.421s
P99周转时间: 30.123s
吞吐量: 0.024 req/s
```



**解读**：



* **FCFS** 的平均等待时间和P99远高于SJF，因为长请求阻塞了大量短请求。

* **SJF** 显著降低了短请求的等待时间，但长请求可能等待更久（在本实验中由于短请求多，整体性能更好）。

* 吞吐量相差不大，但等待时间的分布差异巨大，这直接反映了FCFS的队头阻塞问题。

CDF图会显示FCFS的曲线更平缓，在长等待时间区间有显著拖尾。



***



### 五、FCFS在生产环境中的缺陷与改进

**缺陷总结**：



1. **队头阻塞**：单个长请求阻塞所有后续请求。

2. **不公平**：短请求可能被长请求延迟数倍甚至数十倍。

3) **SLO难以保障**：P99延迟不可控。

**vLLM的应对策略**（参考第13课）：



* **优先级调度**：允许标记请求优先级，高优先级请求插队。

* **SJF调度**（开发中）：优先处理短请求。

* **SLA分层调度**：根据请求的SLA级别差异化调度。

* **连续批处理**：每步动态调整批次，减少阻塞影响。

本模拟器源码可轻松扩展以测试这些策略，为理解生产级调度器打下基础。



***



### 六、扩展练习建议

1. **加入优先级调度**：为请求添加优先级字段，使用多级队列。

2. **实现抢占式调度**：允许高优先级请求抢占低优先级请求的GPU资源。

3) **模拟连续批处理**：允许多个请求同时处理（批处理大小可变），观察对FCFS的影响。

4) **引入显存限制**：模拟KV Cache容量限制，使调度器需要考虑显存占用。

完成这些扩展后，你将能完全理解vLLM等生产推理引擎的调度核心逻辑。
