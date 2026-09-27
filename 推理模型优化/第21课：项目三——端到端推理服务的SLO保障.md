##

本节将带领你完成一个**生产级推理服务SLO保障体系**的完整搭建，涵盖SLA/SLO设计、请求调度与动态批处理、监控告警体系构建、以及基于KEDA的弹性扩缩容。







### 一、项目概览

| 阶段      | 任务           | 技术栈                                 |
| ------- | ------------ | ----------------------------------- |
| **阶段一** | SLO指标体系设计    | vLLM Metrics、Prometheus Histogram   |
| **阶段二** | 监控与告警体系搭建    | Prometheus + Grafana + Alertmanager |
| **阶段三** | 请求调度与动态批处理   | vLLM Scheduler、Dynamic Batch        |
| **阶段四** | 基于KEDA的弹性扩缩容 | KEDA + Prometheus Scaler            |





### 二、阶段一：SLO指标体系设计

#### 2.1 vLLM核心SLO指标

vLLM通过 `/metrics` 端点暴露丰富的Prometheus指标。以下是SLO保障的核心指标：



| 指标名                                | 类型        | 含义                    | SLO阈值示例     |
| ---------------------------------- | --------- | --------------------- | ----------- |
| `vllm:time_to_first_token_seconds` | Histogram | TTFT首Token延迟          | P95 < 800ms |
| `vllm:inter_token_latency_seconds` | Histogram | Token间延迟(TPOT)        | P95 < 150ms |
| `vllm:e2e_request_latency_seconds` | Histogram | 端到端请求延迟               | P95 < 3s    |
| `vllm:num_requests_running`        | Gauge     | 当前运行请求数               | 监控用         |
| `vllm:kv_cache_usage_perc`         | Gauge     | KV缓存利用率(0-1)          | < 0.9       |
| `vllm:request_success_total`       | Counter   | 成功请求数(按finish reason) | 成功率 > 99.9% |





**请求级指标是SLO的核心**，它们以直方图形式暴露，正是SRE团队监控vLLM时追踪的SLO。



#### 2.2 Goodput：SLO感知的有效吞吐量

**Goodput（有效吞吐量）** 定义为在满足指定SLO约束的前提下，系统每秒能完成的请求数。



vLLM已支持 `--goodput` 参数，接受SLO键值对：



```bash
# 在基准测试中指定SLO约束
vllm bench serve \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --goodput ttft:500 tpot:100 e2el:3000
```



其中键为指标名称（`ttft`、`tpot`、`e2el`），值为毫秒数。







### 三、阶段二：监控与告警体系搭建

#### 3.1 部署监控栈（Prometheus + Grafana）

vLLM Production Stack Helm Chart（v0.1.11+）已将监控栈集成到Chart中：



```yaml
# values.yaml - 启用监控栈
servingEngineSpec:
  serviceMonitor:
    enabled: true  # 启用ServiceMonitor让Prometheus采集vLLM指标

routerSpec:
  serviceMonitor:
    enabled: true

# 部署完整的kube-prometheus-stack (Prometheus + Grafana)
kube-prometheus-stack:
  enabled: true
```



部署命令：



```bash
helm install vllm vllm/vllm-stack -f values.yaml
```



#### 3.2 自定义Prometheus告警规则

创建SLO告警规则文件 `slo-alerts.yaml`：



```yaml
# slo-alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: vllm-slo-alerts
  namespace: monitoring
  labels:
    prometheus: kube-prometheus
    role: alert-rules
spec:
  groups:
  - name: vllm-slo-alerts
    interval: 30s
    rules:
    # 1. TTFT SLO违反告警 (P95 > 800ms)
    - alert: VLLMTTFTHigh
      expr: |
        histogram_quantile(0.95, 
          sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le)
        ) > 0.8
      for: 5m
      labels:
        severity: critical
        component: vllm
      annotations:
        summary: "vLLM TTFT P95超过800ms"
        description: "当前TTFT P95 = {{ $value }}s，超过SLO阈值0.8s"

    # 2. TPOT SLO违反告警 (P95 > 150ms)
    - alert: VLLMTPOTHigh
      expr: |
        histogram_quantile(0.95,
          sum(rate(vllm:inter_token_latency_seconds_bucket[5m])) by (le)
        ) > 0.15
      for: 5m
      labels:
        severity: critical
        component: vllm
      annotations:
        summary: "vLLM TPOT P95超过150ms"
        description: "当前TPOT P95 = {{ $value }}s，超过SLO阈值0.15s"

    # 3. KV Cache利用率过高
    - alert: VLLMKVCacheHigh
      expr: vllm:kv_cache_usage_perc > 0.9
      for: 2m
      labels:
        severity: warning
        component: vllm
      annotations:
        summary: "vLLM KV Cache利用率超过90%"
        description: "当前KV Cache利用率 = {{ $value | humanizePercentage }}"

    # 4. 请求错误率过高
    - alert: VLLMHighErrorRate
      expr: |
        sum(rate(vllm:request_success_total{finish_reason!="stop"}[5m])) /
        sum(rate(vllm:request_success_total[5m])) > 0.01
      for: 3m
      labels:
        severity: critical
        component: vllm
      annotations:
        summary: "vLLM请求错误率超过1%"
        description: "当前错误率 = {{ $value | humanizePercentage }}"

    # 5. 等待队列过长
    - alert: VLLMQueueDepthHigh
      expr: vllm:num_requests_waiting > 10
      for: 1m
      labels:
        severity: warning
        component: vllm
      annotations:
        summary: "vLLM等待队列过长"
        description: "当前等待请求数 = {{ $value }}"
```



#### 3.3 Grafana SLO仪表板

创建Grafana Dashboard JSON配置（关键面板）：



```json
{
  "title": "vLLM SLO Dashboard",
  "panels": [
    {
      "title": "TTFT P50/P95/P99",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))",
          "legendFormat": "P50"
        },
        {
          "expr": "histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))",
          "legendFormat": "P95"
        },
        {
          "expr": "histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))",
          "legendFormat": "P99"
        }
      ],
      "fieldConfig": {
        "thresholds": {
          "mode": "absolute",
          "steps": [
            {"color": "green", "value": null},
            {"color": "yellow", "value": 0.5},
            {"color": "red", "value": 0.8}
          ]
        }
      }
    },
    {
      "title": "SLO Attainment Rate",
      "targets": [
        {
          "expr": "sum(vllm:request_success_total{finish_reason=\"stop\"}) / sum(vllm:request_success_total) * 100",
          "legendFormat": "Success Rate"
        }
      ],
      "fieldConfig": {
        "thresholds": {
          "mode": "absolute",
          "steps": [
            {"color": "red", "value": null},
            {"color": "yellow", "value": 99.0},
            {"color": "green", "value": 99.9}
          ]
        }
      }
    }
  ]
}
```







### 四、阶段三：请求调度与动态批处理

#### 4.1 vLLM调度器配置

vLLM支持多种调度策略来保障SLO：



**动态批处理（Dynamic Batch）** ：根据资源和SLO目标动态调整每次推理迭代的块大小。



```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-chunked-prefill \
    --SLO_limits_for_dynamic_batch 50 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 4096
```



`--SLO_limits_for_dynamic_batch` 是动态批处理的调优参数，值越大延迟限制越宽松，有效吞吐量越高。经验值通常为35、50或75。



**分块预填充（Chunked Prefill）** ：将长Prompt切分为小块，与Decode请求混合调度，避免长请求阻塞短请求。



```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048
```



#### 4.2 优先级调度配置

vLLM v1已支持优先级调度，可通过SLA分层实现差异化SLO保障：



```bash
# 启用优先级调度
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --scheduler-config '{"policy": "priority"}'
```



在客户端请求中指定优先级：



```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    scheduler_config={"policy": "priority"}
)

# 交互式请求（高优先级，低延迟SLO）
interactive_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    priority=0  # 数字越小优先级越高
)

# 批处理请求（低优先级）
batch_params = SamplingParams(
    temperature=0.8,
    max_tokens=500,
    priority=10
)
```



#### 4.3 Goodput驱动的容量规划

使用vLLM benchmark进行SLO感知的容量规划：



```bash
# 测试不同并发下的Goodput
for rate in 1 2 4 8 16 32 64; do
    vllm bench serve \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --base-url http://localhost:8000 \
        --num-prompts 200 \
        --request-rate $rate \
        --goodput ttft:500 tpot:100 \
        --save-result
done
```



`--goodput` 参数会统计满足所有SLO约束的请求数，帮助找到**最大Goodput对应的最优并发度**。







### 五、阶段四：基于KEDA的弹性扩缩容

KEDA（Kubernetes Event-driven Autoscaling）是CNCF项目，可直接查询Prometheus指标并创建标准HPA。



#### 5.1 安装KEDA

```bash
# 添加KEDA Helm仓库
helm repo add kedacore https://kedacore.github.io/charts
helm repo update

# 安装KEDA
helm install keda kedacore/keda --namespace keda --create-namespace

# 验证安装
kubectl get pods -n keda
```



#### 5.2 创建KEDA ScaledObject

KEDA基于**队列深度（主要扩缩信号）** 和 **P95端到端延迟（SLO护栏）** 两个指标进行扩缩容：



```yaml
# keda-scaled-object.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-inference-app
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-inference-app
  minReplicaCount: 1
  maxReplicaCount: 10
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 30
          policies:
          - type: Pods
            value: 2
            periodSeconds: 60
          selectPolicy: Max
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
          - type: Pods
            value: 1
            periodSeconds: 120
          selectPolicy: Min
  triggers:
  # 触发器1: 队列深度（主要扩缩信号）
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-operated:9090
      metricName: vllm_queue_depth
      threshold: "25"
      query: |
        sum(vllm:num_requests_waiting) 
        / 
        count(vllm:num_requests_waiting)
  # 触发器2: P95端到端延迟（SLO护栏）
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-operated:9090
      metricName: vllm_p95_latency
      threshold: "5"
      query: |
        histogram_quantile(0.95,
          sum(rate(vllm:e2e_request_latency_seconds_bucket[5m])) by (le)
        )
```



部署ScaledObject：



```bash
kubectl apply -f keda-scaled-object.yaml
```



#### 5.3 验证扩缩容

```bash
# 查看ScaledObject状态
kubectl get scaledobject vllm-inference-app

# 查看HPA
kubectl get hpa

# 查看KEDA事件
kubectl describe scaledobject vllm-inference-app
```



**扩缩容逻辑**：



* 当**平均队列深度 > 25**时，KEDA触发扩容

* 当**P95端到端延迟 > 5秒**时，触发扩容（SLO护栏）

* 稳定窗口：扩容30秒，缩容300秒，避免抖动

### 六、完整项目检查清单

| 步骤 | 任务                    | 验证命令                                   |
| -- | --------------------- | -------------------------------------- |
| 1  | 部署vLLM服务              | `curl http://localhost:8000/v1/models` |
| 2  | 验证指标暴露                | \`curl http://localhost:8000/metrics   |
| 3  | 部署Prometheus Operator | `kubectl get pods -n monitoring`       |
| 4  | 创建ServiceMonitor      | `kubectl get servicemonitor`           |
| 5  | 部署告警规则                | `kubectl get prometheusrule`           |
| 6  | 安装Grafana             | `kubectl get pods -n grafana`          |
| 7  | 导入SLO Dashboard       | 访问Grafana UI                           |
| 8  | 安装KEDA                | `kubectl get pods -n keda`             |
| 9  | 创建ScaledObject        | `kubectl get scaledobject`             |
| 10 | 压测验证扩缩容               | 使用`vllm bench`增加负载                     |





### 七、快速启动脚本

```bash
#!/bin/bash
# deploy_slo_stack.sh - 一键部署SLO保障体系

echo "🚀 部署vLLM SLO保障体系"

# 1. 部署vLLM服务（启用指标）
kubectl apply -f vllm-deployment.yaml

# 2. 等待服务就绪
kubectl wait --for=condition=ready pod -l app=vllm --timeout=300s

# 3. 部署监控栈（Prometheus + Grafana）
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
    -f prometheus-values.yaml

# 4. 部署告警规则
kubectl apply -f slo-alerts.yaml

# 5. 安装KEDA
helm upgrade --install keda kedacore/keda --namespace keda --create-namespace

# 6. 部署KEDA ScaledObject
kubectl apply -f keda-scaled-object.yaml

# 7. 验证
echo "✅ 部署完成！"
echo "Grafana: kubectl port-forward svc/prometheus-grafana 3000:80"
echo "vLLM Metrics: curl http://localhost:8000/metrics"
```





