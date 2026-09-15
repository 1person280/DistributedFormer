"""deployment — 部署配套

    Dockerfile / docker-compose.yml   容器化 (代码在 src/, 见 COPY)
    k8s/                              Kubernetes 编排 (deployment/service/
                                      configmap/hpa)
    redis_kv.py                       Redis 外置 KV 堆
    monitoring.py / cloud_config.py   Prometheus 指标 / 云端配置
"""
