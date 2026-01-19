"""
Intel DeepInsight 增强性能监控系统
实时性能指标收集、历史趋势分析和异常检测
"""
import psutil
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PerformanceMonitor:
    """增强性能监控器"""
    
    def __init__(self):
        self.metrics_file = "data/performance_metrics.json"
        self.max_history_hours = 24
        self._ensure_metrics_file()
    
    def _ensure_metrics_file(self):
        """确保性能指标文件存在"""
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump({"metrics": []}, f, indent=2)
    
    def collect_current_metrics(self, rag_latency: float = 0.0, total_latency: float = 0.0) -> Dict:
        """收集当前性能指标"""
        try:
            # 系统指标
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 网络指标（如果可用）
            try:
                network = psutil.net_io_counters()
                network_sent = network.bytes_sent
                network_recv = network.bytes_recv
            except:
                network_sent = network_recv = 0
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory.percent, 1),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": round(disk.percent, 1),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "rag_latency_ms": round(rag_latency, 2),
                "total_latency_ms": round(total_latency, 2),
                "network_sent_mb": round(network_sent / (1024**2), 2),
                "network_recv_mb": round(network_recv / (1024**2), 2)
            }
            
            return metrics
        except Exception as e:
            print(f"性能指标收集失败: {e}")
            return {}
    
    def save_metrics(self, metrics: Dict):
        """保存性能指标到历史记录"""
        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data["metrics"].append(metrics)
            
            # 清理超过24小时的旧数据
            cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
            data["metrics"] = [
                m for m in data["metrics"] 
                if datetime.fromisoformat(m["timestamp"]) > cutoff_time
            ]
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"保存性能指标失败: {e}")
    
    def get_historical_metrics(self, hours: int = 1) -> List[Dict]:
        """获取历史性能指标"""
        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [
                m for m in data["metrics"]
                if datetime.fromisoformat(m["timestamp"]) > cutoff_time
            ]
            
            return recent_metrics
        except Exception as e:
            print(f"获取历史指标失败: {e}")
            return []
    
    def detect_anomalies(self, current_metrics: Dict) -> List[str]:
        """检测性能异常"""
        anomalies = []
        
        # CPU异常检测
        if current_metrics.get("cpu_percent", 0) > 80:
            anomalies.append("🔥 CPU使用率过高 (>80%)")
        
        # 内存异常检测
        if current_metrics.get("memory_percent", 0) > 85:
            anomalies.append("💾 内存使用率过高 (>85%)")
        
        # 磁盘异常检测
        if current_metrics.get("disk_percent", 0) > 90:
            anomalies.append("💿 磁盘空间不足 (<10%)")
        
        # 延迟异常检测
        if current_metrics.get("total_latency_ms", 0) > 5000:
            anomalies.append("⏱️ 响应延迟过高 (>5s)")
        
        if current_metrics.get("rag_latency_ms", 0) > 1000:
            anomalies.append("🔍 RAG检索延迟过高 (>1s)")
        
        return anomalies
    
    def get_optimization_suggestions(self, current_metrics: Dict, anomalies: List[str]) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        if any("CPU" in anomaly for anomaly in anomalies):
            suggestions.append("💡 建议关闭其他应用程序以释放CPU资源")
            suggestions.append("💡 考虑降低并发查询数量")
        
        if any("内存" in anomaly for anomaly in anomalies):
            suggestions.append("💡 建议重启应用以释放内存")
            suggestions.append("💡 考虑清理浏览器缓存")
        
        if any("延迟" in anomaly for anomaly in anomalies):
            suggestions.append("💡 检查网络连接状态")
            suggestions.append("💡 尝试使用更简单的查询")
        
        if any("RAG" in anomaly for anomaly in anomalies):
            suggestions.append("💡 OpenVINO模型可能需要重新加载")
            suggestions.append("💡 检查模型文件是否完整")
        
        # 通用优化建议
        if not anomalies:
            if current_metrics.get("cpu_percent", 0) > 50:
                suggestions.append("✨ 系统运行良好，可考虑处理更复杂的查询")
            else:
                suggestions.append("🚀 系统性能优秀，运行状态良好")
        
        return suggestions
    
    def create_performance_trend_chart(self, hours: int = 1) -> Optional[go.Figure]:
        """创建性能趋势图"""
        try:
            metrics = self.get_historical_metrics(hours)
            if len(metrics) < 2:
                return None
            
            df = pd.DataFrame(metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU使用率', '内存使用率', 'RAG延迟', '总延迟'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # CPU趋势
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['cpu_percent'], 
                          name='CPU %', line=dict(color='#FF6B35')),
                row=1, col=1
            )
            
            # 内存趋势
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['memory_percent'], 
                          name='Memory %', line=dict(color='#28A745')),
                row=1, col=2
            )
            
            # RAG延迟趋势
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['rag_latency_ms'], 
                          name='RAG延迟 (ms)', line=dict(color='#0068B5')),
                row=2, col=1
            )
            
            # 总延迟趋势
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['total_latency_ms'], 
                          name='总延迟 (ms)', line=dict(color='#FFC107')),
                row=2, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                title_text=f"过去{hours}小时性能趋势"
            )
            
            return fig
        except Exception as e:
            print(f"创建趋势图失败: {e}")
            return None
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        try:
            recent_metrics = self.get_historical_metrics(1)  # 过去1小时
            if not recent_metrics:
                return {}
            
            df = pd.DataFrame(recent_metrics)
            
            summary = {
                "avg_cpu": round(df['cpu_percent'].mean(), 1),
                "max_cpu": round(df['cpu_percent'].max(), 1),
                "avg_memory": round(df['memory_percent'].mean(), 1),
                "max_memory": round(df['memory_percent'].max(), 1),
                "avg_rag_latency": round(df['rag_latency_ms'].mean(), 1),
                "max_rag_latency": round(df['rag_latency_ms'].max(), 1),
                "total_queries": len([m for m in recent_metrics if m.get('total_latency_ms', 0) > 0]),
                "avg_query_latency": round(df[df['total_latency_ms'] > 0]['total_latency_ms'].mean(), 1) if len(df[df['total_latency_ms'] > 0]) > 0 else 0
            }
            
            return summary
        except Exception as e:
            print(f"获取性能摘要失败: {e}")
            return {}
    
    def cleanup_old_metrics(self):
        """清理旧的性能指标"""
        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cutoff_time = datetime.now() - timedelta(hours=self.max_history_hours)
            original_count = len(data["metrics"])
            
            data["metrics"] = [
                m for m in data["metrics"]
                if datetime.fromisoformat(m["timestamp"]) > cutoff_time
            ]
            
            cleaned_count = original_count - len(data["metrics"])
            if cleaned_count > 0:
                with open(self.metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                print(f"已清理 {cleaned_count} 条旧性能记录")
        except Exception as e:
            print(f"清理性能指标失败: {e}")

# 全局实例
performance_monitor = PerformanceMonitor()