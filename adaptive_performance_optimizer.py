#!/usr/bin/env python3
"""
自适应性能优化系统
基于机器学习和历史数据的智能性能调优
目标：实现智能化的性能优化，体现技术方案的先进性
"""

import time
import threading
import logging
import json
import pickle
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import psutil
import hashlib

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """优化策略"""
    CONSERVATIVE = "conservative"    # 保守策略
    BALANCED = "balanced"           # 平衡策略
    AGGRESSIVE = "aggressive"       # 激进策略
    ADAPTIVE = "adaptive"           # 自适应策略

class PerformanceMetricType(Enum):
    """性能指标类型"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"

@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: float
    operation_type: str
    operation_id: str
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    throughput_ops_sec: float
    error_occurred: bool
    cache_hit: bool
    input_size: int
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationRecommendation:
    """优化建议"""
    parameter: str
    current_value: Any
    recommended_value: Any
    confidence: float
    expected_improvement: float
    reason: str
    strategy: OptimizationStrategy

@dataclass
class PerformanceModel:
    """性能模型"""
    operation_type: str
    model_version: str
    training_samples: int
    accuracy_score: float
    last_updated: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_data: Optional[bytes] = None

class PerformancePredictor:
    """性能预测器"""
    
    def __init__(self):
        self.models: Dict[str, PerformanceModel] = {}
        self.feature_extractors: Dict[str, Callable] = {}
        self._setup_feature_extractors()
        
        logger.info("✅ 性能预测器初始化完成")
    
    def _setup_feature_extractors(self):
        """设置特征提取器"""
        self.feature_extractors = {
            'input_size': lambda snapshot: snapshot.input_size,
            'hour_of_day': lambda snapshot: time.localtime(snapshot.timestamp).tm_hour,
            'day_of_week': lambda snapshot: time.localtime(snapshot.timestamp).tm_wday,
            'system_load': lambda snapshot: snapshot.cpu_percent,
            'memory_pressure': lambda snapshot: snapshot.memory_mb,
            'cache_efficiency': lambda snapshot: 1.0 if snapshot.cache_hit else 0.0,
        }
    
    def extract_features(self, snapshot: PerformanceSnapshot) -> Dict[str, float]:
        """提取特征"""
        features = {}
        
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = float(extractor(snapshot))
            except Exception as e:
                logger.warning(f"特征提取失败 {feature_name}: {e}")
                features[feature_name] = 0.0
        
        return features
    
    def train_model(self, operation_type: str, snapshots: List[PerformanceSnapshot]) -> PerformanceModel:
        """训练性能模型"""
        if len(snapshots) < 10:
            logger.warning(f"训练数据不足: {operation_type} ({len(snapshots)} samples)")
            return None
        
        try:
            # 提取特征和目标值
            features_list = []
            targets = []
            
            for snapshot in snapshots:
                features = self.extract_features(snapshot)
                features_list.append(list(features.values()))
                targets.append(snapshot.latency_ms)
            
            X = np.array(features_list)
            y = np.array(targets)
            
            # 简单的线性回归模型（可以替换为更复杂的模型）
            model_coefficients = self._fit_linear_regression(X, y)
            
            # 计算特征重要性
            feature_names = list(self.feature_extractors.keys())
            feature_importance = {
                name: abs(coef) for name, coef in zip(feature_names, model_coefficients[1:])
            }
            
            # 计算模型准确性
            predictions = self._predict_linear(X, model_coefficients)
            accuracy = self._calculate_r2_score(y, predictions)
            
            model = PerformanceModel(
                operation_type=operation_type,
                model_version="1.0",
                training_samples=len(snapshots),
                accuracy_score=accuracy,
                last_updated=time.time(),
                feature_importance=feature_importance,
                model_data=pickle.dumps(model_coefficients)
            )
            
            self.models[operation_type] = model
            
            logger.info(f"✅ 性能模型训练完成: {operation_type} (准确率: {accuracy:.3f})")
            
            return model
            
        except Exception as e:
            logger.error(f"模型训练失败: {operation_type}: {e}")
            return None
    
    def _fit_linear_regression(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """拟合线性回归"""
        # 添加偏置项
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # 正规方程求解
        try:
            coefficients = np.linalg.solve(X_with_bias.T @ X_with_bias, X_with_bias.T @ y)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            coefficients = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        return coefficients
    
    def _predict_linear(self, X: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """线性预测"""
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        return X_with_bias @ coefficients
    
    def _calculate_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算R²分数"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def predict_performance(self, operation_type: str, context: Dict[str, Any]) -> Optional[float]:
        """预测性能"""
        if operation_type not in self.models:
            return None
        
        model = self.models[operation_type]
        
        try:
            # 创建临时快照用于特征提取
            temp_snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                operation_type=operation_type,
                operation_id="prediction",
                latency_ms=0,
                memory_mb=context.get('memory_mb', 0),
                cpu_percent=context.get('cpu_percent', 0),
                throughput_ops_sec=0,
                error_occurred=False,
                cache_hit=context.get('cache_hit', False),
                input_size=context.get('input_size', 0),
                context=context
            )
            
            features = self.extract_features(temp_snapshot)
            X = np.array([list(features.values())])
            
            coefficients = pickle.loads(model.model_data)
            prediction = self._predict_linear(X, coefficients)[0]
            
            return max(0, prediction)  # 确保预测值非负
            
        except Exception as e:
            logger.error(f"性能预测失败: {operation_type}: {e}")
            return None

class AdaptiveOptimizer:
    """自适应优化器"""
    
    def __init__(self):
        self.optimization_params: Dict[str, Dict[str, Any]] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.optimization_history: List[Dict[str, Any]] = []
        self.predictor = PerformancePredictor()
        
        # 优化参数范围
        self.param_ranges = {
            'thread_count': (1, psutil.cpu_count()),
            'batch_size': (1, 1000),
            'cache_size': (100, 10000),
            'timeout_ms': (1000, 60000),
            'retry_count': (0, 5),
            'compression_level': (0, 9)
        }
        
        # 初始化默认参数
        self._initialize_default_params()
        
        logger.info("✅ 自适应优化器初始化完成")
    
    def _initialize_default_params(self):
        """初始化默认参数"""
        default_params = {
            'thread_count': min(4, psutil.cpu_count()),
            'batch_size': 100,
            'cache_size': 1000,
            'timeout_ms': 30000,
            'retry_count': 3,
            'compression_level': 6
        }
        
        self.optimization_params['default'] = default_params
    
    def record_performance(self, snapshot: PerformanceSnapshot):
        """记录性能数据"""
        self.performance_history.append(snapshot)
        
        # 定期重新训练模型
        if len(self.performance_history) % 50 == 0:
            self._retrain_models()
    
    def _retrain_models(self):
        """重新训练模型"""
        # 按操作类型分组
        operation_snapshots = defaultdict(list)
        
        for snapshot in self.performance_history:
            operation_snapshots[snapshot.operation_type].append(snapshot)
        
        # 为每种操作类型训练模型
        for operation_type, snapshots in operation_snapshots.items():
            if len(snapshots) >= 10:
                self.predictor.train_model(operation_type, snapshots)
    
    def optimize_parameters(self, operation_type: str, 
                          current_performance: PerformanceSnapshot,
                          strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE) -> List[OptimizationRecommendation]:
        """优化参数"""
        recommendations = []
        
        # 获取当前参数
        current_params = self.optimization_params.get(operation_type, self.optimization_params['default'])
        
        # 分析历史性能
        similar_snapshots = self._find_similar_snapshots(current_performance)
        
        if len(similar_snapshots) < 5:
            logger.info(f"历史数据不足，使用默认优化策略: {operation_type}")
            return self._get_default_recommendations(current_params, strategy)
        
        # 基于历史数据生成优化建议
        for param_name, (min_val, max_val) in self.param_ranges.items():
            if param_name in current_params:
                recommendation = self._optimize_single_parameter(
                    param_name, current_params[param_name], 
                    similar_snapshots, min_val, max_val, strategy
                )
                
                if recommendation:
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _find_similar_snapshots(self, target_snapshot: PerformanceSnapshot, 
                               similarity_threshold: float = 0.8) -> List[PerformanceSnapshot]:
        """查找相似的性能快照"""
        similar_snapshots = []
        
        for snapshot in self.performance_history:
            if snapshot.operation_type != target_snapshot.operation_type:
                continue
            
            similarity = self._calculate_similarity(target_snapshot, snapshot)
            
            if similarity >= similarity_threshold:
                similar_snapshots.append(snapshot)
        
        return similar_snapshots
    
    def _calculate_similarity(self, snapshot1: PerformanceSnapshot, 
                            snapshot2: PerformanceSnapshot) -> float:
        """计算快照相似度"""
        # 基于输入大小、系统负载等计算相似度
        factors = [
            ('input_size', 0.3),
            ('cpu_percent', 0.2),
            ('memory_mb', 0.2),
            ('cache_hit', 0.3)
        ]
        
        similarity_score = 0.0
        
        for factor, weight in factors:
            val1 = getattr(snapshot1, factor)
            val2 = getattr(snapshot2, factor)
            
            if isinstance(val1, bool) and isinstance(val2, bool):
                factor_similarity = 1.0 if val1 == val2 else 0.0
            else:
                # 数值相似度
                max_val = max(abs(val1), abs(val2), 1)
                factor_similarity = 1.0 - abs(val1 - val2) / max_val
            
            similarity_score += factor_similarity * weight
        
        return similarity_score
    
    def _optimize_single_parameter(self, param_name: str, current_value: Any,
                                 similar_snapshots: List[PerformanceSnapshot],
                                 min_val: Any, max_val: Any,
                                 strategy: OptimizationStrategy) -> Optional[OptimizationRecommendation]:
        """优化单个参数"""
        try:
            # 分析参数值与性能的关系
            param_performance = []
            
            for snapshot in similar_snapshots:
                param_val = snapshot.optimization_params.get(param_name, current_value)
                param_performance.append((param_val, snapshot.latency_ms))
            
            if len(param_performance) < 3:
                return None
            
            # 找到最佳参数值
            param_performance.sort(key=lambda x: x[1])  # 按延迟排序
            best_params = param_performance[:max(1, len(param_performance) // 3)]  # 取前1/3
            
            # 计算推荐值
            best_values = [p[0] for p in best_params]
            recommended_value = np.median(best_values)
            
            # 应用策略调整
            recommended_value = self._apply_strategy_adjustment(
                recommended_value, current_value, min_val, max_val, strategy
            )
            
            # 确保在有效范围内
            recommended_value = max(min_val, min(max_val, recommended_value))
            
            if abs(recommended_value - current_value) / max(abs(current_value), 1) < 0.1:
                return None  # 变化太小，不值得调整
            
            # 估算改进效果
            current_avg_latency = np.mean([p[1] for p in param_performance])
            best_avg_latency = np.mean([p[1] for p in best_params])
            expected_improvement = (current_avg_latency - best_avg_latency) / current_avg_latency
            
            confidence = min(len(similar_snapshots) / 20.0, 1.0)  # 基于样本数量
            
            return OptimizationRecommendation(
                parameter=param_name,
                current_value=current_value,
                recommended_value=recommended_value,
                confidence=confidence,
                expected_improvement=expected_improvement,
                reason=f"基于{len(similar_snapshots)}个相似场景的分析",
                strategy=strategy
            )
            
        except Exception as e:
            logger.error(f"参数优化失败 {param_name}: {e}")
            return None
    
    def _apply_strategy_adjustment(self, recommended_value: float, current_value: float,
                                 min_val: float, max_val: float, 
                                 strategy: OptimizationStrategy) -> float:
        """应用策略调整"""
        if strategy == OptimizationStrategy.CONSERVATIVE:
            # 保守策略：只做小幅调整
            max_change = abs(current_value) * 0.2
            change = recommended_value - current_value
            if abs(change) > max_change:
                change = max_change if change > 0 else -max_change
            return current_value + change
            
        elif strategy == OptimizationStrategy.AGGRESSIVE:
            # 激进策略：直接使用推荐值
            return recommended_value
            
        elif strategy == OptimizationStrategy.BALANCED:
            # 平衡策略：取中间值
            return (current_value + recommended_value) / 2
            
        else:  # ADAPTIVE
            # 自适应策略：基于置信度调整
            confidence = min(len(self.performance_history) / 100.0, 1.0)
            return current_value + (recommended_value - current_value) * confidence
    
    def _get_default_recommendations(self, current_params: Dict[str, Any],
                                   strategy: OptimizationStrategy) -> List[OptimizationRecommendation]:
        """获取默认优化建议"""
        recommendations = []
        
        # 基于系统状态的默认建议
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if current_params.get('thread_count', 1) < cpu_count // 2:
            recommendations.append(OptimizationRecommendation(
                parameter='thread_count',
                current_value=current_params.get('thread_count', 1),
                recommended_value=min(cpu_count // 2, 4),
                confidence=0.7,
                expected_improvement=0.2,
                reason="基于CPU核心数的默认优化",
                strategy=strategy
            ))
        
        if memory_gb > 8 and current_params.get('cache_size', 1000) < 5000:
            recommendations.append(OptimizationRecommendation(
                parameter='cache_size',
                current_value=current_params.get('cache_size', 1000),
                recommended_value=5000,
                confidence=0.6,
                expected_improvement=0.15,
                reason="基于内存容量的缓存优化",
                strategy=strategy
            ))
        
        return recommendations
    
    def apply_recommendations(self, operation_type: str, 
                            recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """应用优化建议"""
        if operation_type not in self.optimization_params:
            self.optimization_params[operation_type] = dict(self.optimization_params['default'])
        
        applied_changes = {}
        
        for rec in recommendations:
            if rec.confidence >= 0.5:  # 只应用高置信度的建议
                old_value = self.optimization_params[operation_type].get(rec.parameter)
                self.optimization_params[operation_type][rec.parameter] = rec.recommended_value
                applied_changes[rec.parameter] = {
                    'old_value': old_value,
                    'new_value': rec.recommended_value,
                    'expected_improvement': rec.expected_improvement
                }
                
                logger.info(f"✅ 应用优化: {operation_type}.{rec.parameter} "
                          f"{old_value} -> {rec.recommended_value}")
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': time.time(),
            'operation_type': operation_type,
            'changes': applied_changes,
            'recommendations': [asdict(rec) for rec in recommendations]
        })
        
        return applied_changes
    
    def get_current_params(self, operation_type: str) -> Dict[str, Any]:
        """获取当前优化参数"""
        return self.optimization_params.get(operation_type, self.optimization_params['default']).copy()
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        return {
            "total_snapshots": len(self.performance_history),
            "trained_models": len(self.predictor.models),
            "optimization_operations": len(self.optimization_params),
            "optimization_history": len(self.optimization_history),
            "model_accuracy": {
                op_type: model.accuracy_score 
                for op_type, model in self.predictor.models.items()
            },
            "recent_optimizations": self.optimization_history[-5:] if self.optimization_history else []
        }

class AdaptivePerformanceManager:
    """自适应性能管理器"""
    
    def __init__(self):
        self.optimizer = AdaptiveOptimizer()
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.performance_callbacks: List[Callable] = []
        
        # 性能阈值
        self.performance_thresholds = {
            'latency_ms': 5000,      # 5秒
            'memory_mb': 1000,       # 1GB
            'cpu_percent': 80,       # 80%
            'error_rate': 0.05       # 5%
        }
        
        logger.info("✅ 自适应性能管理器初始化完成")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """开始性能监控"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"✅ 性能监控已启动 (间隔: {interval_seconds}秒)")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("性能监控已停止")
    
    def _monitoring_loop(self, interval_seconds: int):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._check_system_performance()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"性能监控出错: {e}")
                time.sleep(5)
    
    def _check_system_performance(self):
        """检查系统性能"""
        # 获取系统指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        
        # 检查是否超过阈值
        alerts = []
        
        if cpu_percent > self.performance_thresholds['cpu_percent']:
            alerts.append(f"CPU使用率过高: {cpu_percent:.1f}%")
        
        if memory_mb > self.performance_thresholds['memory_mb']:
            alerts.append(f"内存使用过高: {memory_mb:.1f}MB")
        
        if alerts:
            logger.warning(f"⚠️ 性能告警: {'; '.join(alerts)}")
            
            # 触发自动优化
            self._trigger_auto_optimization()
    
    def _trigger_auto_optimization(self):
        """触发自动优化"""
        try:
            # 为所有操作类型生成优化建议
            for operation_type in self.optimizer.optimization_params.keys():
                if operation_type == 'default':
                    continue
                
                # 创建当前性能快照
                current_snapshot = PerformanceSnapshot(
                    timestamp=time.time(),
                    operation_type=operation_type,
                    operation_id="auto_optimization",
                    latency_ms=0,
                    memory_mb=psutil.virtual_memory().used / (1024 * 1024),
                    cpu_percent=psutil.cpu_percent(),
                    throughput_ops_sec=0,
                    error_occurred=False,
                    cache_hit=False,
                    input_size=0
                )
                
                # 生成优化建议
                recommendations = self.optimizer.optimize_parameters(
                    operation_type, current_snapshot, OptimizationStrategy.CONSERVATIVE
                )
                
                if recommendations:
                    # 应用高置信度的建议
                    high_confidence_recs = [r for r in recommendations if r.confidence >= 0.7]
                    if high_confidence_recs:
                        self.optimizer.apply_recommendations(operation_type, high_confidence_recs)
                        logger.info(f"✅ 自动优化已应用: {operation_type}")
                        
        except Exception as e:
            logger.error(f"自动优化失败: {e}")
    
    def record_operation_performance(self, operation_type: str, operation_id: str,
                                   latency_ms: float, memory_mb: float = None,
                                   error_occurred: bool = False, cache_hit: bool = False,
                                   input_size: int = 0, context: Dict[str, Any] = None):
        """记录操作性能"""
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            operation_type=operation_type,
            operation_id=operation_id,
            latency_ms=latency_ms,
            memory_mb=memory_mb or psutil.Process().memory_info().rss / (1024 * 1024),
            cpu_percent=psutil.cpu_percent(),
            throughput_ops_sec=1000 / latency_ms if latency_ms > 0 else 0,
            error_occurred=error_occurred,
            cache_hit=cache_hit,
            input_size=input_size,
            optimization_params=self.optimizer.get_current_params(operation_type),
            context=context or {}
        )
        
        self.optimizer.record_performance(snapshot)
        
        # 通知回调函数
        for callback in self.performance_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"性能回调失败: {e}")
    
    def optimize_operation(self, operation_type: str, 
                          strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE) -> List[OptimizationRecommendation]:
        """优化特定操作"""
        # 获取最近的性能快照
        recent_snapshots = [
            s for s in self.optimizer.performance_history 
            if s.operation_type == operation_type
        ]
        
        if not recent_snapshots:
            logger.warning(f"没有找到操作的性能数据: {operation_type}")
            return []
        
        # 使用最近的快照作为当前性能基准
        current_snapshot = recent_snapshots[-1]
        
        # 生成优化建议
        recommendations = self.optimizer.optimize_parameters(
            operation_type, current_snapshot, strategy
        )
        
        logger.info(f"✅ 生成了{len(recommendations)}个优化建议: {operation_type}")
        
        return recommendations
    
    def apply_optimization(self, operation_type: str, 
                         recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """应用优化建议"""
        return self.optimizer.apply_recommendations(operation_type, recommendations)
    
    def get_performance_summary(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """获取性能摘要"""
        snapshots = list(self.optimizer.performance_history)
        
        if operation_type:
            snapshots = [s for s in snapshots if s.operation_type == operation_type]
        
        if not snapshots:
            return {"message": "没有性能数据"}
        
        # 计算统计信息
        latencies = [s.latency_ms for s in snapshots]
        memory_usage = [s.memory_mb for s in snapshots]
        error_rate = sum(1 for s in snapshots if s.error_occurred) / len(snapshots)
        cache_hit_rate = sum(1 for s in snapshots if s.cache_hit) / len(snapshots)
        
        return {
            "total_operations": len(snapshots),
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "avg_memory_mb": np.mean(memory_usage),
            "error_rate": error_rate,
            "cache_hit_rate": cache_hit_rate,
            "optimization_status": self.optimizer.get_optimization_status(),
            "monitoring_active": self.monitoring_active
        }
    
    def add_performance_callback(self, callback: Callable[[PerformanceSnapshot], None]):
        """添加性能回调"""
        self.performance_callbacks.append(callback)
    
    def shutdown(self):
        """关闭性能管理器"""
        self.stop_monitoring()
        logger.info("✅ 自适应性能管理器已关闭")

# 全局实例
adaptive_performance_manager = AdaptivePerformanceManager()

def get_adaptive_performance_manager() -> AdaptivePerformanceManager:
    """获取自适应性能管理器实例"""
    return adaptive_performance_manager

def record_performance(operation_type: str, operation_id: str, latency_ms: float, **kwargs):
    """记录性能（便捷方法）"""
    adaptive_performance_manager.record_operation_performance(
        operation_type, operation_id, latency_ms, **kwargs
    )

def optimize_performance(operation_type: str, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
    """优化性能（便捷方法）"""
    recommendations = adaptive_performance_manager.optimize_operation(operation_type, strategy)
    if recommendations:
        adaptive_performance_manager.apply_optimization(operation_type, recommendations)
    return recommendations

# 测试函数
def test_adaptive_performance():
    """测试自适应性能优化"""
    print("🧪 测试自适应性能优化...")
    
    manager = get_adaptive_performance_manager()
    
    # 模拟一些性能数据
    for i in range(20):
        manager.record_operation_performance(
            operation_type="text2sql",
            operation_id=f"query_{i}",
            latency_ms=1000 + np.random.normal(0, 200),
            input_size=100 + i * 10,
            cache_hit=i % 3 == 0
        )
    
    # 生成优化建议
    recommendations = manager.optimize_operation("text2sql")
    print(f"✅ 生成了{len(recommendations)}个优化建议")
    
    for rec in recommendations:
        print(f"   {rec.parameter}: {rec.current_value} -> {rec.recommended_value} "
              f"(置信度: {rec.confidence:.2f})")
    
    # 应用优化
    if recommendations:
        changes = manager.apply_optimization("text2sql", recommendations)
        print(f"✅ 应用了{len(changes)}个优化参数")
    
    # 获取性能摘要
    summary = manager.get_performance_summary("text2sql")
    print(f"✅ 性能摘要: 平均延迟 {summary['avg_latency_ms']:.1f}ms")
    
    print("🎉 自适应性能优化测试完成！")

if __name__ == "__main__":
    test_adaptive_performance()