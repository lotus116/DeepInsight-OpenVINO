#!/usr/bin/env python3
"""
技术卓越性集成模块
将所有技术优化模块集成到主应用中
目标：无缝集成所有技术优化功能，提供统一的管理接口
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import streamlit as st

logger = logging.getLogger(__name__)

# 导入技术模块（容错处理）
try:
    from intel_deep_integration import get_intel_deep_integration, ComputeWorkloadType
    INTEL_INTEGRATION_AVAILABLE = True
except ImportError as e:
    INTEL_INTEGRATION_AVAILABLE = False
    logger.warning(f"Intel深度集成不可用: {e}")

try:
    from enterprise_architecture_manager import get_enterprise_architecture
    ENTERPRISE_ARCH_AVAILABLE = True
except ImportError as e:
    ENTERPRISE_ARCH_AVAILABLE = False
    logger.warning(f"企业级架构不可用: {e}")

try:
    from adaptive_performance_optimizer import get_adaptive_performance_manager, OptimizationStrategy
    ADAPTIVE_PERF_AVAILABLE = True
except ImportError as e:
    ADAPTIVE_PERF_AVAILABLE = False
    logger.warning(f"自适应性能优化不可用: {e}")

@dataclass
class TechnicalStatus:
    """技术状态"""
    intel_integration: bool
    enterprise_architecture: bool
    adaptive_performance: bool
    overall_score: float
    maturity_level: str

class TechnicalExcellenceManager:
    """技术卓越性管理器"""
    
    def __init__(self):
        self.intel_manager = None
        self.arch_manager = None
        self.perf_manager = None
        
        self._initialize_managers()
        
        # 性能统计
        self.operation_count = 0
        self.total_optimization_time = 0.0
        self.last_optimization_time = None
        
        logger.info("✅ 技术卓越性管理器初始化完成")
    
    def _initialize_managers(self):
        """初始化管理器"""
        try:
            if INTEL_INTEGRATION_AVAILABLE:
                self.intel_manager = get_intel_deep_integration()
                logger.info("✅ Intel深度集成管理器已加载")
            
            if ENTERPRISE_ARCH_AVAILABLE:
                self.arch_manager = get_enterprise_architecture()
                logger.info("✅ 企业级架构管理器已加载")
            
            if ADAPTIVE_PERF_AVAILABLE:
                self.perf_manager = get_adaptive_performance_manager()
                # 启动性能监控
                self.perf_manager.start_monitoring(interval_seconds=300)  # 5分钟间隔
                logger.info("✅ 自适应性能管理器已加载")
                
        except Exception as e:
            logger.error(f"管理器初始化失败: {e}")
    
    def optimize_operation(self, operation_type: str, operation_func, *args, **kwargs):
        """优化操作执行"""
        start_time = time.perf_counter()
        
        try:
            # 记录操作开始
            self.operation_count += 1
            
            # Intel硬件优化
            if self.intel_manager and operation_type in ['embedding', 'matrix_ops', 'text_processing']:
                workload_type = {
                    'embedding': ComputeWorkloadType.EMBEDDING,
                    'matrix_ops': ComputeWorkloadType.MATRIX_OPS,
                    'text_processing': ComputeWorkloadType.TEXT_PROCESSING
                }.get(operation_type, ComputeWorkloadType.TEXT_PROCESSING)
                
                result, metrics = self.intel_manager.optimize_workload(
                    workload_type, operation_func, *args, **kwargs
                )
                
                # 记录性能数据
                if self.perf_manager:
                    self.perf_manager.record_operation_performance(
                        operation_type=operation_type,
                        operation_id=f"op_{self.operation_count}",
                        latency_ms=metrics.execution_time_ms,
                        memory_mb=metrics.memory_usage_mb,
                        cache_hit=kwargs.get('cache_hit', False),
                        input_size=kwargs.get('input_size', 0)
                    )
                
                return result
            else:
                # 标准执行
                result = operation_func(*args, **kwargs)
                
                # 记录性能数据
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000
                
                if self.perf_manager:
                    self.perf_manager.record_operation_performance(
                        operation_type=operation_type,
                        operation_id=f"op_{self.operation_count}",
                        latency_ms=execution_time,
                        cache_hit=kwargs.get('cache_hit', False),
                        input_size=kwargs.get('input_size', 0)
                    )
                
                return result
                
        except Exception as e:
            # 记录错误
            if self.perf_manager:
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000
                
                self.perf_manager.record_operation_performance(
                    operation_type=operation_type,
                    operation_id=f"op_{self.operation_count}",
                    latency_ms=execution_time,
                    error_occurred=True
                )
            
            raise e
        finally:
            # 更新统计
            end_time = time.perf_counter()
            operation_time = end_time - start_time
            self.total_optimization_time += operation_time
            self.last_optimization_time = time.time()
    
    def get_technical_status(self) -> TechnicalStatus:
        """获取技术状态"""
        # 计算各模块状态
        intel_status = INTEL_INTEGRATION_AVAILABLE and self.intel_manager is not None
        arch_status = ENTERPRISE_ARCH_AVAILABLE and self.arch_manager is not None
        perf_status = ADAPTIVE_PERF_AVAILABLE and self.perf_manager is not None
        
        # 计算总体评分
        module_scores = [intel_status, arch_status, perf_status]
        overall_score = sum(module_scores) / len(module_scores) * 100
        
        # 确定成熟度等级
        if overall_score >= 80:
            maturity_level = "企业级"
        elif overall_score >= 60:
            maturity_level = "专业级"
        elif overall_score >= 40:
            maturity_level = "标准级"
        else:
            maturity_level = "基础级"
        
        return TechnicalStatus(
            intel_integration=intel_status,
            enterprise_architecture=arch_status,
            adaptive_performance=perf_status,
            overall_score=overall_score,
            maturity_level=maturity_level
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            "operation_count": self.operation_count,
            "total_optimization_time": self.total_optimization_time,
            "avg_optimization_time": self.total_optimization_time / max(self.operation_count, 1),
            "last_optimization": self.last_optimization_time
        }
        
        # 添加各模块的性能数据
        if self.intel_manager:
            intel_status = self.intel_manager.get_optimization_status()
            summary["intel_optimization"] = {
                "optimization_level": intel_status["hardware_profile"]["optimization_level"],
                "mkl_available": intel_status["hardware_profile"]["has_mkl"],
                "average_acceleration": intel_status.get("average_acceleration", 1.0)
            }
        
        if self.perf_manager:
            perf_summary = self.perf_manager.get_performance_summary()
            summary["adaptive_performance"] = {
                "total_operations": perf_summary.get("total_operations", 0),
                "avg_latency_ms": perf_summary.get("avg_latency_ms", 0),
                "error_rate": perf_summary.get("error_rate", 0),
                "cache_hit_rate": perf_summary.get("cache_hit_rate", 0)
            }
        
        if self.arch_manager:
            arch_status = self.arch_manager.get_system_status()
            summary["enterprise_architecture"] = {
                "registered_services": arch_status["architecture"]["services"]["registered_services"],
                "health_status": arch_status["health"]["overall_healthy"]
            }
        
        return summary
    
    def record_operation_performance(self, operation_type: str, operation_id: str,
                                     latency_ms: float, memory_mb: float = None,
                                     error_occurred: bool = False, cache_hit: bool = False,
                                     input_size: int = 0, context: Dict[str, Any] = None):
        """记录操作性能数据（代理到perf_manager）"""
        # 更新内部统计
        self.operation_count += 1
        self.last_optimization_time = time.time()
        
        # 代理到自适应性能管理器
        if self.perf_manager:
            try:
                self.perf_manager.record_operation_performance(
                    operation_type=operation_type,
                    operation_id=operation_id,
                    latency_ms=latency_ms,
                    memory_mb=memory_mb,
                    error_occurred=error_occurred,
                    cache_hit=cache_hit,
                    input_size=input_size
                )
            except Exception as e:
                logger.warning(f"性能数据记录失败: {e}")
        else:
            # 如果perf_manager不可用，仅记录日志
            logger.debug(f"性能记录: {operation_type} - {latency_ms:.2f}ms")
    
    def render_technical_status_ui(self):
        """渲染技术状态UI"""
        st.subheader("🏆 技术卓越性状态")
        
        status = self.get_technical_status()
        
        # 总体状态
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "技术评分", 
                f"{status.overall_score:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "成熟度等级", 
                status.maturity_level,
                delta=None
            )
        
        with col3:
            st.metric(
                "优化操作数", 
                self.operation_count,
                delta=None
            )
        
        # 模块状态
        st.write("**技术模块状态:**")
        
        modules = [
            ("Intel深度集成", status.intel_integration, "🚀"),
            ("企业级架构", status.enterprise_architecture, "🏗️"),
            ("自适应性能优化", status.adaptive_performance, "⚡")
        ]
        
        for name, enabled, icon in modules:
            status_text = f"{icon} {name}: {'✅ 已启用' if enabled else '❌ 未启用'}"
            if enabled:
                st.success(status_text)
            else:
                st.warning(status_text)
        
        # 性能摘要
        if st.expander("📊 性能摘要", expanded=False):
            summary = self.get_performance_summary()
            
            if "intel_optimization" in summary:
                intel_data = summary["intel_optimization"]
                st.write(f"**Intel优化**: {intel_data['optimization_level']} "
                        f"(加速比: {intel_data['average_acceleration']:.2f}x)")
            
            if "adaptive_performance" in summary:
                perf_data = summary["adaptive_performance"]
                st.write(f"**性能监控**: {perf_data['total_operations']}次操作, "
                        f"平均延迟: {perf_data['avg_latency_ms']:.1f}ms")
            
            if "enterprise_architecture" in summary:
                arch_data = summary["enterprise_architecture"]
                st.write(f"**架构状态**: {arch_data['registered_services']}个服务, "
                        f"健康状态: {'✅' if arch_data['health_status'] else '❌'}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        status = self.get_technical_status()
        
        if not status.intel_integration:
            recommendations.append("🚀 建议启用Intel深度集成以获得硬件加速")
        
        if not status.enterprise_architecture:
            recommendations.append("🏗️ 建议启用企业级架构以提升系统稳定性")
        
        if not status.adaptive_performance:
            recommendations.append("⚡ 建议启用自适应性能优化以提升响应速度")
        
        if status.overall_score < 80:
            recommendations.append("📈 建议完善技术模块集成以达到企业级标准")
        
        # 基于性能数据的建议
        if self.perf_manager:
            try:
                perf_recommendations = self.perf_manager.optimize_operation(
                    "general", OptimizationStrategy.ADAPTIVE
                )
                
                for rec in perf_recommendations[:3]:  # 最多显示3个建议
                    recommendations.append(
                        f"⚙️ 建议调整{rec.parameter}: "
                        f"{rec.current_value} → {rec.recommended_value} "
                        f"(预期提升: {rec.expected_improvement:.1%})"
                    )
            except:
                pass
        
        if not recommendations:
            recommendations.append("✅ 技术实现已达到最优状态")
        
        return recommendations
    
    def shutdown(self):
        """关闭管理器"""
        logger.info("正在关闭技术卓越性管理器...")
        
        if self.perf_manager:
            self.perf_manager.shutdown()
        
        if self.arch_manager:
            self.arch_manager.shutdown()
        
        if self.intel_manager:
            self.intel_manager.shutdown()
        
        logger.info("✅ 技术卓越性管理器已关闭")

# 全局实例
_technical_excellence_manager = None

def get_technical_excellence_manager() -> TechnicalExcellenceManager:
    """获取技术卓越性管理器实例"""
    global _technical_excellence_manager
    if _technical_excellence_manager is None:
        _technical_excellence_manager = TechnicalExcellenceManager()
    return _technical_excellence_manager

def optimize_operation(operation_type: str, operation_func, *args, **kwargs):
    """优化操作执行（便捷方法）"""
    manager = get_technical_excellence_manager()
    return manager.optimize_operation(operation_type, operation_func, *args, **kwargs)

def render_technical_excellence_ui():
    """渲染技术卓越性UI（便捷方法）"""
    manager = get_technical_excellence_manager()
    manager.render_technical_status_ui()

def get_technical_recommendations() -> List[str]:
    """获取技术优化建议（便捷方法）"""
    manager = get_technical_excellence_manager()
    return manager.get_optimization_recommendations()

# 测试函数
def test_technical_excellence_integration():
    """测试技术卓越性集成"""
    print("🧪 测试技术卓越性集成...")
    
    manager = get_technical_excellence_manager()
    
    # 测试状态获取
    status = manager.get_technical_status()
    print(f"✅ 技术状态: {status.overall_score:.1f}% ({status.maturity_level})")
    
    # 测试操作优化
    def test_operation(data):
        return f"processed: {data}"
    
    result = manager.optimize_operation("text_processing", test_operation, "test data")
    print(f"✅ 优化操作结果: {result}")
    
    # 测试性能摘要
    summary = manager.get_performance_summary()
    print(f"✅ 性能摘要: {summary['operation_count']}次操作")
    
    # 测试优化建议
    recommendations = manager.get_optimization_recommendations()
    print(f"✅ 优化建议: {len(recommendations)}条")
    
    for rec in recommendations[:3]:
        print(f"   {rec}")
    
    print("🎉 技术卓越性集成测试完成！")

if __name__ == "__main__":
    test_technical_excellence_integration()