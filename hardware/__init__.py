"""
Intel® DeepInsight - 硬件优化模块

统一的硬件优化接口，支持：
- 多厂商硬件检测 (Intel/NVIDIA/AMD)
- CPU多核并行优化
- 基础性能监控

使用方式:
    from hardware import get_optimization_status, optimize_query_performance
"""

# ============================================
# 通用硬件优化器 - 唯一核心模块
# ============================================
from hardware.universal_hardware_optimizer import (
    # 主要类
    UniversalHardwareOptimizer,
    UniversalHardwareDetector,
    HardwareInfo,
    OptimizationResult,
    # 枚举
    HardwareVendor,
    OptimizationType,
    # 数据类
    QueryProfile,
    SystemLoad,
)

# ============================================
# 全局实例和便捷函数
# ============================================

# 创建全局优化器实例
_universal_optimizer = None

def get_universal_optimizer():
    """获取全局通用硬件优化器实例"""
    global _universal_optimizer
    if _universal_optimizer is None:
        _universal_optimizer = UniversalHardwareOptimizer()
    return _universal_optimizer

def get_optimization_status():
    """获取当前硬件优化状态"""
    optimizer = get_universal_optimizer()
    return optimizer.get_optimization_status()

def optimize_query_performance(query: str, estimated_result_size: int = 100):
    """为查询优化性能"""
    optimizer = get_universal_optimizer()
    return optimizer.optimize_query_performance(query, estimated_result_size)

# 导出列表
__all__ = [
    # 通用硬件
    'UniversalHardwareOptimizer',
    'UniversalHardwareDetector', 
    'HardwareInfo',
    'OptimizationResult',
    'HardwareVendor',
    'OptimizationType',
    'QueryProfile',
    'SystemLoad',
    # 便捷函数
    'get_universal_optimizer',
    'get_optimization_status',
    'optimize_query_performance',
]
