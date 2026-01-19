"""
Intel优化系统集成模块
将Intel CPU+Iris Xe优化器集成到现有的DeepInsight系统中

主要功能：
1. 与现有RAG引擎和Text2SQL Agent集成
2. 实时性能监控和优化
3. 动态工作负载分析和调整
4. 优化效果可视化展示
"""

import time
import threading
from typing import Dict, Any, Optional, List
import pandas as pd
import streamlit as st
from intel_cpu_iris_optimizer import (
    IntelCPUIrisXeOptimizer, 
    Text2SQLWorkload, 
    OptimizationResult,
    create_sample_workload
)
import logging

logger = logging.getLogger(__name__)

class IntelOptimizationManager:
    """Intel优化管理器 - 与现有系统集成"""
    
    def __init__(self):
        self.optimizer = None
        self.current_optimization = None
        self.performance_history = []
        self.optimization_enabled = True
        self.real_time_monitoring = True
        
        # 初始化优化器
        self._initialize_optimizer()
    
    def _initialize_optimizer(self):
        """初始化Intel优化器 - 改进的错误处理"""
        try:
            self.optimizer = IntelCPUIrisXeOptimizer()
            
            # 检查硬件检测结果
            if self.optimizer.hardware_info:
                logger.info("✅ Intel优化器初始化成功")
                
                # 记录硬件信息
                hw_info = self.optimizer.hardware_info
                logger.info(f"   CPU: {hw_info.cpu_model}")
                logger.info(f"   核心: {hw_info.cpu_cores}核/{hw_info.cpu_threads}线程")
                logger.info(f"   Intel GPU: {'可用' if hw_info.has_iris_xe else '不可用'}")
                
                if not hw_info.has_iris_xe:
                    logger.info("   注意: 未检测到Intel集成显卡，将仅使用CPU优化")
                    
            else:
                logger.warning("⚠️ 硬件检测失败，但优化器仍可使用")
                
        except Exception as e:
            logger.error(f"❌ Intel优化器初始化失败: {e}")
            self.optimization_enabled = False
            self.optimizer = None
    
    def analyze_query_workload(self, query: str, expected_result_size: int = 100, 
                             concurrent_users: int = 1) -> Text2SQLWorkload:
        """分析查询工作负载特征"""
        # 分析查询复杂度
        complexity = self._analyze_query_complexity(query)
        
        # 估算文本长度
        text_length = len(query.split()) * 10  # 简化估算
        
        # 估算内存需求
        memory_requirement = max(expected_result_size / 1000, 0.5)  # 最少0.5GB
        
        workload = Text2SQLWorkload(
            query_complexity=complexity,
            text_length=text_length,
            expected_result_size=expected_result_size,
            concurrent_users=concurrent_users,
            memory_requirement=memory_requirement
        )
        
        return workload
    
    def _analyze_query_complexity(self, query: str) -> str:
        """分析查询复杂度"""
        query_lower = query.lower()
        
        # 复杂查询特征
        complex_keywords = ['join', 'subquery', 'union', 'group by', 'having', 'window', 'case when']
        medium_keywords = ['where', 'order by', 'distinct', 'count', 'sum', 'avg']
        
        complex_count = sum(1 for keyword in complex_keywords if keyword in query_lower)
        medium_count = sum(1 for keyword in medium_keywords if keyword in query_lower)
        
        if complex_count >= 2 or len(query.split()) > 50:
            return "complex"
        elif complex_count >= 1 or medium_count >= 2 or len(query.split()) > 20:
            return "medium"
        else:
            return "simple"
    
    def optimize_for_query(self, query: str, expected_result_size: int = 100) -> Optional[OptimizationResult]:
        """为特定查询优化系统"""
        if not self.optimization_enabled:
            return None
        
        try:
            # 分析工作负载
            workload = self.analyze_query_workload(query, expected_result_size)
            
            # 执行优化
            optimization_result = self.optimizer.optimize_text2sql_workload(workload)
            
            # 保存当前优化结果
            self.current_optimization = optimization_result
            
            # 记录性能历史
            self._record_performance(workload, optimization_result)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"查询优化失败: {e}")
            return None
    
    def _record_performance(self, workload: Text2SQLWorkload, result: OptimizationResult):
        """记录性能历史"""
        performance_record = {
            'timestamp': time.time(),
            'query_complexity': workload.query_complexity,
            'text_length': workload.text_length,
            'expected_result_size': workload.expected_result_size,
            'cpu_gain': result.cpu_performance_gain,
            'gpu_gain': result.gpu_acceleration_gain,
            'overall_speedup': result.overall_speedup,
            'memory_efficiency': result.memory_efficiency
        }
        
        self.performance_history.append(performance_record)
        
        # 保持最近100条记录
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_current_optimization_status(self) -> Dict[str, Any]:
        """获取当前优化状态 - 改进的GPU状态报告"""
        if not self.optimization_enabled:
            return {
                'enabled': False,
                'message': 'Intel优化器未启用或初始化失败'
            }
        
        if not self.current_optimization:
            return {
                'enabled': True,
                'optimized': False,
                'message': '等待查询以进行优化',
                'hardware_info': self._get_hardware_info_safe()
            }
        
        # 获取硬件信息
        hw_info = self._get_hardware_info_safe()
        
        return {
            'enabled': True,
            'optimized': True,
            'cpu_gain': f"{self.current_optimization.cpu_performance_gain:.1%}",
            'gpu_speedup': f"{self.current_optimization.gpu_acceleration_gain + 1:.2f}x" if hw_info.get('has_iris_xe') else "不可用",
            'overall_speedup': f"{self.current_optimization.overall_speedup:.2f}x",
            'memory_efficiency': f"{self.current_optimization.memory_efficiency:.1%}",
            'hardware_info': hw_info
        }
    
    def _get_hardware_info_safe(self) -> Dict[str, Any]:
        """安全地获取硬件信息"""
        try:
            if self.optimizer and self.optimizer.hardware_info:
                hw = self.optimizer.hardware_info
                return {
                    'cpu_model': hw.cpu_model,
                    'cpu_cores': hw.cpu_cores,
                    'has_iris_xe': hw.has_iris_xe,
                    'has_avx2': hw.has_avx2,
                    'memory_total': f"{hw.memory_total:.1f}GB"
                }
            else:
                return {
                    'cpu_model': 'Unknown',
                    'cpu_cores': 0,
                    'has_iris_xe': False,
                    'has_avx2': False,
                    'memory_total': 'Unknown'
                }
        except Exception as e:
            logger.warning(f"获取硬件信息失败: {e}")
            return {
                'cpu_model': 'Error',
                'cpu_cores': 0,
                'has_iris_xe': False,
                'has_avx2': False,
                'memory_total': 'Error'
            }
    
    def get_performance_history_df(self) -> pd.DataFrame:
        """获取性能历史数据框"""
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def get_optimization_recommendations(self, query: str) -> List[str]:
        """获取优化建议 - 改进的GPU兼容性"""
        recommendations = []
        
        if not self.optimization_enabled:
            recommendations.append("⚠️ Intel优化器未启用，建议检查硬件支持")
            return recommendations
        
        # 分析查询并给出建议
        workload = self.analyze_query_workload(query)
        hw_info = self._get_hardware_info_safe()
        
        if workload.query_complexity == "complex":
            if hw_info.get('has_iris_xe'):
                recommendations.append("🚀 复杂查询检测到，建议启用GPU并行计算")
            else:
                recommendations.append("🔧 复杂查询检测到，将使用CPU多线程优化")
            recommendations.append("💾 建议增加内存缓存以提升性能")
        
        if workload.text_length > 500:
            if hw_info.get('has_avx2'):
                recommendations.append("⚡ 长文本查询，已启用AVX向量化优化")
            else:
                recommendations.append("📝 长文本查询，建议升级到支持AVX2的CPU")
        
        if hw_info.get('has_iris_xe'):
            recommendations.append("🎯 检测到Intel集成显卡，已启用GPU加速")
        else:
            recommendations.append("💻 未检测到Intel集成显卡，使用CPU优化模式")
        
        if not recommendations:
            recommendations.append("✅ 当前查询已优化，无需额外调整")
        
        return recommendations

class IntelOptimizationUI:
    """Intel优化系统UI组件"""
    
    def __init__(self, optimization_manager: IntelOptimizationManager):
        self.manager = optimization_manager
    
    def render_optimization_panel(self):
        """渲染优化面板"""
        st.subheader("🚀 Intel平台优化状态")
        
        # 获取当前状态
        status = self.manager.get_current_optimization_status()
        
        if not status['enabled']:
            st.error(status['message'])
            return
        
        # 硬件信息
        with st.expander("💻 硬件配置信息", expanded=False):
            hw_info = status.get('hardware_info', {})
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("CPU型号", hw_info.get('cpu_model', 'Unknown')[:30] + "...")
                st.metric("CPU核心数", f"{hw_info.get('cpu_cores', 0)}核")
            
            with col2:
                st.metric("AVX2支持", "✅" if hw_info.get('has_avx2') else "❌")
                st.metric("Iris Xe显卡", "✅" if hw_info.get('has_iris_xe') else "❌")
        
        # 当前优化状态
        if status['optimized']:
            st.success("🎯 系统已优化")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("CPU性能提升", status['cpu_gain'])
            with col2:
                st.metric("GPU加速比", status['gpu_speedup'])
            with col3:
                st.metric("总体加速比", status['overall_speedup'])
            with col4:
                st.metric("内存效率", status['memory_efficiency'])
        else:
            st.info(status['message'])
    
    def render_performance_history(self):
        """渲染性能历史图表"""
        df = self.manager.get_performance_history_df()
        
        if df.empty:
            st.info("暂无性能历史数据")
            return
        
        st.subheader("📊 性能优化历史")
        
        # 性能趋势图
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(df.set_index('datetime')[['overall_speedup']], 
                         use_container_width=True)
            st.caption("总体加速比趋势")
        
        with col2:
            st.line_chart(df.set_index('datetime')[['cpu_gain', 'gpu_gain']], 
                         use_container_width=True)
            st.caption("CPU/GPU性能提升趋势")
        
        # 查询复杂度分布
        complexity_counts = df['query_complexity'].value_counts()
        st.bar_chart(complexity_counts, use_container_width=True)
        st.caption("查询复杂度分布")
    
    def render_optimization_recommendations(self, query: str):
        """渲染优化建议"""
        if not query.strip():
            return
        
        recommendations = self.manager.get_optimization_recommendations(query)
        
        if recommendations:
            st.subheader("💡 优化建议")
            for rec in recommendations:
                st.info(rec)

# 全局优化管理器实例
_optimization_manager = None

def get_intel_optimization_manager() -> IntelOptimizationManager:
    """获取全局Intel优化管理器实例"""
    global _optimization_manager
    if _optimization_manager is None:
        _optimization_manager = IntelOptimizationManager()
    return _optimization_manager

def optimize_query_performance(query: str, expected_result_size: int = 100) -> Optional[OptimizationResult]:
    """优化查询性能 - 供外部调用"""
    manager = get_intel_optimization_manager()
    return manager.optimize_for_query(query, expected_result_size)

def get_optimization_status() -> Dict[str, Any]:
    """获取优化状态 - 供外部调用"""
    manager = get_intel_optimization_manager()
    return manager.get_current_optimization_status()

def render_intel_optimization_ui():
    """渲染Intel优化UI - 供Streamlit应用调用"""
    manager = get_intel_optimization_manager()
    ui = IntelOptimizationUI(manager)
    
    # 渲染优化面板
    ui.render_optimization_panel()
    
    # 渲染性能历史
    ui.render_performance_history()
    
    return ui

# 测试函数
def test_integration():
    """测试集成功能"""
    print("🧪 测试Intel优化集成...")
    
    manager = IntelOptimizationManager()
    
    # 测试查询优化
    test_queries = [
        "SELECT * FROM customers WHERE age > 25",
        "SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name",
        "SELECT * FROM products WHERE category IN (SELECT category FROM top_categories WHERE rank <= 5) AND price BETWEEN 100 AND 500 ORDER BY rating DESC"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n📋 测试查询 {i+1}: {query[:50]}...")
        result = manager.optimize_for_query(query, expected_result_size=100 * (i+1))
        
        if result:
            print(f"✅ 优化完成，加速比: {result.overall_speedup:.2f}x")
        else:
            print("❌ 优化失败")
    
    # 获取状态
    status = manager.get_current_optimization_status()
    print(f"\n📊 当前状态: {status}")
    
    # 获取建议
    recommendations = manager.get_optimization_recommendations(test_queries[-1])
    print(f"\n💡 优化建议: {recommendations}")
    
    print("\n🎉 集成测试完成！")

if __name__ == "__main__":
    test_integration()