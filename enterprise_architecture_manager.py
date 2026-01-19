#!/usr/bin/env python3
"""
企业级架构管理器
实现微服务架构、依赖注入、设计模式等企业级特性
目标：提升架构设计成熟度，体现技术方案的专业性
"""

import asyncio
import threading
import time
import logging
import inspect
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Type, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import weakref
from concurrent.futures import ThreadPoolExecutor, Future
import json

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')
ServiceType = TypeVar('ServiceType')

class ServiceLifecycle(Enum):
    """服务生命周期"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"

class ServiceStatus(Enum):
    """服务状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ServiceDescriptor:
    """服务描述符"""
    service_type: Type
    implementation: Type
    lifecycle: ServiceLifecycle
    dependencies: List[Type] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    health_check: Optional[Callable] = None
    startup_priority: int = 100  # 数字越小优先级越高

@dataclass
class ServiceInstance:
    """服务实例"""
    descriptor: ServiceDescriptor
    instance: Any
    status: ServiceStatus
    created_at: float
    last_health_check: Optional[float] = None
    health_status: bool = True
    error_count: int = 0

class IService(ABC):
    """服务接口"""
    
    @abstractmethod
    async def start(self) -> None:
        """启动服务"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """停止服务"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass

class IRepository(ABC, Generic[T]):
    """仓储接口"""
    
    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        pass
    
    @abstractmethod
    async def find_all(self) -> List[T]:
        pass

class IEventBus(ABC):
    """事件总线接口"""
    
    @abstractmethod
    async def publish(self, event: Any) -> None:
        pass
    
    @abstractmethod
    def subscribe(self, event_type: Type, handler: Callable) -> None:
        pass

class DependencyInjectionContainer:
    """依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, ServiceInstance] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = threading.RLock()
        self._scope_counter = 0
        
        logger.info("✅ 依赖注入容器初始化完成")
    
    def register(self, 
                service_type: Type[ServiceType], 
                implementation: Type[ServiceType],
                lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
                dependencies: Optional[List[Type]] = None,
                configuration: Optional[Dict[str, Any]] = None,
                health_check: Optional[Callable] = None,
                startup_priority: int = 100) -> 'DependencyInjectionContainer':
        """注册服务"""
        
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation=implementation,
                lifecycle=lifecycle,
                dependencies=dependencies or [],
                configuration=configuration or {},
                health_check=health_check,
                startup_priority=startup_priority
            )
            
            self._services[service_type] = descriptor
            
            logger.info(f"✅ 服务注册成功: {service_type.__name__} -> {implementation.__name__}")
            
        return self
    
    def register_singleton(self, service_type: Type[ServiceType], implementation: Type[ServiceType]) -> 'DependencyInjectionContainer':
        """注册单例服务"""
        return self.register(service_type, implementation, ServiceLifecycle.SINGLETON)
    
    def register_transient(self, service_type: Type[ServiceType], implementation: Type[ServiceType]) -> 'DependencyInjectionContainer':
        """注册瞬态服务"""
        return self.register(service_type, implementation, ServiceLifecycle.TRANSIENT)
    
    def register_scoped(self, service_type: Type[ServiceType], implementation: Type[ServiceType]) -> 'DependencyInjectionContainer':
        """注册作用域服务"""
        return self.register(service_type, implementation, ServiceLifecycle.SCOPED)
    
    def resolve(self, service_type: Type[ServiceType], scope_id: Optional[str] = None) -> ServiceType:
        """解析服务"""
        
        with self._lock:
            if service_type not in self._services:
                raise ValueError(f"服务未注册: {service_type.__name__}")
            
            descriptor = self._services[service_type]
            
            # 根据生命周期处理
            if descriptor.lifecycle == ServiceLifecycle.SINGLETON:
                return self._get_singleton_instance(descriptor)
            elif descriptor.lifecycle == ServiceLifecycle.TRANSIENT:
                return self._create_transient_instance(descriptor)
            elif descriptor.lifecycle == ServiceLifecycle.SCOPED:
                return self._get_scoped_instance(descriptor, scope_id or "default")
            else:
                raise ValueError(f"不支持的生命周期: {descriptor.lifecycle}")
    
    def _get_singleton_instance(self, descriptor: ServiceDescriptor) -> Any:
        """获取单例实例"""
        if descriptor.service_type in self._instances:
            return self._instances[descriptor.service_type].instance
        
        instance = self._create_instance(descriptor)
        
        service_instance = ServiceInstance(
            descriptor=descriptor,
            instance=instance,
            status=ServiceStatus.RUNNING,
            created_at=time.time()
        )
        
        self._instances[descriptor.service_type] = service_instance
        
        return instance
    
    def _create_transient_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建瞬态实例"""
        return self._create_instance(descriptor)
    
    def _get_scoped_instance(self, descriptor: ServiceDescriptor, scope_id: str) -> Any:
        """获取作用域实例"""
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}
        
        scope_instances = self._scoped_instances[scope_id]
        
        if descriptor.service_type not in scope_instances:
            instance = self._create_instance(descriptor)
            scope_instances[descriptor.service_type] = instance
        
        return scope_instances[descriptor.service_type]
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        try:
            # 解析依赖
            dependencies = {}
            for dep_type in descriptor.dependencies:
                dependencies[dep_type.__name__.lower()] = self.resolve(dep_type)
            
            # 获取构造函数参数
            sig = inspect.signature(descriptor.implementation.__init__)
            constructor_args = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                # 尝试从依赖中获取
                if param_name in dependencies:
                    constructor_args[param_name] = dependencies[param_name]
                # 尝试从配置中获取
                elif param_name in descriptor.configuration:
                    constructor_args[param_name] = descriptor.configuration[param_name]
                # 检查是否有默认值
                elif param.default != inspect.Parameter.empty:
                    continue
                else:
                    logger.warning(f"无法解析参数: {param_name} for {descriptor.implementation.__name__}")
            
            # 创建实例
            instance = descriptor.implementation(**constructor_args)
            
            logger.debug(f"✅ 创建服务实例: {descriptor.implementation.__name__}")
            
            return instance
            
        except Exception as e:
            logger.error(f"❌ 创建服务实例失败: {descriptor.implementation.__name__}: {e}")
            raise
    
    def create_scope(self) -> str:
        """创建新的作用域"""
        with self._lock:
            self._scope_counter += 1
            scope_id = f"scope_{self._scope_counter}_{int(time.time())}"
            self._scoped_instances[scope_id] = {}
            return scope_id
    
    def dispose_scope(self, scope_id: str) -> None:
        """销毁作用域"""
        with self._lock:
            if scope_id in self._scoped_instances:
                # 清理作用域内的实例
                scope_instances = self._scoped_instances[scope_id]
                for instance in scope_instances.values():
                    if hasattr(instance, 'dispose'):
                        try:
                            instance.dispose()
                        except Exception as e:
                            logger.warning(f"销毁实例时出错: {e}")
                
                del self._scoped_instances[scope_id]
                logger.debug(f"✅ 作用域已销毁: {scope_id}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        with self._lock:
            return {
                "registered_services": len(self._services),
                "singleton_instances": len(self._instances),
                "scoped_instances": sum(len(scope) for scope in self._scoped_instances.values()),
                "active_scopes": len(self._scoped_instances),
                "services": [
                    {
                        "type": desc.service_type.__name__,
                        "implementation": desc.implementation.__name__,
                        "lifecycle": desc.lifecycle.value,
                        "dependencies": [dep.__name__ for dep in desc.dependencies]
                    }
                    for desc in self._services.values()
                ]
            }

class EventBus(IEventBus):
    """事件总线实现"""
    
    def __init__(self):
        self._handlers: Dict[Type, List[Callable]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="EventBus")
        
        logger.info("✅ 事件总线初始化完成")
    
    async def publish(self, event: Any) -> None:
        """发布事件"""
        event_type = type(event)
        
        with self._lock:
            handlers = self._handlers.get(event_type, [])
        
        if not handlers:
            logger.debug(f"没有找到事件处理器: {event_type.__name__}")
            return
        
        # 异步执行所有处理器
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                # 在线程池中执行同步处理器
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(self._executor, handler, event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"✅ 事件已发布: {event_type.__name__} -> {len(handlers)}个处理器")
    
    def subscribe(self, event_type: Type, handler: Callable) -> None:
        """订阅事件"""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            
            self._handlers[event_type].append(handler)
            
        logger.debug(f"✅ 事件订阅成功: {event_type.__name__}")
    
    def unsubscribe(self, event_type: Type, handler: Callable) -> None:
        """取消订阅"""
        with self._lock:
            if event_type in self._handlers:
                try:
                    self._handlers[event_type].remove(handler)
                    logger.debug(f"✅ 取消事件订阅: {event_type.__name__}")
                except ValueError:
                    logger.warning(f"处理器未找到: {event_type.__name__}")
    
    def get_subscription_count(self, event_type: Type) -> int:
        """获取订阅数量"""
        with self._lock:
            return len(self._handlers.get(event_type, []))
    
    def shutdown(self):
        """关闭事件总线"""
        self._executor.shutdown(wait=True)
        logger.info("事件总线已关闭")

class HealthCheckManager:
    """健康检查管理器"""
    
    def __init__(self, container: DependencyInjectionContainer):
        self.container = container
        self._health_checks: Dict[str, Callable] = {}
        self._last_check_results: Dict[str, bool] = {}
        self._check_interval = 30  # 30秒
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        
        logger.info("✅ 健康检查管理器初始化完成")
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """注册健康检查"""
        self._health_checks[name] = check_func
        logger.debug(f"✅ 健康检查注册: {name}")
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._check_thread.start()
        
        logger.info("✅ 健康检查监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        
        logger.info("健康检查监控已停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                self._perform_health_checks()
                time.sleep(self._check_interval)
            except Exception as e:
                logger.error(f"健康检查出错: {e}")
                time.sleep(5)  # 出错时短暂等待
    
    def _perform_health_checks(self) -> None:
        """执行健康检查"""
        for name, check_func in self._health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    # 异步健康检查
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(check_func())
                    loop.close()
                else:
                    # 同步健康检查
                    result = check_func()
                
                self._last_check_results[name] = bool(result)
                
                if not result:
                    logger.warning(f"⚠️ 健康检查失败: {name}")
                
            except Exception as e:
                logger.error(f"❌ 健康检查异常: {name}: {e}")
                self._last_check_results[name] = False
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            "overall_healthy": all(self._last_check_results.values()) if self._last_check_results else True,
            "checks": dict(self._last_check_results),
            "monitoring_active": self._running,
            "check_interval": self._check_interval
        }

class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self):
        self._configurations: Dict[str, Any] = {}
        self._watchers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
        # 加载默认配置
        self._load_default_configuration()
        
        logger.info("✅ 配置管理器初始化完成")
    
    def _load_default_configuration(self) -> None:
        """加载默认配置"""
        default_config = {
            "system": {
                "max_threads": 4,
                "timeout_seconds": 30,
                "retry_count": 3,
                "cache_size": 1000
            },
            "database": {
                "connection_timeout": 10,
                "query_timeout": 30,
                "max_connections": 10
            },
            "performance": {
                "enable_caching": True,
                "cache_ttl": 300,
                "enable_compression": True,
                "batch_size": 100
            }
        }
        
        self._configurations.update(default_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        with self._lock:
            keys = key.split('.')
            value = self._configurations
            
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        with self._lock:
            keys = key.split('.')
            config = self._configurations
            
            # 导航到父级
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # 设置值
            old_value = config.get(keys[-1])
            config[keys[-1]] = value
            
            # 通知观察者
            if old_value != value:
                self._notify_watchers(key, old_value, value)
    
    def watch(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """监听配置变化"""
        with self._lock:
            if key not in self._watchers:
                self._watchers[key] = []
            
            self._watchers[key].append(callback)
    
    def _notify_watchers(self, key: str, old_value: Any, new_value: Any) -> None:
        """通知配置观察者"""
        watchers = self._watchers.get(key, [])
        
        for watcher in watchers:
            try:
                watcher(key, old_value, new_value)
            except Exception as e:
                logger.error(f"配置观察者通知失败: {e}")
    
    def get_all_configurations(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            return dict(self._configurations)

class EnterpriseArchitectureManager:
    """企业级架构管理器"""
    
    def __init__(self):
        self.container = DependencyInjectionContainer()
        self.event_bus = EventBus()
        self.health_manager = HealthCheckManager(self.container)
        self.config_manager = ConfigurationManager()
        
        # 注册核心服务
        self._register_core_services()
        
        # 启动健康检查
        self.health_manager.start_monitoring()
        
        logger.info("✅ 企业级架构管理器初始化完成")
    
    def _register_core_services(self) -> None:
        """注册核心服务"""
        # 注册事件总线
        self.container.register_singleton(IEventBus, type(self.event_bus))
        self.container._instances[IEventBus] = ServiceInstance(
            descriptor=ServiceDescriptor(IEventBus, type(self.event_bus), ServiceLifecycle.SINGLETON),
            instance=self.event_bus,
            status=ServiceStatus.RUNNING,
            created_at=time.time()
        )
        
        # 注册配置管理器
        self.container.register_singleton(ConfigurationManager, type(self.config_manager))
        self.container._instances[ConfigurationManager] = ServiceInstance(
            descriptor=ServiceDescriptor(ConfigurationManager, type(self.config_manager), ServiceLifecycle.SINGLETON),
            instance=self.config_manager,
            status=ServiceStatus.RUNNING,
            created_at=time.time()
        )
    
    @asynccontextmanager
    async def create_scope(self):
        """创建服务作用域上下文管理器"""
        scope_id = self.container.create_scope()
        try:
            yield scope_id
        finally:
            self.container.dispose_scope(scope_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "architecture": {
                "pattern": "Microservices with DI Container",
                "services": self.container.get_service_status(),
                "event_bus": {
                    "active": True,
                    "total_subscriptions": sum(
                        self.event_bus.get_subscription_count(event_type) 
                        for event_type in self.event_bus._handlers.keys()
                    )
                }
            },
            "health": self.health_manager.get_health_status(),
            "configuration": {
                "total_configs": len(self.config_manager.get_all_configurations()),
                "watchers": sum(len(watchers) for watchers in self.config_manager._watchers.values())
            },
            "timestamp": time.time()
        }
    
    def shutdown(self) -> None:
        """关闭架构管理器"""
        logger.info("正在关闭企业级架构管理器...")
        
        # 停止健康检查
        self.health_manager.stop_monitoring()
        
        # 关闭事件总线
        self.event_bus.shutdown()
        
        # 清理所有作用域
        for scope_id in list(self.container._scoped_instances.keys()):
            self.container.dispose_scope(scope_id)
        
        logger.info("✅ 企业级架构管理器已关闭")

# 全局实例
enterprise_architecture = EnterpriseArchitectureManager()

def get_enterprise_architecture() -> EnterpriseArchitectureManager:
    """获取企业级架构管理器实例"""
    return enterprise_architecture

def get_service(service_type: Type[T]) -> T:
    """获取服务实例（便捷方法）"""
    return enterprise_architecture.container.resolve(service_type)

def publish_event(event: Any) -> None:
    """发布事件（便捷方法）"""
    asyncio.create_task(enterprise_architecture.event_bus.publish(event))

def get_config(key: str, default: Any = None) -> Any:
    """获取配置（便捷方法）"""
    return enterprise_architecture.config_manager.get(key, default)

# 装饰器
def service(service_type: Type, lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON):
    """服务注册装饰器"""
    def decorator(cls):
        enterprise_architecture.container.register(service_type, cls, lifecycle)
        return cls
    return decorator

def event_handler(event_type: Type):
    """事件处理器装饰器"""
    def decorator(func):
        enterprise_architecture.event_bus.subscribe(event_type, func)
        return func
    return decorator

# 测试函数
def test_enterprise_architecture():
    """测试企业级架构功能"""
    print("🧪 测试企业级架构...")
    
    # 定义测试服务
    class ITestService(ABC):
        @abstractmethod
        def process(self, data: str) -> str:
            pass
    
    @service(ITestService)
    class TestService(ITestService):
        def __init__(self, config_manager: ConfigurationManager):
            self.config_manager = config_manager
        
        def process(self, data: str) -> str:
            timeout = self.config_manager.get("system.timeout_seconds", 30)
            return f"Processed: {data} (timeout: {timeout}s)"
    
    # 定义测试事件
    @dataclass
    class TestEvent:
        message: str
        timestamp: float = field(default_factory=time.time)
    
    @event_handler(TestEvent)
    def handle_test_event(event: TestEvent):
        print(f"📨 收到事件: {event.message} at {event.timestamp}")
    
    # 测试服务解析
    test_service = get_service(ITestService)
    result = test_service.process("test data")
    print(f"✅ 服务调用结果: {result}")
    
    # 测试事件发布
    test_event = TestEvent("Hello from enterprise architecture!")
    publish_event(test_event)
    
    # 测试配置管理
    print(f"✅ 配置值: system.max_threads = {get_config('system.max_threads')}")
    
    # 获取系统状态
    status = enterprise_architecture.get_system_status()
    print(f"✅ 系统状态: {status['architecture']['services']['registered_services']}个已注册服务")
    
    print("🎉 企业级架构测试完成！")

if __name__ == "__main__":
    test_enterprise_architecture()