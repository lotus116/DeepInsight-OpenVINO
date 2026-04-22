/*
 * Intel® DeepInsight - 交互增强脚本 v2.0
 * 移动端侧边栏控制、键盘快捷键、动画效果
 */
(function() {
    'use strict';
    
    let hintTimeout;
    let animationObserver;
    
    // ========================================
    // 🎨 动态样式注入
    // ========================================
    function injectDynamicStyles() {
        if (document.getElementById('deepinsight-dynamic-styles')) return;
        
        const styleSheet = document.createElement('style');
        styleSheet.id = 'deepinsight-dynamic-styles';
        styleSheet.textContent = `
            /* 键盘提示样式 */
            .keyboard-hint {
                position: fixed;
                bottom: 100px;
                left: 50%;
                transform: translateX(-50%) translateY(20px);
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                color: white;
                padding: 12px 24px;
                border-radius: 12px;
                font-size: 14px;
                font-weight: 500;
                z-index: 99999;
                opacity: 0;
                pointer-events: none;
                transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(10px);
            }
            
            .keyboard-hint.show {
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }
            
            /* 按钮点击波纹效果 */
            .ripple-effect {
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.4);
                transform: scale(0);
                animation: rippleAnimation 0.6s ease-out;
                pointer-events: none;
            }
            
            @keyframes rippleAnimation {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
            
            /* 成功反馈动画 */
            .success-feedback {
                animation: successPulse 0.4s ease !important;
            }
            
            @keyframes successPulse {
                0% { transform: scale(1); }
                50% { transform: scale(0.95); box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.3); }
                100% { transform: scale(1); }
            }
            
            /* 元素入场动画 */
            .animate-fade-in {
                animation: fadeInUp 0.5s ease-out forwards;
            }
            
            .animate-slide-in {
                animation: slideInFromRight 0.4s ease-out forwards;
            }
            
            .animate-scale-in {
                animation: scaleIn 0.3s ease-out forwards;
            }
            
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes slideInFromRight {
                from {
                    opacity: 0;
                    transform: translateX(30px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            @keyframes scaleIn {
                from {
                    opacity: 0;
                    transform: scale(0.9);
                }
                to {
                    opacity: 1;
                    transform: scale(1);
                }
            }
            
            /* 骨架屏加载效果 */
            .skeleton-loading {
                background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                background-size: 200% 100%;
                animation: skeletonPulse 1.5s infinite;
            }
            
            @keyframes skeletonPulse {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
            
            /* 悬浮卡片效果 */
            .hover-lift {
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .hover-lift:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
            }
            
            /* 渐变边框动画 */
            .gradient-border {
                position: relative;
            }
            
            .gradient-border::before {
                content: '';
                position: absolute;
                top: -2px;
                left: -2px;
                right: -2px;
                bottom: -2px;
                background: linear-gradient(45deg, #0068B5, #3b82f6, #10b981, #0068B5);
                background-size: 300% 300%;
                border-radius: inherit;
                z-index: -1;
                animation: gradientBorderMove 4s ease infinite;
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .gradient-border:hover::before {
                opacity: 1;
            }
            
            @keyframes gradientBorderMove {
                0%, 100% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
            }
            
            /* 打字机光标效果 */
            .typing-cursor::after {
                content: '|';
                animation: cursorBlink 1s infinite;
                color: #0068B5;
            }
            
            @keyframes cursorBlink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0; }
            }
            
            /* 平滑滚动 */
            html {
                scroll-behavior: smooth;
            }
            
            /* 消息气泡尾巴 */
            .message-tail::after {
                content: '';
                position: absolute;
                bottom: 20px;
                width: 0;
                height: 0;
                border: 10px solid transparent;
            }
        `;
        document.head.appendChild(styleSheet);
    }
    
    // ========================================
    // ✨ 入场动画观察器
    // ========================================
    function setupAnimationObserver() {
        if (animationObserver) animationObserver.disconnect();
        
        animationObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType !== 1) return;
                    
                    // 聊天消息入场动画
                    if (node.matches && node.matches('[data-testid="stChatMessage"]')) {
                        node.classList.add('animate-fade-in');
                    }
                    
                    // 查找子元素中的聊天消息
                    const chatMessages = node.querySelectorAll ? 
                        node.querySelectorAll('[data-testid="stChatMessage"]') : [];
                    chatMessages.forEach((msg, index) => {
                        setTimeout(() => {
                            msg.classList.add('animate-fade-in');
                        }, index * 100);
                    });
                    
                    // Metric 指标入场动画
                    const metrics = node.querySelectorAll ? 
                        node.querySelectorAll('[data-testid="stMetric"]') : [];
                    metrics.forEach((metric, index) => {
                        setTimeout(() => {
                            metric.classList.add('animate-slide-in');
                        }, index * 80);
                    });
                    
                    // 按钮入场动画
                    const buttons = node.querySelectorAll ? 
                        node.querySelectorAll('section.main button') : [];
                    buttons.forEach((btn, index) => {
                        setTimeout(() => {
                            btn.classList.add('animate-scale-in');
                        }, index * 50);
                    });
                });
            });
        });
        
        // 观察整个文档
        animationObserver.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    // ========================================
    // 🌊 按钮波纹效果
    // ========================================
    function addRippleEffect(e) {
        const button = e.currentTarget;
        const rect = button.getBoundingClientRect();
        const ripple = document.createElement('span');
        
        const size = Math.max(rect.width, rect.height);
        const x = e.clientX - rect.left - size / 2;
        const y = e.clientY - rect.top - size / 2;
        
        ripple.className = 'ripple-effect';
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        
        button.style.position = 'relative';
        button.style.overflow = 'hidden';
        button.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    }
    
    function setupRippleEffects() {
        document.querySelectorAll('button').forEach(button => {
            if (!button.dataset.rippleAttached) {
                button.addEventListener('click', addRippleEffect);
                button.dataset.rippleAttached = 'true';
            }
        });
    }
    
    // ========================================
    // 📱 移动端侧边栏控制
    // ========================================
    function setupMobileSidebar() {
        const isMobile = window.innerWidth <= 768;
        
        if (!isMobile) return;
        
        setTimeout(function() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            const collapseButton = document.querySelector('[data-testid="collapsedControl"]');
            
            if (!sidebar) return;
            
            let overlay = document.getElementById('mobile-sidebar-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'mobile-sidebar-overlay';
                overlay.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    background: rgba(0, 0, 0, 0.6);
                    backdrop-filter: blur(4px);
                    z-index: 999998;
                    display: none;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                `;
                document.body.appendChild(overlay);
            }
            
            function isSidebarOpen() {
                const sidebarContent = sidebar.querySelector('[data-testid="stSidebarContent"]');
                return sidebarContent && window.getComputedStyle(sidebarContent).display !== 'none';
            }
            
            function closeSidebar() {
                const closeBtn = sidebar.querySelector('button[kind="header"]');
                if (closeBtn) closeBtn.click();
                overlay.style.display = 'none';
                overlay.style.opacity = '0';
                document.body.style.overflow = '';
            }
            
            function openSidebar() {
                overlay.style.display = 'block';
                setTimeout(() => overlay.style.opacity = '1', 10);
                document.body.style.overflow = 'hidden';
            }
            
            overlay.onclick = closeSidebar;
            
            if (collapseButton) {
                collapseButton.addEventListener('click', () => setTimeout(openSidebar, 100));
            }
            
            const sidebarCloseBtn = sidebar.querySelector('button[kind="header"]');
            if (sidebarCloseBtn) {
                sidebarCloseBtn.addEventListener('click', () => setTimeout(closeSidebar, 100));
            }
            
            sidebar.addEventListener('click', function(e) {
                const target = e.target;
                if (target.tagName === 'A' || 
                    target.closest('[role="option"]') ||
                    target.closest('button[kind="secondary"]')) {
                    setTimeout(closeSidebar, 300);
                }
            });
            
            if (isSidebarOpen()) openSidebar();
            else closeSidebar();
            
            const sidebarObserver = new MutationObserver(function() {
                if (isSidebarOpen()) openSidebar();
                else {
                    overlay.style.display = 'none';
                    overlay.style.opacity = '0';
                }
            });
            
            const sidebarContent = sidebar.querySelector('[data-testid="stSidebarContent"]');
            if (sidebarContent) {
                sidebarObserver.observe(sidebarContent, {
                    attributes: true,
                    attributeFilter: ['style']
                });
            }
        }, 500);
    }
    
    // ========================================
    // ⌨️ 键盘快捷键
    // ========================================
    function showKeyboardHint(text, type = 'info') {
        let hint = document.querySelector('.keyboard-hint');
        if (!hint) {
            hint = document.createElement('div');
            hint.className = 'keyboard-hint';
            document.body.appendChild(hint);
        }
        
        // 根据类型设置不同的图标
        const icons = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'send': '📤',
            'new': '✨',
            'focus': '🎯',
            'clear': '🗑️'
        };
        
        hint.textContent = (icons[type] || '') + ' ' + text;
        hint.classList.add('show');
        
        clearTimeout(hintTimeout);
        hintTimeout = setTimeout(() => hint.classList.remove('show'), 2500);
    }
    
    function setupKeyboardShortcuts() {
        document.addEventListener('keydown', function(e) {
            // Ctrl/Cmd + Enter: 发送消息
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const chatInput = document.querySelector('.stChatInput input, .stChatInput textarea');
                if (chatInput && chatInput.value.trim()) {
                    const submitBtn = document.querySelector('.stChatInput button');
                    if (submitBtn) {
                        submitBtn.click();
                        showKeyboardHint('消息已发送', 'send');
                    }
                }
            }
            
            // Ctrl/Cmd + N: 新建会话
            if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
                e.preventDefault();
                const newSessionBtn = document.querySelector('button[title*="新建"]');
                if (newSessionBtn) {
                    newSessionBtn.click();
                    showKeyboardHint('新建会话', 'new');
                }
            }
            
            // Ctrl/Cmd + /: 聚焦输入框
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                const chatInput = document.querySelector('.stChatInput input, .stChatInput textarea');
                if (chatInput) {
                    chatInput.focus();
                    showKeyboardHint('聚焦输入框', 'focus');
                }
            }
            
            // Esc: 清除输入
            if (e.key === 'Escape') {
                const chatInput = document.querySelector('.stChatInput input, .stChatInput textarea');
                if (chatInput && chatInput.value) {
                    chatInput.value = '';
                    chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                    showKeyboardHint('输入已清除', 'clear');
                }
            }
            
            // F1: 显示快捷键帮助
            if (e.key === 'F1') {
                e.preventDefault();
                showKeyboardHint('Ctrl+Enter 发送 | Ctrl+N 新建 | Ctrl+/ 聚焦 | Esc 清除', 'info');
            }
        });
    }
    
    // ========================================
    // 🔄 定期更新增强
    // ========================================
    function periodicEnhancements() {
        // 每秒检查新按钮并添加波纹效果
        setInterval(setupRippleEffects, 1000);
        
        // 为推荐问题卡片添加悬浮效果
        document.querySelectorAll('section.main div[data-testid="column"] button').forEach(btn => {
            if (!btn.classList.contains('hover-lift')) {
                btn.classList.add('hover-lift');
            }
        });
    }
    
    // ========================================
    // 🚀 初始化
    // ========================================
    function initialize() {
        injectDynamicStyles();
        setupAnimationObserver();
        setupMobileSidebar();
        setupKeyboardShortcuts();
        setupRippleEffects();
        periodicEnhancements();
        
        console.log('✅ Intel DeepInsight UI enhancements loaded');
    }
    
    // 等待DOM加载完成
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
    
    // 监听窗口大小变化
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(setupMobileSidebar, 250);
    });
    
    // Streamlit重新渲染后重新初始化
    window.addEventListener('load', function() {
        setTimeout(initialize, 1000);
    });
})();
