document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetView = item.getAttribute('data-view');
            
            navItems.forEach(i => i.classList.remove('active'));
            views.forEach(v => v.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(`${targetView}-view`).classList.add('active');

            if (targetView === 'memory') loadMemory();
            if (targetView === 'stats') loadStats();
        });
    });

    // Chat functionality
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    const addMessage = (text, role) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        let contentHTML = text;
        if (role === 'assistant') {
            try {
                const parsed = marked.parse(text);
                contentHTML = DOMPurify.sanitize(parsed);
            } catch (e) {
                // simple fallback
                contentHTML = DOMPurify.sanitize(text);
            }
        } else {
            // For user messages, escape HTML but keep newlines
            contentHTML = text.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>");
        }

        msgDiv.innerHTML = `
            <div class="avatar">${role === 'user' ? 'U' : 'A'}</div>
            <div class="bubble markdown-content">${contentHTML}</div>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const sendMessage = async () => {
        const query = userInput.value.trim();
        if (!query) return;

        addMessage(query, 'user');
        userInput.value = '';
        userInput.rows = 1;

        // Show typing indicator or just wait
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant loading';
        loadingDiv.innerHTML = '<div class="avatar">A</div><div class="bubble">...</div>';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            const data = await response.json();
            
            chatMessages.removeChild(loadingDiv);
            addMessage(data.response, 'assistant');
        } catch (err) {
            chatMessages.removeChild(loadingDiv);
            addMessage('Error connecting to the AI server.', 'assistant');
        }
    };

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
        if (userInput.scrollHeight > 150) {
            userInput.style.overflowY = 'scroll';
            userInput.style.height = '150px';
        } else {
            userInput.style.overflowY = 'hidden';
        }
    });

    // Memory loader
    const loadMemory = async () => {
        const memoryContent = document.getElementById('memory-content');
        try {
            const response = await fetch('/api/memory');
            const data = await response.json();
            memoryContent.textContent = data.content || 'Memory is empty.';
        } catch (err) {
            memoryContent.textContent = 'Failed to load memory.';
        }
    };

    // Stats loader
    const loadStats = async () => {
        try {
            const response = await fetch('/api/stats');
            if (!response.ok) throw new Error("Server error");
            const data = await response.json();
            
            document.getElementById('cpu-stat').textContent = `${data.cpu}%`;
            document.querySelector('#cpu-stat + .stat-bar .bar-fill').style.width = `${data.cpu}%`;
            
            document.getElementById('ram-stat').textContent = `${data.ram}%`;
            document.querySelector('#ram-stat + .stat-bar .bar-fill').style.width = `${data.ram}%`;
            
            document.getElementById('disk-stat').textContent = `${data.disk}%`;
            document.querySelector('#disk-stat + .stat-bar .bar-fill').style.width = `${data.disk}%`;

            document.querySelector('.status-indicator').classList.add('online');
            document.querySelector('.status-indicator').classList.remove('offline');
            document.querySelector('.system-status span').textContent = 'System Online';
        } catch (err) {
            console.error('Failed to load stats');
            document.querySelector('.status-indicator').classList.remove('online');
            document.querySelector('.status-indicator').classList.add('offline');
            document.querySelector('.system-status span').textContent = 'System Offline';
        }
    };

    // Initial load
    setInterval(loadStats, 3000);
});
