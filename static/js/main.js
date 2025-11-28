// 主JavaScript文件

// Flash消息自动消失
document.addEventListener('DOMContentLoaded', function() {
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(function(message) {
        setTimeout(function() {
            message.style.opacity = '0';
            message.style.transition = 'opacity 0.3s';
            setTimeout(function() {
                message.remove();
            }, 300);
        }, 3000);
    });
});

// 表单验证
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;
    
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;
    
    inputs.forEach(function(input) {
        if (!input.value.trim()) {
            isValid = false;
            input.classList.add('error');
        } else {
            input.classList.remove('error');
        }
    });
    
    return isValid;
}

// 密码确认验证
function validatePasswordMatch() {
    const password = document.getElementById('password');
    const confirmPassword = document.getElementById('confirm_password');
    
    if (password && confirmPassword) {
        confirmPassword.addEventListener('input', function() {
            if (password.value !== confirmPassword.value) {
                confirmPassword.setCustomValidity('密码不匹配');
            } else {
                confirmPassword.setCustomValidity('');
            }
        });
    }
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', function() {
    validatePasswordMatch();
});


