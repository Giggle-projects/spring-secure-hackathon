const emailInput = document.getElementById("email"); // 변경된 이메일 입력 요소 가져오기
const mergedEmailResult = document.getElementById("mergedEmail");
const sendEmail = document.getElementById("sendEmail");

emailInput.addEventListener("input", updateMergedEmail);

// const currentDomain = window.location.origin
const currentDomain = "http://localhost:8080"

let mail_result = 0

// 정규식
const regex = /^[a-z0-9]+$/i;  // 대소문자 구분 없이 검사하는 정규식

function updateMergedEmail() {
    const email = emailInput.value;
    let errorMessage = '';

    if (!email) {
        errorMessage = "이메일을 입력하세요.";
    } else if (!email.includes("@")) {
        errorMessage = "올바른 이메일 주소를 입력하세요.";
    } else {
        const parts = email.split("@");
        var emailFront = parts[0];
        var emailAfterAt = parts[1];

        if (!regex.test(emailFront)) {
            errorMessage = "올바른 이메일 앞 부분을 입력하세요.";
        }
    }

    if (errorMessage) {
        mergedEmailResult.style.opacity = 1;
        mergedEmailResult.textContent = errorMessage;
        mergedEmailResult.style.color = "red"
        mail_result = 0
    } else {
        mergedEmailResult.style.opacity = 0;
        mail_result = 1
        mergedEmailResult.style.color = "black"; // Change text color to black
    }
}

function getCookieValue(name) {
    let cookieValue = "";
    const cookies = document.cookie.split("; ");
    for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].split("=");
        if (cookie[0] === name) {
            cookieValue = decodeURIComponent(cookie[1]);
            break;
        }
    }
    return cookieValue;
}

sendEmail.addEventListener("click", async function (e) {
    if (mail_result) {
        const userEmail = emailInput.value
        // 이메일 인증
        const response = await fetch(currentDomain + "/api/isExist/email?" + new URLSearchParams({
            email: userEmail
        }))

        if (!response.ok) {
            sAlert("이메일이 존재하지 않습니다.")
            throw new Error('not validate Email');
        }
        document.cookie = "userEmail=" + userEmail;

        const email = emailInput.value;
        let emailAuthResponse = await fetch(`http://localhost:8080/api/send/mail?email=${email}`)
        if (!emailAuthResponse.ok) {
            throw new Error('fetch error');
        }
        window.location.href = "./reset-password-auth.html";
    }
});

// sAlert('custom alert example!');
function sAlert(txt, title = 'ERROR',) {
    Swal.fire({
        title: title,
        text: txt,
        confirmButtonText: '닫기'
    });
}
