const currentDomain = window.location.origin
// const currentDomain = "http://localhost:8080"

const codeInput = document.getElementById("code");
const timerSpan = document.getElementById("timer");
const verificationError = document.getElementById("verificationError");
const authNumberResult = document.getElementById("auth_number");
const authButton = document.querySelector(".frame977");
// 쿠키에서 userEmail 값 가져오기
const userEmail = getCookie("userEmail");
// 이메일 주소 표시
const mergedEmailResult = document.getElementById("mergedEmail");

// 쿠키 가져오는 함수
function getCookie(name) {
    var value = "; " + document.cookie;
    var parts = value.split("; " + name + "=");
    if (parts.length === 2) {
        return parts.pop().split(";").shift();
    }
}

document.addEventListener("DOMContentLoaded", function () {
    let timerInterval;
    const timerDuration = 180; // 3 minutes in seconds

    if (userEmail) {
        mergedEmailResult.textContent = decodeURIComponent(userEmail);
    } else {
        mergedEmailResult.textContent = "저장된 이메일 주소가 없습니다.";
    }

    function updateTimerDisplay(remainingTime) {
        const minutes = Math.floor(remainingTime / 60);
        const seconds = remainingTime % 60;
        timerSpan.textContent = minutes + "m " + seconds + "s";
    }

    function startTimer() {
        var startTime = Date.now();
        timerInterval = setInterval(function () {
            const currentTime = Date.now();
            const elapsedTime = Math.floor((currentTime - startTime) / 1000);

            if (elapsedTime >= timerDuration) {
                clearInterval(timerInterval);
                timerSpan.textContent = "Timer expired!";
                window.location.href = "../html/sign-up.html";
            } else {
                var remainingTime = timerDuration - elapsedTime;
                updateTimerDisplay(remainingTime);
            }
        }, 1000);
    }

    // Call startTimer() when the page loads
    startTimer();

    // 인증하기 버튼 클릭 시
    codeInput.addEventListener("input", function (e) {
        const codeInputData= codeInput.value;
        if (!(codeInputData)){
            authNumberResult.textContent="인증번호를 입력하세요."
            authNumberResult.style.opacity = 1;
            authNumberResult.style.color='red'
            return
        }
    });



    authButton.addEventListener("click", async function () {
        const authCode = codeInput.value;

            // 인증 코드 검증 로직
            const validAuthApiData = {
                email: userEmail,
                authCode: authCode
            };
            try {
                const response = await fetch(currentDomain + "/api/signUp/valid/authCode", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(validAuthApiData),
                });
                if (!response.ok) {
                    throw new Error("인증에 실패했습니다. 다시 시도해주시기 바랍니다.");
                }
//                    authNumberResult.textContent = "인증이 완료되었습니다.";
//                    alert(authNumberResult.textContent);
                window.close();
            } catch (error) {
                alert(error.message);
            }
    });
});

