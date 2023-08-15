var codeInput = document.getElementById("code"); // 변경된 이메일 입력 요소 가져오기
let autocodeResult=0;
// reset-password-auth.html 페이지의 JavaScript 부분
document.addEventListener("DOMContentLoaded", function () {
    // 쿠키에서 userEmail 값 가져오기
    var userEmail = getCookie("userEmail");

    // 이메일 주소 표시
    var mergedEmailResult = document.getElementById("mergedEmail");
    if (userEmail) {
        mergedEmailResult.textContent = decodeURIComponent(userEmail);
    } else {
        mergedEmailResult.textContent = "저장된 이메일 주소가 없습니다.";
    }
    var codeInput = document.getElementById("code");
    var timerSpan = document.getElementById("timer");
    var timerInterval;
    var timerDuration = 180; // 3 minutes in seconds

    function updateTimerDisplay(remainingTime) {
        var minutes = Math.floor(remainingTime / 60);
        var seconds = remainingTime % 60;
        timerSpan.textContent = minutes + "m " + seconds + "s";
    }

    function startTimer() {
        var startTime = Date.now();

        timerInterval = setInterval(function () {
            var currentTime = Date.now();
            var elapsedTime = Math.floor((currentTime - startTime) / 1000);

            if (elapsedTime >= timerDuration) {
                clearInterval(timerInterval);
                timerSpan.textContent = "Timer expired!";
                // Redirect to reset-password.html after timer expiration
                window.location.href = "./reset-password.html";
            } else {
                var remainingTime = timerDuration - elapsedTime;
                updateTimerDisplay(remainingTime);
            }
        }, 1000);
    }

    // Call startTimer() when the page loads
    startTimer();


    var authNumberResult = document.getElementById("auth_number");

    // 예외 결과창
    codeInput.addEventListener("input", autoCodeCheck);
    function autoCodeCheck() {
        var code = codeInput.value;
        // 입력값이 숫자인지 확인
        if (!(/^\d+$/.test(code))) {
            authNumberResult.textContent = "숫자를 입력해주세요."; // 숫자가 아닐 경우 메시지 변경
        } else {
            // 숫자일 때는
            if (code === "1231") { // 실제 올바른 인증번호로 수정해야 함 1231 을 api로 받아오게
                authNumberResult.textContent = "인증번호가 맞습니다.";
                authNumberResult.style.color="black";
                authNumberResult.style.opacity = 0;
                autocodeResult=1;
            } else {
                authNumberResult.textContent = "인증번호가 다릅니다.";
                authNumberResult.style.color="red";
                authNumberResult.style.opacity = 1;
                autocodeResult=0;
            }

        }
    }

    // 인증하기 버튼 클릭 시
    var authButton = document.querySelector(".frame977");
    authButton.addEventListener("click", function (e) {
            console.log(autocodeResult)
            if (autocodeResult){
                //인증번호 맞음
                //a
                window.location.href = "./sign-in.html";
            }
            else{
                sAlert("인증번호가 다릅니다.")
            }

        }
    );
});

// 쿠키 가져오는 함수
function getCookie(name) {
    console.log("쿠키",name)
    var value = "; " + document.cookie;
    var parts = value.split("; " + name + "=");
    if (parts.length === 2) {
        return parts.pop().split(";").shift();
    }
}

// sAlert('custom alert example!');
function sAlert(txt, title = 'ERROR',) {
    Swal.fire({
        title: title,
        text: txt,
        confirmButtonText: '닫기'
    });
}