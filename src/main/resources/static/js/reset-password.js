var validDomains = ['naver.com', 'google.com', 'daum.net'];
var emailInput = document.getElementById("email"); // 변경된 이메일 입력 요소 가져오기
var mergedEmailResult = document.getElementById("mergedEmail");

emailInput.addEventListener("input", updateMergedEmail);

function updateMergedEmail() {
  var email = emailInput.value;
  var errorMessage = '';
  var regex = /^[a-z0-9]+$/i;  // 대소문자 구분 없이 검사하는 정규식

  if (!email) {
    errorMessage = "이메일을 입력하세요.";
  } else if (!email.includes("@")) {
    errorMessage = "올바른 이메일 주소를 입력하세요.";
  } else {
    var parts = email.split("@");
    var emailFront = parts[0];
    var emailAfterAt = parts[1];

    if (!regex.test(emailFront)) {
      errorMessage = "올바른 이메일 앞 부분을 입력하세요.";
    } else if (!isValidDomain(emailAfterAt)) {
      errorMessage = "올바른 도메인 주소를 입력하세요.";
    } else if (!isValidDomainInList(emailAfterAt)) {
      errorMessage = "지원하지 않는 도메인 주소입니다.";
    }
  }

  if (errorMessage) {
    mergedEmailResult.textContent = errorMessage;
  } else {
    mergedEmailResult.textContent = emailFront + "@" + emailAfterAt;
  }
}
function isValidDomain(domain) {
return validDomains.includes(domain);
}

function isValidDomainInList(domain) {
return validDomains.includes(domain);
}

function getCookieValue(name) {
var cookieValue = "";
var cookies = document.cookie.split("; ");
for (var i = 0; i < cookies.length; i++) {
var cookie = cookies[i].split("=");
if (cookie[0] === name) {
  cookieValue = decodeURIComponent(cookie[1]);
  break;
}
}
return cookieValue;
}


var sendEmail = document.getElementById("sendEmail");
if (sendEmail) {
sendEmail.addEventListener("click", function (e) {
updateMergedEmail();

var mergedEmail = mergedEmailResult.textContent; // 업데이트된 값 읽어오기
console.log("!!!",mergedEmail);
console.log(mergedEmail);

document.cookie = "userEmail=" + encodeURIComponent(mergedEmail);
console.log("!!!쿠키", document.cookie);

window.location.href = "../html/reset-password-auth.html";
});
}

var sendEmail = document.getElementById("sendEmail");
if (sendEmail) {
  sendEmail.addEventListener("click", function (e) {
    window.location.href = "../html/reset-password-auth.html";
  });
}