const emailInput = document.getElementById("email"); // 변경된 이메일 입력 요소 가져오기
const passwordInput = document.getElementById("password1");
const passwordInput2 = document.getElementById("password2");
const passwordResult = document.getElementById("password_result");
const passwordResult2 = document.getElementById("password_result2");
const nameInput = document.getElementById("name");
const nameCheckResult = document.getElementById("name_check");
const mergedEmailResult = document.getElementById("mergedEmail");
const check_box = document.querySelector(".check-box");

// dropdown
const dropdown = document.querySelector(".dropdown");
const dropdownText = dropdown.querySelector(".text704");
const dropdownContent = dropdown.querySelector(".dropdown-content");

let mail_result = 0

/* 쿠키 가져오는 함수
function getCookie(name) {
    const value = "; " + document.cookie;
    const parts = value.split("; " + name + "=");
    if (parts.length === 2) return parts.pop().split(";").shift();
}
var cookie_userName = getCookie("userName");
var cookie_userEmail = getCookie("userEmail");
var cookie_userPassword = getCookie("passWord");
var cookie_userPassword2 = getCookie("passWord2");
var cookie_userTestTpye = getCookie("testType");
var cookie_userAgree = getCookie("agree");
*/

document.addEventListener("DOMContentLoaded", function () {
    let name_result = 0
    let password1_result = 0
    let password2_result = 0

    // 이름 처리
    nameInput.addEventListener("input", function () {
        const name = nameInput.value;

        if (/[!@#$%^&*()_+{}\[\]:;<>,.?~\\/`'" ]/.test(name)) {
            nameCheckResult.textContent = "특수 문자 또는 공백은 입력할 수 없습니다.";
        } else if (!/^[가-힣]+$/.test(name)) {
            nameCheckResult.textContent = "한글만 입력 가능합니다.";
        } else if (name.length < 2 || name.length > 4) {
            nameCheckResult.textContent = "이름은 2글자 이상, 4글자 이하로 입력해주세요.";
        } else {
            nameCheckResult.textContent = "올바른 이름입니다.";
            name_result = 1
        }
    });

    // 이메일 입력 처리
    emailInput.addEventListener("input", function (e) {
        const email = emailInput.value;
        const parts = email.split("@");
        const emailFront = parts[0];
        const emailAfterAt = parts[1];
        let errorMessage = '';
        const regex = /^[a-z0-9]+$/i;  // 대소문자 구분 없이 검사하는 정규식

        if (!email) {
            errorMessage = "이메일을 입력하세요.";
        } else if (!email.includes("@")) {
            errorMessage = "올바른 이메일 주소를 입력하세요.";
        } else {
            if (!regex.test(emailFront)) {
                errorMessage = "올바른 이메일 앞 부분을 입력하세요.";
            }
        }
        if (errorMessage) {
            mergedEmailResult.textContent = errorMessage;
        } else {
            mergedEmailResult.textContent = emailFront + "@" + emailAfterAt;
            mail_result = 1
        }
    });

    // 이메일 인증하기 버튼 이벤트 리스너
    const email_Auth = document.getElementById("email_auth");
    if (email_Auth) {
        email_Auth.addEventListener("click", async function (e) {
            console.log(emailInput)
            const emailInputValue = emailInput.value.trim();
            if (!emailInputValue) {
                alert("이메일을 입력하세요.");
                return;
            }

            //이메일 형식
            const emailRegex = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/;

            if (!emailRegex.test(emailInputValue)) {
                alert("올바른 이메일 주소 형식이 아닙니다.");
            } else {
                document.cookie = "userEmail=" + emailInputValue;

                // 이메일 인증 버튼 클릭 시 인증 코드를 메일로 보내는 로직
                const email = emailInput.value;
                fetch(`http://localhost:8080/api/send/mail?email=${email}`)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! Status: ${response.status}`);
                        }
                        return response.json();
                    })
                    .catch(error => {
                        console.error('Failed to fetch data');
                    });

                window.open("./sign-up-email-auth.html", "SignUpEmailAuth", "width=800,height=600");
            }
        });
    }

    // password 처리
    passwordInput.addEventListener("change", checkPasswordValidity);

    function checkPasswordValidity() {
        const password = passwordInput.value;
        var errorMessage = '';

        const lengthRegex = /^.{9,16}$/;
        const alphanumericRegex = /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$/;

        if (!lengthRegex.test(password)) {
            errorMessage = "비밀번호는 9자 이상 16자 이하로 입력하세요.";
        } else if (!alphanumericRegex.test(password)) {
            errorMessage = "비밀번호는 영어, 숫자, 특수문자를 모두 포함해야 합니다.";
        }

        /*
        else if (!nameCheck) {
            errorMessage = "비밀번호에 사용자 이름이 포함되어 있습니다.";
        } else if (!pastPasswordsCheck) {
            errorMessage = "비밀번호가 과거에 사용된 적이 있습니다.";
        }
        */

        passwordResult.textContent = errorMessage || "비밀번호가 유효합니다.";
    }

    // repassword 처리
    passwordInput2.addEventListener("change", checkPasswordMatch);

    function checkPasswordMatch() {
        const password1 = passwordInput.value;
        const password2 = passwordInput2.value;

        if (password1 === password2) {
            passwordResult2.textContent = "비밀번호가 일치합니다.";
            password1_result = 1
            password2_result = 1
        } else {
            passwordResult2.textContent = "비밀번호가 일치하지 않습니다.";
        }
    }

    // application type 드롭 다운 데이터 처리
    dropdownText.addEventListener("click", function () {
        dropdown.classList.toggle("active");
    });

    dropdownContent.addEventListener("click", function (event) {
        if (event.target.tagName === "A") {
            event.preventDefault();
            dropdownText.textContent = event.target.textContent;
            dropdownText.id = event.target.id;
            dropdown.classList.remove("active");
        }
    });

    window.addEventListener("click", function (event) {
        if (!dropdown.contains(event.target)) {
            dropdown.classList.remove("active");
        }
    });

    //회원가입 버튼 클릭
    const signUpBtn = document.querySelector(".sign-up-btn");
    signUpBtn.addEventListener("click", async function () {
        var nameInputValue = ""
        if (name_result) {
            nameInputValue = nameInput.value;
        }

        var emailInputValue = "";
        mail_result = 1
        if (mail_result) {
            emailInputValue = emailInput.value; // Get the value from the email input
        }

        var password1InputValue = ""
        if (password_result) {
            password1InputValue = passwordInput.value; // Get the value from the first password input
        }

        var password2InputValue = ""
        if (password2_result) {
            password2InputValue = passwordInput2.value; // Get the value from the second password input
        }

        if (password1InputValue !== password2InputValue) {
            alert("비밀번호와 재비밀번호의 값은 같아야합니다.");
            return;
        }

        const checkBoxSelected = check_box.classList.contains("selected_box");
        const applicationTypeKey = dropdownText.id;

        if (checkBoxSelected) {
            // 프론트 로그
            console.log("이름 입력 값:", nameInputValue);
            console.log("이메일 입력 값:", emailInputValue);
            console.log("비밀번호 입력 값:", password1InputValue);
            console.log("개인정보 이용 동의 여부:", checkBoxSelected);
            console.log("희망 지망 직렬 선택:", applicationTypeKey);

            // 회원가입 - 서버에 요청
            const userData = {
                name: nameInputValue,
                email: emailInputValue,
                password: password1InputValue,
                applicationType: applicationTypeKey
            };
            try {
                const response = await fetch(currentDomain + "/api/member/signUp", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(userData),
                });
                if (!response.ok) {
                    const data = await response.text();
                    throw new Error(data || "회원가입 실패");
                }

                // 이메일 정보 리셋 후 dashboard로 이동
                eraseCookie("userEmail");
                window.location.href = "../html/dashboard.html";
                alert("회원가입을 성공했습니다.");
            } catch (error) {
                alert(error.message);
            }
        } else {
            alert("개인정보 이용 동의 여부를 체크해주세요");
        }
    });
});

function eraseCookie(name) {
    document.cookie = name + '=; Max-Age=0'
}

check_box.addEventListener("click", function () {
    check_box.classList.toggle("selected_box");
});


function togglePasswordVisibility(inputElementId) {
    var passwordInput = document.getElementById(inputElementId);
    var eyeIcon = document.querySelector(`#${inputElementId} + .password-toggle`);

    if (passwordInput.type === "password") {
        passwordInput.type = "text";

        // 변경된 path 또는 추가적인 path 등을 포함한 아이콘 내용으로 변경
        eyeIcon.innerHTML = `<path d="M20.2501 21.0001C20.1516 21.0003 20.054 20.9809 19.963 20.9432C19.872 20.9054 19.7894 20.8501 19.7199 20.7802L3.21994 4.28025C3.08522 4.13845 3.01123 3.94964 3.01373 3.75407C3.01624 3.5585 3.09504 3.37164 3.23334 3.23334C3.37164 3.09504 3.5585 3.01624 3.75407 3.01373C3.94964 3.01123 4.13845 3.08522 4.28025 3.21994L20.7803 19.7199C20.8851 19.8248 20.9564 19.9584 20.9854 20.1039C21.0143 20.2493 20.9994 20.4001 20.9427 20.5371C20.8859 20.6741 20.7899 20.7912 20.6666 20.8736C20.5433 20.956 20.3984 21 20.2501 21.0001V21.0001Z" fill="#21272A"/>
<path d="M11.984 18.0002C10.0392 18.0002 8.16371 17.4245 6.40965 16.2892C4.81261 15.258 3.37496 13.7809 2.25183 12.0236V12.0198C3.18652 10.6806 4.21027 9.54812 5.30996 8.635C5.31991 8.62668 5.32802 8.61639 5.33379 8.60478C5.33956 8.59317 5.34285 8.58048 5.34347 8.56753C5.34409 8.55458 5.34201 8.54164 5.33737 8.52953C5.33273 8.51742 5.32563 8.50641 5.31652 8.49719L4.38277 7.56484C4.36618 7.54811 4.34389 7.53824 4.32035 7.5372C4.29682 7.53616 4.27375 7.54402 4.25574 7.55922C3.08761 8.54359 2.0034 9.75109 1.01761 11.1648C0.848014 11.4083 0.754608 11.6966 0.749266 11.9932C0.743924 12.2899 0.826888 12.5814 0.987615 12.8308C2.22558 14.7681 3.81886 16.3998 5.59449 17.5487C7.59371 18.8439 9.74527 19.5002 11.984 19.5002C13.1924 19.4964 14.3923 19.2973 15.5371 18.9105C15.5522 18.9053 15.5658 18.8965 15.5765 18.8846C15.5872 18.8728 15.5947 18.8585 15.5983 18.8429C15.602 18.8274 15.6015 18.8112 15.5972 18.7959C15.5928 18.7805 15.5845 18.7666 15.5732 18.7553L14.5617 17.7437C14.5384 17.721 14.5096 17.7048 14.4781 17.6965C14.4466 17.6883 14.4136 17.6884 14.3821 17.6969C13.5987 17.8986 12.793 18.0005 11.984 18.0002V18.0002Z" fill="#21272A"/>
<path d="M23.0081 11.1844C21.7678 9.26625 20.1586 7.63688 18.3548 6.47203C16.3594 5.18203 14.1562 4.5 11.9841 4.5C10.7884 4.50212 9.60172 4.70543 8.47358 5.10141C8.45855 5.10664 8.4451 5.11561 8.4345 5.12747C8.42389 5.13934 8.41648 5.15371 8.41296 5.16923C8.40945 5.18475 8.40994 5.20091 8.41439 5.21619C8.41885 5.23147 8.42712 5.24536 8.43843 5.25656L9.44858 6.26672C9.4721 6.28984 9.5013 6.30634 9.53323 6.31457C9.56516 6.3228 9.5987 6.32246 9.63046 6.31359C10.3978 6.10602 11.1891 6.00058 11.9841 6C13.8914 6 15.7612 6.58266 17.5411 7.73438C19.1681 8.78437 20.6226 10.26 21.7486 12C21.7494 12.0011 21.7499 12.0024 21.7499 12.0037C21.7499 12.0051 21.7494 12.0064 21.7486 12.0075C20.9313 13.2942 19.9171 14.4446 18.743 15.4167C18.7329 15.425 18.7247 15.4353 18.7188 15.4469C18.713 15.4586 18.7096 15.4713 18.7089 15.4843C18.7083 15.4973 18.7103 15.5103 18.715 15.5225C18.7196 15.5347 18.7268 15.5457 18.7359 15.555L19.6687 16.4873C19.6852 16.504 19.7074 16.5139 19.7308 16.515C19.7543 16.5161 19.7773 16.5084 19.7953 16.4934C21.0487 15.4381 22.1337 14.1975 23.0128 12.8147C23.1682 12.571 23.2503 12.2878 23.2495 11.9988C23.2487 11.7098 23.1649 11.4272 23.0081 11.1844V11.1844Z" fill="#21272A"/>
<path d="M12 7.5C11.6629 7.49982 11.3269 7.53756 10.9983 7.6125C10.9817 7.61595 10.9663 7.62383 10.9538 7.63531C10.9414 7.64679 10.9322 7.66144 10.9274 7.6777C10.9226 7.69395 10.9223 7.71121 10.9265 7.72763C10.9308 7.74405 10.9393 7.75901 10.9514 7.77094L16.2291 13.0472C16.241 13.0592 16.2559 13.0678 16.2724 13.0721C16.2888 13.0763 16.306 13.076 16.3223 13.0712C16.3386 13.0663 16.3532 13.0572 16.3647 13.0447C16.3762 13.0323 16.384 13.0169 16.3875 13.0003C16.5378 12.3413 16.5376 11.6568 16.3871 10.9979C16.2365 10.3389 15.9395 9.72228 15.518 9.19385C15.0964 8.66542 14.5613 8.23874 13.9523 7.9455C13.3432 7.65226 12.6759 7.49999 12 7.5V7.5Z" fill="#21272A"/>
<path d="M7.77091 10.9526C7.75899 10.9406 7.74402 10.932 7.7276 10.9278C7.71118 10.9235 7.69392 10.9238 7.67767 10.9287C7.66141 10.9335 7.64676 10.9426 7.63528 10.9551C7.6238 10.9675 7.61592 10.9829 7.61247 10.9995C7.44248 11.7422 7.46382 12.5158 7.67451 13.248C7.88519 13.9801 8.27832 14.6468 8.81704 15.1855C9.35577 15.7243 10.0224 16.1174 10.7546 16.3281C11.4868 16.5388 12.2604 16.5601 13.0031 16.3901C13.0197 16.3867 13.035 16.3788 13.0475 16.3673C13.06 16.3558 13.0691 16.3412 13.0739 16.3249C13.0787 16.3087 13.0791 16.2914 13.0748 16.275C13.0706 16.2586 13.062 16.2436 13.05 16.2317L7.77091 10.9526Z" fill="#21272A"/>
`;

    } else {
        passwordInput.type = "password";
        // 변경된 path 또는 추가적인 path 등을 포함한 아이콘 내용으로 변경
        eyeIcon.innerHTML = `<path fill-rule="evenodd" clip-rule="evenodd" d="M1.01759 11.1645C4.0676 6.79251 8.02225 4.5 11.9841 4.5C16.6371 4.5 20.6453 7.53129 23.0101 11.1836L23.0112 11.1853C23.1677 11.4285 23.2509 11.7115 23.2509 12.0007C23.2509 12.2893 23.168 12.5719 23.012 12.8147C20.6493 16.5141 16.6671 19.5 11.9841 19.5C7.2513 19.5 3.3458 16.5203 0.988892 12.8317C0.828743 12.583 0.745679 12.2925 0.750173 11.9968C0.754679 11.7002 0.846976 11.4117 1.0154 11.1676L1.01759 11.1645ZM2.25001 12.0196L2.25208 12.0228C4.42901 15.4304 7.91813 18 11.9841 18C16.0045 18 19.5689 15.4195 21.7484 12.0065L21.7498 12.0043C21.7505 12.0032 21.7509 12.002 21.7509 12.0007C21.7509 12.0002 21.7508 11.9996 21.7507 11.9991C21.7505 11.9985 21.7503 11.9979 21.7499 11.9973C19.5621 8.61915 15.9686 6 11.9841 6C8.63995 6 5.10025 7.93555 2.25001 12.0196Z" fill="#21272A"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9ZM7.5 12C7.5 9.51472 9.51472 7.5 12 7.5C14.4853 7.5 16.5 9.51472 16.5 12C16.5 14.4853 14.4853 16.5 12 16.5C9.51472 16.5 7.5 14.4853 7.5 12Z" fill="#21272A"/>
      `;
    }
}

const signInText = document.getElementById("signInText");
if (signInText) {
    signInText.addEventListener("click", function (e) {
        window.location.href = "../html/sign-in.html";
    });
}

// const currentDomain = window.location.origin
const currentDomain = "http://localhost:8080"

async function fetchApplicationTypeName() {
    const applicationTypeDropDownContent = document.getElementById("applicationTypeDropDownContent");
    const applicationTypeDropDownKey = document.getElementById("applicationTypeDropDownKey");

    const response = await fetch(currentDomain + "/api/applicationType")
    if (!response.ok) {
        alert("Invalid score input")
        throw new Error('Error fetching.');
    }

    await fetch(currentDomain + "/api/applicationType")
        .then(response => response.json())
        .then(responseJson => {
            responseJson.forEach(function (option) {
                const aElement = document.createElement("a");
                aElement.textContent = option.applicationTypeStandardName;
                aElement.id = option.applicationType;
                applicationTypeDropDownContent.appendChild(aElement);
            });
        })
        .catch(() => {
            throw new Error('Failed to fetch data');
        });
}

fetchApplicationTypeName()