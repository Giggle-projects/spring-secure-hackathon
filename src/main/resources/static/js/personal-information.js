const currentDomain = window.location.origin
// const currentDomain = "http://localhost:8080"
var imgElement = document.querySelector('.frame-icon59');

async function fetchMyInfo() {
    const responseMemberInfo = await fetch(currentDomain + "/api/member/me");
    if (!responseMemberInfo.ok) {
        sAlert("Failed to fetch member")
        throw new Error('Error fetching.');
    }

    const responseScoreInfo = await fetch(currentDomain + "/api/score/me");
    if (!responseScoreInfo.ok) {
        sAlert("Failed to fetch expected")
        throw new Error('Error fetching.');
    }

    const responseMemberInfoValue = await responseMemberInfo.json();
    const responseScoreInfoValue = await responseScoreInfo.json();

    const name = document.getElementById('name');
    const applicationType = document.getElementById('applicationType');
    const currentScore = document.getElementById('currentScore');
    const maxScore = document.getElementById('maxScore');

    name.innerText = responseMemberInfoValue.name
    applicationType.innerText = responseMemberInfoValue.applicationTypeName


    // 아이콘 추가하기 코드
    // 불러오 이용자 이미지를 여기에 넣어주세요
    image_data=''
    if (!(image_data)){
        imgElement.src = "../files/586-frame.svg";
    }
    else{
        imgElement.src = "../files/586-frame.svg";
    }

//    if (responseMemberInfoValue.applicationTypeName === "경찰직공무원(남)" || responseMemberInfoValue.applicationTypeName === "경찰직공무원(여)") {
//        imgElement.src = "../files/type_icon4.png"; // 경찰직 이미지 경로로 변경
//    } else if (responseMemberInfoValue.applicationTypeName === "소방직공무원(남)" || responseMemberInfoValue.applicationTypeName === "소방직공무원(여)") {
//        imgElement.src = "../files/type_icon2.jpeg"; // 소방직 이미지 경로로 변경
//    } else if (responseMemberInfoValue.applicationTypeName === "교정직공무원(남)" || responseMemberInfoValue.applicationTypeName === "교정직공무원(여)") {
//        imgElement.src = "../files/type_icon3.png";// 교정직 이미지 경로로 변경
//    }
    currentScore.innerText = "현재 점수 : " + responseScoreInfoValue.score + "점"
    maxScore.innerText = "최고 점수 : " + responseScoreInfoValue.maxScore + "점"
}

fetchMyInfo()

async function showRadarChart() {
    let responseScore = await fetch(currentDomain + "/api/score/detail/me");
    if (!responseScore.ok) {
        throw new Error('Error fetching.');
    }
    const scores = await responseScore.json()
    const data = {
        labels:  scores.map(function(element){
            return element.evaluationItemName;
        }),
        datasets: [
            {
                label: "나",
                data: scores.map(function(element){
                    return element.evaluationScore;
                }),
                backgroundColor: "rgba(66, 135, 245, 0.5)",
                borderColor: "rgba(66, 135, 245, 1)",
                pointBackgroundColor: "rgba(66, 135, 245, 1)"
            }
        ]
    };

    const ctx = document.getElementById("radarChart").getContext("2d");
    new Chart(ctx, {
        type: "radar",
        data: data,
        options: {
            responsive: false,
            line: {
                borderWidth: 10
            },
            scale: {
                r: {
                    pointLabels: {
                        font: {
                            size: 100
                        }
                    }
                },
                ticks: {
                    beginAtZero: true,
                    max: 10, // You can adjust the maximum scale value
                    stepSize: 1,
                }
            },
            plugins: {
                legend: {
                    position: "right" // Set the position of the legend to top
                }
            },
            tooltips: {
                enabled: false, // Disable tooltips
            },
        }
    });
}

showRadarChart()

async function showBarChart() {
    let responseScoreMe = await fetch(currentDomain + "/api/score/detail/me");
    if (!responseScoreMe.ok) {
        throw new Error('Error fetching.');
    }

    let responseScoreTop = await fetch(currentDomain + "/api/score/detail/top?percent=30");
    if (!responseScoreTop.ok) {
        throw new Error('Error fetching.');
    }
    const scoresMe = await responseScoreMe.json()
    const scoresTop = await responseScoreTop.json()

    var bar_labels = (scoresMe).map(function(element){
        return element.evaluationItemName;
    });
    var myScore = (scoresMe).map(function(element){
        return element.evaluationScore;
    });
    var top30Score = (scoresTop).map(function(element){
        return element.evaluationScore;
    });

    var bar_ctx = document.getElementById('myChart').getContext('2d');
    new Chart(bar_ctx, {
        type: 'bar',
        data: {
            labels: bar_labels,
            datasets: [{
                label: '나',
                data: myScore,
                backgroundColor: "rgba(66, 135, 245, 0.5)",
                borderColor: "rgba(66, 135, 245, 1)",
                borderWidth: 1
            },{
                label: '상위 30%',
                data: top30Score,
                backgroundColor: "rgba(66, 135, 245, 0.2)",
                borderColor: "rgba(66, 135, 245, 1)",
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '점'
                    },
                    ticks: {
                        stepSize: 1, // 간격을 1로 설정하여 1점 단위로 표시
                        precision: 0 // 소수점 없이 정수로 표시
                    }
                }
            },
            plugins: {
                legend: {
                    position: "right"
                }
            }
        }
    });
}

showBarChart()

var settingContainer = document.getElementById("settingContainer");
if (settingContainer) {
    settingContainer.addEventListener("click", function (e) {
        window.location.href = "setting-account.html";
    });
}

var menu01Container = document.getElementById("menu01Container");
if (menu01Container) {
    menu01Container.addEventListener("click", function (e) {
        window.location.href = "dashboard.html";
    });
}

var menu02 = document.getElementById("menu02");
if (menu02) {
    menu02.addEventListener("click", function (e) {
        window.location.href = "record.html";
    });
}

var menu03Container = document.getElementById("menu03Container");
if (menu03Container) {
    menu03Container.addEventListener("click", function (e) {
        window.location.href = "personal-information.html";
    });
}

var menu04Container = document.getElementById("menu04Container");
if (menu04Container) {
    menu04Container.addEventListener("click", function (e) {
        window.location.href = "application-information1.html";
    });
}

// logout component
const logoutContainer = document.getElementById("logoutContainer");
logoutContainer.addEventListener("click", async function (e) {
    // 로그아웃에 필요한 처리를 여기에 추가하세요.

    const cookieKey = "accessToken"
    const removeTokenResponse = await fetch(currentDomain + "/api/remove/token?" + new URLSearchParams({
        cookieKey: cookieKey
    }))

    if (!removeTokenResponse.ok) {
        window.location.href = "../html/error-500.html";
        throw new Error('token remove fail');
    }

    // 로그아웃 후 리다이렉트 등의 동작을 수행할 수 있습니다.
    window.location.href = '../html/sign-in.html'; // 로그인 페이지로 리다이렉트 예시
});
