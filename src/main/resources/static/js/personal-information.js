const currentDomain = "http://localhost:8080"

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
    currentScore.innerText = "현재 점수 : " + responseScoreInfoValue.score + "점"
    maxScore.innerText = "최고 점수 : " + responseScoreInfoValue.maxScore + "점"
}

fetchMyInfo()

const data = {
    labels: ["악력", "팔굽혀피기", "윗몸일으키기", "50m 오래달리기", "왕복오래달리기"],
    datasets: [
        {
            label: "나",
            data: [1, 2, 3, 4, 5], // Sample values
            backgroundColor: "rgba(66, 135, 245, 0.5)",
            borderColor: "rgba(66, 135, 245, 1)",
            pointBackgroundColor: "rgba(66, 135, 245, 1)"
        }, {
            label: "상위 30%",
            data: [5, 4, 3, 2, 1], // Sample values
            backgroundColor: "rgba(33, 33, 33, 0.5)",
            borderColor: "rgba(33, 33, 33, 1)",
            pointBackgroundColor: "rgba(66, 135, 245, 1)"
        },
        // Add additional dataset here
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


//막대 그래프
// 데이터 준비
// 데이터 준비
var bar_labels = ['악력 (kg)', '팔굽혀피기 (회/분)', '윗몸일으키기 (회/분)', '50m 오래달리기 (초)', '왕복오래달리기 (회)'];
var myScore = [7, 9, 8, 6, 5];
var top30Score = [9, 10, 9, 9, 8];

// 차트 생성
var bar_ctx = document.getElementById('myChart').getContext('2d');
new Chart(bar_ctx, {
    type: 'bar',
    data: {
        labels: bar_labels,
        datasets: [{
            label: '나',
            data: myScore,
            backgroundColor: 'rgba(0, 0, 0, 0.2)', // 검은색
            borderColor: 'rgba(0, 0, 0, 1)',     // 검은색
            borderWidth: 1
        },{
            label: '상위 30%',
            data: top30Score,
            backgroundColor: 'rgba(192, 192, 192, 0.2)', // 밝은 회색
            borderColor: 'rgba(192, 192, 192, 1)',     // 밝은 회색
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
                position: "right" // Set the position of the legend to top
            }
        }
    }
});

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