const data = {
    labels: ["악력", "팔굽혀피기", "윗몸일으키기", "50m 오래달리기", "왕복오래달리기"],
    datasets: [
        {
            label: "My Scores",
            data: [71, 92, 63, 80, 56], // Sample values
            backgroundColor: "rgba(66, 135, 245, 0.5)",
            borderColor: "rgba(66, 135, 245, 1)",
            pointBackgroundColor: "rgba(66, 135, 245, 1)"
        }, {
            label: "your Scores",
            data: [31, 92, 23, 80, 96], // Sample values
            backgroundColor: "rgba(33, 33, 33, 0.5)",
            borderColor: "rgba(33, 33, 33, 1)",
            pointBackgroundColor: "rgba(66, 135, 245, 1)"
        },
        // Add additional dataset here
    ]
};

// Create the radar chart
const ctx = document.getElementById("radarChart").getContext("2d");
const radarChart = new Chart(ctx, {
    type: "radar",
    data: data,
    options: {
        line: {
            borderWidth: 3
        },
        legend: {
            display: false, // 범례 숨기기
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
                max: 100, // You can adjust the maximum scale value
                stepSize: 10,
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
var avgScore = [5, 7, 6, 8, 7];
var top30Score = [9, 10, 9, 9, 8];

// 차트 생성
var bar_ctx = document.getElementById('myChart').getContext('2d');
var myChart = new Chart(bar_ctx, {
    type: 'bar',
    data: {
        labels: bar_labels,
        datasets: [{
            label: '나',
            data: myScore,
            backgroundColor: 'rgba(0, 0, 0, 0.2)', // 검은색
            borderColor: 'rgba(0, 0, 0, 1)',     // 검은색
            borderWidth: 1
        }, {
            label: '평균',
            data: avgScore,
            backgroundColor: 'rgba(128, 128, 128, 0.2)', // 회색
            borderColor: 'rgba(128, 128, 128, 1)',     // 회색
            borderWidth: 1
        }, {
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
            title: {
                display: true,
                text: '종목별 점수 분포'
            }
        }
    }
});


var settingContainer = document.getElementById("settingContainer");
if (settingContainer) {
    settingContainer.addEventListener("click", function (e) {
        window.location.href = "../html/setting-account.html";
    });
}

var menu01Container = document.getElementById("menu01Container");
if (menu01Container) {
    menu01Container.addEventListener("click", function (e) {
        window.location.href = "../html/dashboard.html";
    });
}

var menu02 = document.getElementById("menu02");
if (menu02) {
    menu02.addEventListener("click", function (e) {
        window.location.href = "../html/record.html";
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
        window.location.href = "../html/application-information1.html";
    });
}
