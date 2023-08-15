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
        console.log("a" + element.evaluationScore)
        return element.evaluationScore;
    });
    var top30Score = (scoresTop).map(function(element){
        console.log("b" + element.evaluationScore)
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
                backgroundColor: 'rgba(0, 0, 0, 0.2)',
                borderColor: 'rgba(0, 0, 0, 1)',
                borderWidth: 1
            },{
                label: '상위 30%',
                data: top30Score,
                backgroundColor: 'rgba(192, 192, 192, 0.2)',
                borderColor: 'rgba(192, 192, 192, 1)',
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