const settingContainer = document.getElementById("settingContainer");
settingContainer.addEventListener("click", function (e) {
    window.location.href = "../html/setting-account.html";
});

const menu02Container = document.getElementById("menu02Container");
menu02Container.addEventListener("click", function (e) {
    window.location.href = "../html/record.html";
});

const menu03Container = document.getElementById("menu03Container");
menu03Container.addEventListener("click", function (e) {
    window.location.href = "../html/personal-information.html";
});

const menu04Container = document.getElementById("menu04Container")
menu04Container.addEventListener("click", function (e) {
    window.location.href = "../html/application-information1.html";
});


// const currentDomain = window.location.origin
const currentDomain = "http://localhost:8080"

// 사용자의 applicationTypeName값
let applicationTypeName;

async function fetchTotalApplication() {
    const totalApplicationContainer = document.getElementById("total-application")
    const totalRecordsContainer = document.getElementById("total-records")

    let response1 = await fetch(currentDomain + "/api/applicant/count");
    if (!response1.ok) {
        throw new Error('Error fetching.');
    }
    totalApplicationContainer.innerText = (await response1.json()).count

    let response2 = await fetch(currentDomain + "/api/score/count");
    if (!response2.ok) {
        throw new Error('Error fetching.');
    }
    totalRecordsContainer.innerText = (await response2.json()).count
}

async function fetchMyInfo() {
    const memberName = document.getElementById("memberName")

    const currentScoreContainer = document.getElementById("current-score")
    const expectedScoreContainer = document.getElementById("expected-score")

    let responseMember = await fetch(currentDomain + "/api/member/me");
    if (!responseMember.ok) {
        throw new Error('Error fetching.');
    } else {
        const body = await responseMember.json();
        memberName.innerText = (body).name + "님 안녕하세요."
        applicationTypeName = (body).applicationTypeName
    }

    let responseScore = await fetch(currentDomain + "/api/score/me");
    if (!responseScore.ok) {
        throw new Error('Error fetching.');
    } else {
        currentScoreContainer.innerText = (await responseScore.json()).score
    }
}

async function fetchRank() {
    let response = await fetch(currentDomain + "/api/score/rank?rankCnt=7");
    if (!response.ok) {
        throw new Error('Error fetching.');
    }
    const rankTable = document.getElementById("rankTable");
    for (const item of (await response.json())) {
        const rankContainer = document.createElement("div");
        rankContainer.className = "rank-01";

        const frameDiv = document.createElement("div");
        frameDiv.className = "frame1475";

        const tierDiv = document.createElement("div");
        tierDiv.className = "tier-01";
        tierDiv.textContent = item.rank + "등";

        const nameDiv = document.createElement("div");
        nameDiv.className = "name-01";
        nameDiv.id = "rank-name-1";
        nameDiv.textContent = item.memberName;

        const scoreFrameDiv = document.createElement("div");
        scoreFrameDiv.className = "frame1476";

        const scoreDiv = document.createElement("div");
        scoreDiv.className = "score-01";
        scoreDiv.id = "rank-score-1";
        scoreDiv.textContent = item.score + "점";

        const labelDiv = document.createElement("div");
        labelDiv.className = "label-016";
        labelDiv.id = "rank-application-type-1";
        labelDiv.textContent = item.applicationType;

        frameDiv.appendChild(tierDiv);
        scoreFrameDiv.appendChild(scoreDiv);
        scoreFrameDiv.appendChild(labelDiv);
        rankContainer.appendChild(frameDiv);
        rankContainer.appendChild(nameDiv);
        rankContainer.appendChild(scoreFrameDiv);

        rankTable.appendChild(rankContainer);
    }
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

fetchTotalApplication()
fetchMyInfo()
fetchRank()
