const currentDomain = window.location.origin
// const currentDomain = "http://localhost:8080"

const tableBody = document.getElementById('table-body');
const pagination = document.getElementById('pagination');

const userNameInput = document.getElementById('userName');
const pageSizeSelection = document.getElementById('pageSize');
const minScoreSelection = document.getElementById('minScore');
const maxScoreSelection = document.getElementById('maxScore');
const sortBySelection = document.getElementById('sortBy');

let pageSize = 100
let pageCount = 5
let startPage = 1;
let currentPage = 1;
let endPage = startPage + pageCount -1;
let startUserRead;
let lastUserRead;
let users = []

function blockButtonEventHandler(productId) {
    const orderUrl = currentDomain + "/api/admin/users"
    const data = {
        productId: productId,
        userId: 1,
        quantity: 1
    }
    axios({
        method: "post",
        url: currentDomain + orderUrl,
        params: data
    }).then(() => {
        window.location.reload();
    }).catch(function () {
        if (confirm("Error")) {
            window.location.reload();
        }
    })
}

userNameInput.addEventListener("keydown", function(event) {
    if(event.keyCode === 13) {
        fetchAndDisplayUsers().then(() => {});
    }
});

userNameInput.addEventListener("change", function(event) {
    fetchAndDisplayUsers().then(() => {});
});

minScoreSelection.addEventListener("change", (event) => {
    if(maxScoreSelection.value !== '' && minScoreSelection.value > maxScoreSelection.value) {
        alert("invalid score search range")
        minScoreSelection.value = maxScoreSelection.value
        return
    }
    fetchAndDisplayUsers().then(() => {});
});

maxScoreSelection.addEventListener("change", (event) => {
    if(minScoreSelection.value !== '' && minScoreSelection.value > maxScoreSelection.value) {
        alert("invalid score search range")
        maxScoreSelection.value = minScoreSelection.value
        return
    }
    fetchAndDisplayUsers().then(() => {});
});

sortBySelection.addEventListener("change", (event) => {
  fetchAndDisplayUsers().then(() => {});
});

pageSizeSelection.addEventListener("change", (event) => {
  fetchAndDisplayUsers().then(() => {});
});

// Function to fetch and display products based on the current page and search query
async function fetchAndDisplayUsers() {
    pageSize = pageSizeSelection.options[pageSizeSelection.selectedIndex].value;
    startPage = 1;
    currentPage = 1;
    endPage = startPage + pageCount -1;
    startUserRead = undefined;
    lastUserRead = undefined
    users = await fetchUser();
    startUserRead = users[0];
    lastUserRead = users[users.length-1]
    endPage = startPage + users.length / pageSize -1
    displayUsers()
    updatePagination()
}

async function fetchUser() {
    try {
        const containsName = userNameInput.value.trim();
        const minScore = minScoreSelection.value.trim();
        const maxScore = maxScoreSelection.value.trim();
        const sortBy = sortBySelection.options[sortBySelection.selectedIndex].value;
        const pageSize = pageSizeSelection.options[pageSizeSelection.selectedIndex].value;
        const fetchSize = pageSize * pageCount;

        const fetchUrl = currentDomain + "/api/admin/users/cursor/?"
        let response = await fetch(fetchUrl + new URLSearchParams({
            containsName: containsName,
            minScore: minScore,
            maxScore: maxScore,
            pageSize: fetchSize,
            userSortType : sortBy,
            sortBy: sortBy
        }));
        if (!response.ok) {
            throw new Error('Error fetching scores.');
        }
        return await response.json();
    } catch (error) {
        console.log(error)
        alert('Error fetching scores.')
        return [];
    }
}

async function fetchNextUsers() {
    try {
        const containsName = userNameInput.value.trim();
        const minScore = minScoreSelection.value.trim();
        const maxScore = maxScoreSelection.value.trim();
        const sortBy = sortBySelection.options[sortBySelection.selectedIndex].value;
        const fetchSize = pageSize * pageCount;

        const fetchUrl = currentDomain +  "/api/admin/users/cursor/?"
        let response = await fetch(fetchUrl + new URLSearchParams({
            containsName: containsName,
            minScore: minScore,
            maxScore: maxScore,
            size: 40,
            cursorUserId: lastProductRead.id,
            cursorUserName : lastProductRead.name,
            cursorUserScore : lastProductRead.score,
            UserortType : sortBy,
            sortBy: sortBy
        }));
        if (!response.ok) {
            throw new Error('Error fetching users.');
        }
        return await response.json();
    } catch (error) {
        console.log(error)
        alert('Error fetching users.')
        return [];
    }
}

async function fetchPrevUsers() {
    try {
        const containsName = userNameInput.value.trim();
        const minScore = minScoreSelection.value.trim();
        const maxScore = maxScoreSelection.value.trim();
        const sortBy = sortBySelection.options[sortBySelection.selectedIndex].value;
        const fetchSize = pageSize * pageCount;

        const fetchUrl = currentDomain + "/api/admin/users/cursor/prev?"
        let response = await fetch(fetchUrl + new URLSearchParams({
            containsName: containsName,
            minScore: minScore,
            maxScore: maxScore,
            pageSize: fetchSize,
            cursorUserId: startUserRead.id,
            cursorUserName: startUserRead.name,
            cursorUserScore: cursorUserScore.price,
            cursorUserDate: cursorUserDate.quantity,
            userSortType : sortBy,
            sortBy: sortBy
        }));
        if (!response.ok) {
            throw new Error('Error fetching users.');
        }
        return await response.json();
    } catch (error) {
        console.log(error)
        alert('Error fetching users.')
        return [];
    }
}

// Function to display products in the table
async function displayUsers() {
    tableBody.innerHTML = '';
    for(let index = 0; index < pageSize; index++) {
        const productIndex = ((currentPage -1) % pageCount ) * pageSize + index
        const user = users[productIndex]
        const row = document.createElement('tr');
        row.id = `user-${user.id}`
        row.innerHTML = `
          <td>${user.id}</td>
          <td>${user.name}</td>
          <td>${user.email}</td>
          <td>${user.score}</td>
          <td>${user.lastLogin}</td>
          <td><button onclick="blockButtonEventHandler(${user.id})">Block</button></td>`;
        tableBody.appendChild(row);
    }
}

async function updatePagination() {
    pagination.innerHTML = '';
    if(startPage >= pageCount) {
        const leftArrow = document.createElement('a');
        leftArrow.href = '#';
        leftArrow.textContent = '<<';
        leftArrow.className = "pagination-link";
        leftArrow.addEventListener('click', async () => {
            currentPage = startPage - 1;
            endPage = currentPage
            startPage = startPage - pageCount
            users = await fetchPrevUsers();
            startUserRead = users[0];
            lastUserRead = users[users.length-1]
            await displayUsers();
            await updatePagination()
        });
        pagination.appendChild(leftArrow);
    }

    for (let pageNumber = startPage; pageNumber <= endPage; pageNumber++) {
        const link = document.createElement('a');
        link.href = '#';
        link.textContent = pageNumber;
        link.className = "pagination-link";
        if (pageNumber === currentPage) {
            link.classList.add('pagination-active');
        }
        link.addEventListener('click', async () => {
            currentPage = pageNumber;
            await displayUsers();
            await updatePagination()
        });
        pagination.appendChild(link);
    }

    if(users.length >= pageSize * pageCount) {
        const rightArrow = document.createElement('a');
        rightArrow.href = '#';
        rightArrow.textContent = '>>';
        rightArrow.className = "pagination-link";
        rightArrow.addEventListener('click', async () => {
            currentPage = endPage + 1;
            startPage = currentPage
            users = await fetchNextUsers()
            startUserRead = users[0];
            lastUserRead = users[users.length-1]
            endPage = startPage + users.length / pageSize -1
            await displayUsers()
            await updatePagination()
        });
        pagination.appendChild(rightArrow);
    }
}

// Initial fetch and display of products
fetchAndDisplayUsers();
