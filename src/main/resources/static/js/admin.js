const currentDomain = window.location.origin

const tableBody = document.getElementById('table-body');
const pagination = document.getElementById('pagination');

const userNameInput = document.getElementById('userName');
const pageSizeSelection = document.getElementById('pageSize');
const minScoreSelection = document.getElementById('minScore');
const maxScoreSelection = document.getElementById('maxScore');
const sortBySelection = document.getElementById('sortBy');

let pageSize = 10
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
    users = await fetchProducts();
    startUserRead = users[0];
    lastUserRead = users[users.length-1]
    endPage = startPage + users.length / pageSize -1
    displayUsers()
    updatePagination()
}

async function fetchProducts() {
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
            productSortType : sortBy,
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

async function fetchNextProducts() {
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
            pageSize: fetchSize,
            cursorUserId: lastProductRead.id,
            cursorUserName : lastProductRead.name,
            cursorUserScore : lastProductRead.score,
            productSortType : sortBy,
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
            productSortType : sortBy,
            sortBy: sortBy
        }));
        if (!response.ok) {
            throw new Error('Error fetching products.');
        }
        return await response.json();
    } catch (error) {
        console.log(error)
        alert('Error fetching products.')
        return [];
    }
}

// Function to display products in the table
async function displayProducts() {
    tableBody.innerHTML = '';
    for(let index = 0; index < pageSize; index++) {
        const productIndex = ((currentPage -1) % pageCount ) * pageSize + index
        const user = products[productIndex]
        const row = document.createElement('tr');
        row.id = `user-${user.id}`
        row.innerHTML = `
          <td>${user.id}</td>
          <td>${user.name}</td>
          <td>${user.price}</td>
          <td>${user.quantity}</td>
          <td><button onclick="orderButtonEventHandler(${user.id})">Order</button></td>`;
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
            products = await fetchPrevProducts();
            startProductRead = products[0];
            lastProductRead = products[products.length-1]
            await displayProducts();
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
            await displayProducts();
            await updatePagination()
        });
        pagination.appendChild(link);
    }

    if(products.length >= pageSize * pageCount) {
        const rightArrow = document.createElement('a');
        rightArrow.href = '#';
        rightArrow.textContent = '>>';
        rightArrow.className = "pagination-link";
        rightArrow.addEventListener('click', async () => {
            currentPage = endPage + 1;
            startPage = currentPage
            products = await fetchNextProducts()
            startProductRead = products[0];
            lastProductRead = products[products.length-1]
            endPage = startPage + products.length / pageSize -1
            await displayProducts()
            await updatePagination()
        });
        pagination.appendChild(rightArrow);
    }
}

// Initial fetch and display of products
fetchAndDisplayUsers();
