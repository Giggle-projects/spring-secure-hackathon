// const currentDomain = window.location.origin
const currentDomain = "http://localhost:8080"

const tableBody = document.getElementById('table-body');
const pagination = document.getElementById('pagination');

const userNameInput = document.getElementById('userName');
const pageSizeSelection = document.getElementById('pageSize');
const sortBySelection = document.getElementById('sortBy');
const dateFromSelection = document.getElementById('dateFrom');
const dateToSelection = document.getElementById('dateTo');

let pageSize = 20
let currentPage = 1;
let users = []
let hiPage =1

function blockButtonEventHandler(memberId) {
    const orderUrl = currentDomain + "/api/admin/block/users?memberId="+memberId
    const response = fetch(orderUrl, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        }
    });
    sAlert("부정사용자 등록")

}

userNameInput.addEventListener("keydown", function(event) {
    if(event.keyCode === 13) {
        fetchAndDisplayUsers().then(() => {});
    }
});

userNameInput.addEventListener("change", function(event) {
    fetchAndDisplayUsers().then(() => {});
});

sortBySelection.addEventListener("change", (event) => {
  fetchAndDisplayUsers().then(() => {});
});

pageSizeSelection.addEventListener("change", (event) => {
  fetchAndDisplayUsers().then(() => {});
});
async function fetchAndDisplayUsers() {
    currentPage = 1; // Reset current page to 1 when fetching new data
    pageSize = parseInt(pageSizeSelection.value); // Update pageSize based on selection
    users = await fetchUser();
    await displayUsers();
    await updatePagination();
}


async function fetchUser() {
    try {
        const memberName = userNameInput.value.trim();
        const dateFrom = dateFromSelection.value.trim();
        const dateTo = dateToSelection.value.trim();
        const sortBy = sortBySelection.options[sortBySelection.selectedIndex].value;

        const fetchUrl = currentDomain + "/api/admin/users/access?"
        let response = await fetch(fetchUrl + new URLSearchParams({
            memberName: memberName,
            size: pageSize, // Use the updated pageSize
            dateFrom: dateFrom,
            dateTo: dateTo,
            sort: sortBy
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

// Function to display products in the table
async function displayUsers() {
    tableBody.innerHTML = '';
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;

    for (let index = startIndex; index < endIndex && index < users.length; index++) {
        const user = users[index];
        const row = document.createElement('tr');
        row.id = `user-${user.id}`;
        row.innerHTML = `
          <td>${user.dateTime}</td>
          <td>${user.memberId}</td>
          <td>${user.memberName}</td>
          <td>${user.memberEmail}</td>
          <td>${user.memberEncryptedPassword}</td>
          <td><button onclick="blockButtonEventHandler(${user.memberId})">Block</button></td>`;
        tableBody.appendChild(row);
    }
}
function navigateToPage(option) {
    if (option === "data") {
        // Navigate to the data page (replace with the actual URL)
        window.location.href = "./admin-data.html";
    } else if (option === "email") {
        // Navigate to the email page (replace with the actual URL)
        window.location.href = "./admin-access.html";
    }
}


async function updatePagination() {
    pagination.innerHTML = '';

    const totalPages = Math.ceil(users.length / pageSize);
    const maxVisiblePages = 10;

    const startPage = Math.max(currentPage - Math.floor(maxVisiblePages / 2), 1);
    const endPage = Math.min(startPage + maxVisiblePages - 1, totalPages);

    if (currentPage > 1) {
        const leftArrow = document.createElement('a');
        leftArrow.href = '#';
        leftArrow.textContent = '<<';
        leftArrow.className = "pagination-link";
        leftArrow.addEventListener('click', async () => {
            currentPage--;
            await updatePagination();
            await displayUsers();
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
            await updatePagination();
            await displayUsers();
        });
        pagination.appendChild(link);
    }

    if (currentPage < totalPages) {
        const rightArrow = document.createElement('a');
        rightArrow.href = '#';
        rightArrow.textContent = '>>';
        rightArrow.className = "pagination-link";
        rightArrow.addEventListener('click', async () => {
            currentPage++;
            await updatePagination();
            await displayUsers();
        });
        pagination.appendChild(rightArrow);
    }
}


// Initial fetch and display of products
fetchAndDisplayUsers();
updatePagination()

// sAlert('custom alert example!');
function noError_sAlert(txt, title = 'Success',) {
    Swal.fire({
        title: title,
        text: txt,
        confirmButtonText: '닫기'
    });
}