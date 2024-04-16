/*function redirectToMonthPage(year) {
    document.body.innerHTML = `<div id="monthSelection">
        <h1>Select a Month in ${year}</h1>
        <!-- All Months -->
        <button onclick="alert('Selected January ${year}')">January</button>
        <button onclick="alert('Selected February ${year}')">February</button>
        <button onclick="alert('Selected March ${year}')">March</button>
        <button onclick="alert('Selected April ${year}')">April</button>
        <button onclick="alert('Selected May ${year}')">May</button>
        <button onclick="alert('Selected June ${year}')">June</button>
        <button onclick="alert('Selected July ${year}')">July</button>
        <button onclick="alert('Selected August ${year}')">August</button>
        <button onclick="alert('Selected September ${year}')">September</button>
        <button onclick="alert('Selected October ${year}')">October</button>
        <button onclick="alert('Selected November ${year}')">November</button>
        <button onclick="alert('Selected December ${year}')">December</button>
        <button onclick="window.location.reload();">Back to Year Selection</button>
    </div>`;
}*/
function redirectToMonthPage(year) {
    const overlay = document.getElementById('pageTransitionOverlay');
    overlay.classList.remove('hidden'); 
    setTimeout(() => {
        window.location.href = `months.html?year=${year}`;
    }, 100); 
}

/*function redirectToMonthPage(year) {
    // Display the month selection area
    var monthSelection = document.getElementById('monthSelection');
    monthSelection.style.display = 'block';
    monthSelection.innerHTML = '<h2>Select a Month in ' + year + '</h2>';

    // Example of adding buttons for each month
    var months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December'];
    months.forEach(function(month) {
        var button = document.createElement('button');
        button.innerText = month;
        button.onclick = function() { showImage(month); };
        monthSelection.appendChild(button);
    });
}

function showImage(month) {
    // Logic to determine the image source based on selected month
    var imageUrl = 'path_to_images/' + month + '.png'; // Modify path as needed
    var imageDisplay = document.getElementById('imageDisplay');
    var selectedImage = document.getElementById('selectedImage');

    selectedImage.src = imageUrl;
    selectedImage.alt = 'Water Surface Map of ' + month;
    
    imageDisplay.style.display = 'block';
}*/



