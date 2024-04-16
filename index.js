function redirectToMonthPage(year) {
    const overlay = document.getElementById('pageTransitionOverlay');
    overlay.classList.remove('hidden'); 
    setTimeout(() => {
        window.location.href = `months.html?year=${year}`;
    }, 100); 
}