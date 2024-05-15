// Fetch data for a given page
async function fetchDataForPage(path) {
    const response = await fetch(path);
    return response.text();
}

// Handle navigation with smooth transitions
async function spaNavigate(path) {
    const data = await fetchDataForPage(path);
    if (!document.startViewTransition || matchMedia('(prefers-reduced-motion: reduce)').matches) {
        await updateDOMForPage(data);
        console.timeLog("Loaded");
        return;
    }

    const transition = document.startViewTransition(() => updateDOMForPage(data));
    await transition.finished;
}

// Update DOM with new page content
function updateDOMForPage(data) {
    document.open();
    document.write(data);
    document.close();
}

// Add event listeners for navigation
document.addEventListener('DOMContentLoaded', () => {
    const navigateLinks = document.querySelectorAll('.navigate-link');
    navigateLinks.forEach(link => {
        link.addEventListener('click', async (e) => {
            e.preventDefault();
            await spaNavigate(link.getAttribute('href'));
        });
    });
});