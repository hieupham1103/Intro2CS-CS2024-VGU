
async function loadVideos() {
    const videoGrid = document.getElementById('videoGrid');
    
    try {
        const response = await fetch('videos.json');
        const videos = await response.json();
        
        videoGrid.innerHTML = '';
        
        if (videos.length === 0) {
            videoGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-secondary);">No videos found. Run update_videos.py to scan for videos.</div>';
            return;
        }
        
        videos.forEach(video => {
            const card = createVideoCard(video);
            videoGrid.appendChild(card);
        });
        
        // Re-observe new cards for animation
        document.querySelectorAll('.video-card').forEach(el => {
            observer.observe(el);
        });
        
    } catch (error) {
        console.error('Error loading videos:', error);
        videoGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 3rem; color: var(--text-secondary);">
                <i class="fas fa-exclamation-triangle" style="font-size: 2rem; margin-bottom: 1rem; display: block; color: var(--warning);"></i>
                Could not load videos. Make sure videos.json exists.<br>
                <small>Run: python update_videos.py</small>
            </div>`;
    }
}

/**
 * Create a video card element
 * @param {Object} video - Video data object
 * @returns {HTMLElement} - The video card element
 */
function createVideoCard(video) {
    const card = document.createElement('div');
    card.className = 'video-card';
    card.setAttribute('data-type', `${video.type} ${video.category}`);
    
    const icon = video.category === 'drone' ? 'fa-drone' : 'fa-dove';
    const typeClass = video.category === 'drone' ? 'drone' : 'bird';
    const categoryLabel = video.category === 'drone' ? 'Drone' : 'Bird';
    const typeLabel = video.type.toUpperCase();
    
    card.innerHTML = `
        <video controls preload="metadata">
            <source src="${video.path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <div class="video-info">
            <div class="video-title">${video.name}</div>
            <div class="video-type ${typeClass}">
                <i class="fas ${icon}"></i> ${categoryLabel} Detection - ${typeLabel}
            </div>
        </div>
    `;
    
    return card;
}


/**
 * Filter videos by type (all, rgb, ir, drone, bird)
 * @param {string} filter - The filter to apply
 */
function filterVideos(filter) {
    const cards = document.querySelectorAll('.video-card');
    const buttons = document.querySelectorAll('.tab-btn');
    
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    cards.forEach(card => {
        const types = card.getAttribute('data-type');
        if (filter === 'all' || types.includes(filter)) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}


const scrollTopBtn = document.querySelector('.scroll-top');

window.addEventListener('scroll', () => {
    if (window.pageYOffset > 300) {
        scrollTopBtn.classList.add('visible');
    } else {
        scrollTopBtn.classList.remove('visible');
    }
});

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}


document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        target.scrollIntoView({
            behavior: 'smooth'
        });
    });
});


const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate');
        }
    });
}, observerOptions);

document.querySelectorAll('.card, .method-card').forEach(el => {
    observer.observe(el);
});

document.addEventListener('DOMContentLoaded', loadVideos);
