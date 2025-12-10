
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
        <div class="youtube-player">
            <video loop muted playsinline>
                <source src="${video.path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div class="video-overlay">
                <div class="play-button">
                    <svg height="100%" version="1.1" viewBox="0 0 68 48" width="100%">
                        <path d="M66.52,7.74c-0.78-2.93-2.49-5.41-5.42-6.19C55.79,.13,34,0,34,0S12.21,.13,6.9,1.55 C3.97,2.33,2.27,4.81,1.48,7.74C0.06,13.05,0,24,0,24s0.06,10.95,1.48,16.26c0.78,2.93,2.49,5.41,5.42,6.19 C12.21,47.87,34,48,34,48s21.79-0.13,27.1-1.55c2.93-0.78,4.64-3.26,5.42-6.19C67.94,34.95,68,24,68,24S67.94,13.05,66.52,7.74z" fill="#0ea5e9"></path>
                        <path d="M 45,24 27,14 27,34" fill="#fff"></path>
                    </svg>
                </div>
            </div>
            <div class="video-controls">
                <button class="fullscreen-btn" title="Fullscreen">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
                    </svg>
                </button>
            </div>
        </div>
        <div class="video-info">
            <div class="video-title">${video.name}</div>
            <div class="video-metadata">
                <span class="video-type ${typeClass}">
                    <i class="fas ${icon}"></i> ${categoryLabel}
                </span>
                <span class="video-format">${typeLabel}</span>
            </div>
        </div>
    `;
    
    // Add click handler for play button
    const videoElement = card.querySelector('video');
    const overlay = card.querySelector('.video-overlay');
    const fullscreenBtn = card.querySelector('.fullscreen-btn');
    
    overlay.addEventListener('click', () => {
        if (videoElement.paused) {
            videoElement.play();
            overlay.style.opacity = '0';
            overlay.style.pointerEvents = 'none';
        }
    });
    
    videoElement.addEventListener('click', () => {
        if (!videoElement.paused) {
            videoElement.pause();
            overlay.style.opacity = '1';
            overlay.style.pointerEvents = 'all';
        }
    });
    
    videoElement.addEventListener('play', () => {
        overlay.style.opacity = '0';
        overlay.style.pointerEvents = 'none';
    });
    
    videoElement.addEventListener('pause', () => {
        overlay.style.opacity = '1';
        overlay.style.pointerEvents = 'all';
    });
    
    // Fullscreen functionality
    fullscreenBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const player = card.querySelector('.youtube-player');
        
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            if (player.requestFullscreen) {
                player.requestFullscreen();
            } else if (player.webkitRequestFullscreen) {
                player.webkitRequestFullscreen();
            } else if (player.msRequestFullscreen) {
                player.msRequestFullscreen();
            }
        }
    });
        
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
        const types = card.getAttribute('data-type').split(' ');
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

// Disable animations
// const observerOptions = {
//     threshold: 0.1,
//     rootMargin: '0px 0px -50px 0px'
// };

// const observer = new IntersectionObserver((entries) => {
//     entries.forEach(entry => {
//         if (entry.isIntersecting) {
//             entry.target.classList.add('animate');
//         }
//     });
// }, observerOptions);

// document.querySelectorAll('.card, .method-card').forEach(el => {
//     observer.observe(el);
// });

document.addEventListener('DOMContentLoaded', loadVideos);
