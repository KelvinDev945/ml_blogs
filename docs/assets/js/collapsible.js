/**
 * Collapsible sections functionality
 * Allows long derivations and proofs to be collapsed/expanded
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all collapsible sections
    initializeCollapsibles();
});

function initializeCollapsibles() {
    const collapsibleHeaders = document.querySelectorAll('.collapsible-header');

    collapsibleHeaders.forEach(header => {
        // Add click event listener
        header.addEventListener('click', function() {
            this.classList.toggle('active');
            const content = this.nextElementSibling;

            if (content && content.classList.contains('collapsible-content')) {
                content.classList.toggle('active');
            }
        });

        // Check if should be collapsed by default
        const section = header.parentElement;
        const shouldCollapse = section.hasAttribute('data-collapsed') ||
                             section.classList.contains('collapsed-by-default');

        if (!shouldCollapse) {
            // Expand by default
            header.classList.add('active');
            const content = header.nextElementSibling;
            if (content && content.classList.contains('collapsible-content')) {
                content.classList.add('active');
            }
        }
    });
}

// Function to expand all collapsibles
function expandAll() {
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.classList.add('active');
        const content = header.nextElementSibling;
        if (content && content.classList.contains('collapsible-content')) {
            content.classList.add('active');
        }
    });
}

// Function to collapse all collapsibles
function collapseAll() {
    document.querySelectorAll('.collapsible-header').forEach(header => {
        header.classList.remove('active');
        const content = header.nextElementSibling;
        if (content && content.classList.contains('collapsible-content')) {
            content.classList.remove('active');
        }
    });
}

// Make functions globally available
window.expandAll = expandAll;
window.collapseAll = collapseAll;
