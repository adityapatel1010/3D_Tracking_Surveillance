/**
 * Reference Object Selection Module
 * Handles interactive canvas-based object selection using SAM
 */

class ReferenceObjectSelector {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.canvas = null;
        this.ctx = null;
        this.currentFilename = '';
        this.firstFrameImage = null;
        this.selectedMask = null;
        this.isSelecting = false;

        this.initializeUI();
    }

    initializeUI() {
        // Create reference object selection card (insert after upload section)
        const uploadCard = document.querySelector('.upload-section').parentElement;

        const refCard = document.createElement('div');
        refCard.className = 'card';
        refCard.id = 'referenceObjectCard';
        refCard.style.display = 'none';

        refCard.innerHTML = `
            <h2 style="margin-bottom: 20px; color: #333;">🎯 Select Reference Object</h2>
            
            <div class="alert alert-info" style="margin-bottom: 20px;">
                <strong>Instructions:</strong> Click on the payment terminal or reference object in the image below.
                The system will track people based on their distance to this object.
            </div>
            
            <div style="position: relative; display: inline-block; margin-bottom: 20px;">
                <canvas id="referenceCanvas" 
                        style="border: 3px solid #667eea; border-radius: 10px; cursor: crosshair; max-width: 100%; height: auto;">
                </canvas>
                <div id="selectionSpinner" style="display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
                    <div class="loading-spinner"></div>
                    <p style="color: white; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; margin-top: 10px;">
                        Segmenting object...
                    </p>
                </div>
            </div>
            
            <div id="referenceInfo" style="display: none; margin-bottom: 20px;">
                <div class="alert alert-success">
                    <strong>✅ Object segmented!</strong><br>
                    Area: <span id="refArea"></span> pixels<br>
                    Centroid: <span id="refCentroid"></span>
                </div>
            </div>
            
            <div style="text-align: center;">
                <button class="btn" id="confirmReferenceBtn" style="display: none;">Confirm Selection</button>
                <button class="btn btn-secondary" id="reselectBtn" style="display: none;">Re-select Object</button>
            </div>
        `;

        uploadCard.insertAdjacentElement('afterend', refCard);

        // Get canvas and context
        this.canvas = document.getElementById('referenceCanvas');
        this.ctx = this.canvas.getContext('2d');

        // Attach event listeners
        this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
        document.getElementById('confirmReferenceBtn').addEventListener('click', () => this.confirmReference());
        document.getElementById('reselectBtn').addEventListener('click', () => this.resetSelection());
    }

    async showFirstFrame(filename) {
        this.currentFilename = filename;
        this.isSelecting = false;

        try {
            // Fetch first frame
            const response = await fetch(`${this.apiUrl}/api/first-frame/${filename}`);
            const data = await response.json();

            if (data.status === 'success') {
                // Load image
                const img = new Image();
                img.onload = () => {
                    this.firstFrameImage = img;

                    // Set canvas size
                    this.canvas.width = data.width;
                    this.canvas.height = data.height;

                    // Draw image
                    this.ctx.drawImage(img, 0, 0);

                    // Show card
                    document.getElementById('referenceObjectCard').style.display = 'block';

                    // Scroll to card
                    document.getElementById('referenceObjectCard').scrollIntoView({
                        behavior: 'smooth',
                        block: 'center'
                    });
                };
                img.src = 'data:image/jpeg;base64,' + data.frame;
            }
        } catch (error) {
            console.error('Failed to load first frame:', error);
            alert('Failed to load first frame. Please try again.');
        }
    }

    async handleCanvasClick(event) {
        if (this.isSelecting) return;  // Prevent multiple clicks

        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const x = Math.floor((event.clientX - rect.left) * scaleX);
        const y = Math.floor((event.clientY - rect.top) * scaleY);

        console.log(`Clicked at (${x}, ${y})`);

        this.isSelecting = true;
        document.getElementById('selectionSpinner').style.display = 'block';

        try {
            // Send to SAM for segmentation
            const response = await fetch(`${this.apiUrl}/api/segment-object`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: this.currentFilename,
                    x: x,
                    y: y
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                // Load and display overlay
                const overlayImg = new Image();
                overlayImg.onload = () => {
                    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    this.ctx.drawImage(overlayImg, 0, 0);

                    this.selectedMask = data;

                    // Show info
                    document.getElementById('refArea').textContent = data.area.toLocaleString();
                    document.getElementById('refCentroid').textContent =
                        `(${data.centroid[0]}, ${data.centroid[1]})`;
                    document.getElementById('referenceInfo').style.display = 'block';

                    // Show buttons
                    document.getElementById('confirmReferenceBtn').style.display = 'inline-block';
                    document.getElementById('reselectBtn').style.display = 'inline-block';

                    document.getElementById('selectionSpinner').style.display = 'none';
                    this.isSelecting = false;
                };
                overlayImg.src = 'data:image/jpeg;base64,' + data.overlay;
            }
        } catch (error) {
            console.error('Segmentation failed:', error);
            alert('Object segmentation failed. Please try clicking on a different location.');
            document.getElementById('selectionSpinner').style.display = 'none';
            this.isSelecting = false;
        }
    }

    async confirmReference() {
        try {
            const response = await fetch(`${this.apiUrl}/api/confirm-reference`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: this.currentFilename })
            });

            const data = await response.json();

            if (data.status === 'success') {
                // Hide reference card
                document.getElementById('referenceObjectCard').style.display = 'none';

                // Enable process button
                document.getElementById('processBtn').disabled = false;

                // Show success message
                const uploadCard = document.querySelector('.upload-section').parentElement;
                const successMsg = document.createElement('div');
                successMsg.className = 'alert alert-success';
                successMsg.style.marginTop = '20px';
                successMsg.innerHTML = `
                    <strong>✅ Reference object confirmed!</strong><br>
                    You can now process the video. The system will track people approaching this object.
                `;
                uploadCard.appendChild(successMsg);

                // Scroll to process button
                document.getElementById('processBtn').scrollIntoView({
                    behavior: 'smooth',
                    block: 'center'
                });
            }
        } catch (error) {
            console.error('Failed to confirm reference:', error);
            alert('Failed to confirm reference object. Please try again.');
        }
    }

    resetSelection() {
        // Reset canvas to original image
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.firstFrameImage, 0, 0);

        // Hide info and buttons
        document.getElementById('referenceInfo').style.display = 'none';
        document.getElementById('confirmReferenceBtn').style.display = 'none';
        document.getElementById('reselectBtn').style.display = 'none';

        this.selectedMask = null;
    }
}

// Export for use in main script
window.ReferenceObjectSelector = ReferenceObjectSelector;
