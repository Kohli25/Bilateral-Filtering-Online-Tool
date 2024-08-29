// script.js
document.getElementById('imageForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const image1 = document.getElementById('image1').files[0];
    const image2 = document.getElementById('image2').files[0];

    if (!image1 || !image2) {
        alert("Please upload both images.");
        return;
    }

    const formData = new FormData();
    formData.append('image1', image1);
    formData.append('image2', image2);

    const socket = io();

    socket.on('update', function(data) {
        const progressElement = document.createElement('p');
        progressElement.textContent = data.message;
        document.getElementById('progress').appendChild(progressElement);
    });

    try {
        const response = await fetch('/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Error processing images');
        }

        const result = await response.json();
        document.getElementById('outputImage').src = result.processedImageUrl;
    } catch (error) {
        alert("An error occurred: " + error.message);
    }
});
