const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');

  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    const fileName = fileInput.files[0].name;
    dropZone.textContent = fileName;
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    const fileName = fileInput.files[0].name;
    dropZone.textContent = fileName;
  }
});
