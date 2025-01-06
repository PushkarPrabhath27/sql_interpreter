document.getElementById('queryForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const userInput = document.getElementById('userInput').value;
    const schemaInput = document.getElementById('schemaInput').value;
    const resultDiv = document.getElementById('result');
    const sqlOutput = document.getElementById('sqlOutput');
    
    // Show loading state
    sqlOutput.textContent = 'Generating SQL...';
    resultDiv.classList.remove('hidden');
    
    try {
        console.log('Sending request with:', { userInput, schemaInput });
        
        const response = await fetch('/generate-sql', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                userInput,
                schemaInput
            })
        });
        
        const data = await response.json();
        console.log('Received response:', data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display the result
        sqlOutput.textContent = data.sql || 'No SQL generated';
        
    } catch (error) {
        console.error('Error:', error);
        sqlOutput.textContent = `Error: ${error.message}`;
    }
});

// Add syntax highlighting
document.getElementById('sqlOutput').addEventListener('DOMSubtreeModified', () => {
    hljs.highlightElement(document.getElementById('sqlOutput'));
}); 