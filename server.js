import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { initializeLangChain, config } from './Exercise/02-chatbot-history.js';

const app = express();
const PORT = 3000;

// __dirname workaround for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

app.use(cors());
app.use(bodyParser.json());

// Serve the static HTML file
app.use(express.static(path.join(__dirname)));

let chainInstance; // Variable to hold the LangChain sequence instance

// Initialize LangChain sequence on server start
initializeLangChain()
  .then((chain) => {
    chainInstance = chain; // Store the initialized chain instance
    console.log('LangChain initialized.');
  })
  .catch((err) => {
    console.error('Error initializing LangChain:', err);
  });

// Endpoint to handle user messages
app.post('/api/chat', async (req, res) => {
  const userMessage = req.body.message;

  try {
    // Check if LangChain sequence is initialized
    if (!chainInstance) {
      throw new Error('LangChain sequence is not initialized.');
    }

    // Invoke the LangChain sequence with user input
    const response = await chainInstance.invoke({ input: userMessage }, config);
    console.log(response);

    // Extract the response text from the LangChain response object
    const aiMessage = response.content; // Adjust based on your actual response structure
    console.log(aiMessage);

    // Send the AI message back to the frontend
    res.json({ response: aiMessage });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Something went wrong' });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
