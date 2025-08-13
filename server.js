const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { buildTransformerModel, generateText } = require('./transformer');

const app = express();
app.use(express.json());
app.use(express.static('public'));

// Global variables
let vocab = null;
let model = null;
let modelConfig = {
  seqLen: 64,  // Reduced from 128
  dModel: 128, // Reduced from 256
  numHeads: 4, // Reduced from 8
  dff: 512,    // Reduced from 1024
  numBlocks: 3 // Reduced from 6
};

// Improved tokenization for Myanmar text
function tokenizeMyanmarText(text) {
  const sentences = text.split(/[။\n]+/).filter(s => s.trim().length > 0);
  const tokens = [];
  
  sentences.forEach(sentence => {
    const words = sentence.split(/[\s\u200b]+/).filter(w => w.trim().length > 0);
    tokens.push(...words);
    tokens.push('<SEP>');
  });
  
  return tokens;
}

// Build vocabulary with special tokens
function buildVocab(tokens) {
  const vocab = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<START>': 2,
    '<END>': 3,
    '<SEP>': 4
  };
  
  let idx = 5;
  tokens.forEach(token => {
    if (!(token in vocab)) {
      vocab[token] = idx++;
    }
  });
  
  return vocab;
}

// Load vocabulary and model
async function initializeSystem() {
  try {
    console.log('📖 Loading Myanmar text data...');
    const text = fs.readFileSync('input_myanmar.txt', 'utf8');
    
    console.log('✂️ Tokenizing text...');
    const tokens = tokenizeMyanmarText(text);
    
    console.log('🔤 Building vocabulary...');
    vocab = buildVocab(tokens);
    console.log(`📚 Vocabulary size: ${Object.keys(vocab).length}`);
    
    console.log('🏗️ Loading transformer model...');
    try {
      model = await tf.loadLayersModel('file://transformer_checkpoint/model.json');
      console.log('✅ Model loaded from checkpoint');
    } catch (err) {
      console.log('🆕 Building new model...');
      const vocabSize = Object.keys(vocab).length;
      model = buildTransformerModel(
        modelConfig.seqLen,
        modelConfig.dModel,
        modelConfig.numHeads,
        modelConfig.dff,
        vocabSize,
        modelConfig.numBlocks
      );
      console.log('✅ New model built');
    }
    
    console.log('🚀 System initialized successfully!');
  } catch (err) {
    console.error('❌ Error initializing system:', err);
    throw err;
  }
}

// Prepare input for model
function prepareInput(text) {
  const tokens = text.split(/[\s\u200b]+/).filter(w => w.trim().length > 0);
  const tokenIds = tokens.map(token => vocab[token] || vocab['<UNK>']);
  
  // Pad or truncate to seqLen
  let padded = tokenIds.slice(-modelConfig.seqLen);
  while (padded.length < modelConfig.seqLen) {
    padded.unshift(0); // <PAD> token
  }
  
  return tf.tensor([padded]);
}

// Generate multiple response options
function generateResponses(inputText, numOptions = 3) {
  try {
    const inputTensor = prepareInput(inputText);
    const prediction = model.predict(inputTensor);
    const logits = tf.log(prediction);
    
    const responses = [];
    for (let i = 0; i < numOptions; i++) {
      // Use different temperature for variety
      const temperature = 0.7 + (i * 0.2);
      
      // Generate text using the start tokens from input
      const startTokens = inputText.split(/[\s\u200b]+/).slice(-5).map(token => 
        vocab[token] || vocab['<UNK>']
      );
      
      try {
        const generatedText = generateText(model, vocab, startTokens, 20, temperature);
        responses.push({
          text: generatedText,
          confidence: temperature,
          type: i === 0 ? 'primary' : 'alternative'
        });
      } catch (genErr) {
        console.error('Generation error for option', i, genErr);
        responses.push({
          text: 'ကျေးဇူးပြု၍ နောက်တစ်ကြိမ် ပြန်လည်ကြိုးစားကြည့်ပါ။',
          confidence: 0.0,
          type: 'error'
        });
      }
    }
    
    // Clean up tensors
    inputTensor.dispose();
    prediction.dispose();
    logits.dispose();
    
    return responses;
  } catch (err) {
    console.error('Error generating responses:', err);
    return [{
      text: 'ကျေးဇူးပြု၍ နောက်တစ်ကြိမ် ပြန်လည်ကြိုးစားကြည့်ပါ။',
      confidence: 0.0,
      type: 'error'
    }];
  }
}

// Chat endpoint with improved functionality
app.post('/chat', async (req, res) => {
  const { text, generateMultiple = false } = req.body;
  
  if (!text || !text.trim()) {
    return res.status(400).json({ 
      error: 'ကျေးဇူးပြု၍ စာသားတစ်ခုခု ထည့်သွင်းပါ။' 
    });
  }
  
  try {
    if (generateMultiple) {
      const responses = generateResponses(text, 3);
      res.json({ 
        success: true,
        responses,
        originalText: text
      });
    } else {
      const responses = generateResponses(text, 1);
      res.json({ 
        success: true,
        response: responses[0],
        originalText: text
      });
    }
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ 
      error: 'ဆာဗာတွင် အမှားတစ်ခု ဖြစ်ပွားနေပါသည်။',
      details: err.message 
    });
  }
});

// Generate text endpoint
app.post('/generate', async (req, res) => {
  const { prompt, maxLength = 50, temperature = 0.8 } = req.body;
  
  if (!prompt || !prompt.trim()) {
    return res.status(400).json({ 
      error: 'ကျေးဇူးပြု၍ စာသားတစ်ခုခု ထည့်သွင်းပါ။' 
    });
  }
  
  try {
    const startTokens = prompt.split(/[\s\u200b]+/).map(token => 
      vocab[token] || vocab['<UNK>']
    );
    
    const generatedText = generateText(model, vocab, startTokens, maxLength, temperature);
    
    res.json({
      success: true,
      generatedText,
      prompt,
      parameters: { maxLength, temperature }
    });
  } catch (err) {
    console.error('Generation error:', err);
    res.status(500).json({ 
      error: 'စာသားထုတ်လုပ်ရာတွင် အမှားတစ်ခု ဖြစ်ပွားနေပါသည်။',
      details: err.message 
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    modelLoaded: model !== null,
    vocabSize: vocab ? Object.keys(vocab).length : 0,
    timestamp: new Date().toISOString()
  });
});

// Serve a simple chat interface
app.get('/', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="my">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>မြန်မာဘာသာစကား AI Chatbot</title>
        <style>
            body { font-family: 'Pyidaungsu', 'Myanmar Text', sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 800px; margin: 0 auto; }
            .chat-box { background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; backdrop-filter: blur(10px); }
            .input-group { margin: 20px 0; }
            input[type="text"], textarea { width: 100%; padding: 15px; border: none; border-radius: 10px; font-size: 16px; background: rgba(255,255,255,0.9); }
            button { background: #4CAF50; color: white; padding: 15px 30px; border: none; border-radius: 10px; cursor: pointer; font-size: 16px; margin: 10px 5px; }
            button:hover { background: #45a049; }
            .response { background: rgba(255,255,255,0.2); padding: 15px; margin: 10px 0; border-radius: 10px; }
            .multiple-responses { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }
            .response-card { background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 မြန်မာဘာသာစကား AI Chatbot</h1>
            <div class="chat-box">
                <div class="input-group">
                    <textarea id="userInput" placeholder="သင့်စကားကို ဒီနေရာတွင် ရေးသားပါ..." rows="3"></textarea>
                </div>
                <button onclick="sendMessage()">💬 စကားပြောရန်</button>
                <button onclick="generateMultiple()">🎲 ရွေးချယ်မှုများ</button>
                <button onclick="generateText()">✍️ စာသားထုတ်လုပ်ရန်</button>
                
                <div id="responseArea"></div>
            </div>
        </div>
        
        <script>
            async function sendMessage() {
                const text = document.getElementById('userInput').value.trim();
                if (!text) return;
                
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                
                const data = await response.json();
                displayResponse(data.response.text, '💬 AI ထံမှ အဖြေ');
            }
            
            async function generateMultiple() {
                const text = document.getElementById('userInput').value.trim();
                if (!text) return;
                
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, generateMultiple: true })
                });
                
                const data = await response.json();
                displayMultipleResponses(data.responses);
            }
            
            async function generateText() {
                const text = document.getElementById('userInput').value.trim();
                if (!text) return;
                
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: text, maxLength: 100, temperature: 0.8 })
                });
                
                const data = await response.json();
                displayResponse(data.generatedText, '✍️ ထုတ်လုပ်ထားသော စာသား');
            }
            
            function displayResponse(text, title) {
                const area = document.getElementById('responseArea');
                area.innerHTML = \`<div class="response"><h3>\${title}</h3><p>\${text}</p></div>\`;
            }
            
            function displayMultipleResponses(responses) {
                const area = document.getElementById('responseArea');
                let html = '<h3>🎲 ရွေးချယ်မှုများ</h3><div class="multiple-responses">';
                
                responses.forEach((resp, i) => {
                    html += \`<div class="response-card"><h4>\${resp.type === 'primary' ? '🌟 အဓိက' : '💡 အခြား'} \${i+1}</h4><p>\${resp.text}</p></div>\`;
                });
                
                html += '</div>';
                area.innerHTML = html;
            }
        </script>
    </body>
    </html>
  `);
});

// Start server
const PORT = process.env.PORT || 3001;

async function startServer() {
  try {
    await initializeSystem();
    
    app.listen(PORT, () => {
      console.log(`🚀 Myanmar NLP Chatbot server running on port ${PORT}`);
      console.log(`🌐 Open http://localhost:${PORT} in your browser`);
    });
  } catch (err) {
    console.error('❌ Failed to start server:', err);
    process.exit(1);
  }
}

startServer();
