const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const { buildTransformerModel } = require('../transformer');

const app = express();
app.use(express.json());

// Load vocab and checkpoint
let vocab = null;
let model = null;
let embedDim = 32;
let numHeads = 4;
let ffDim = 64;
let windowSize = 32;
let vocabSize = null;
let embeddingMatrix = null;

function loadVocab() {
  const text = fs.readFileSync('input_myanmar.txt', 'utf8');
  const tokens = text.split(/\s+/);
  vocab = {};
  let idx = 0;
  tokens.forEach(token => {
    if (!(token in vocab)) {
      vocab[token] = idx++;
    }
  });
  vocabSize = Object.keys(vocab).length;
  // Create embedding matrix once
  if (!embeddingMatrix) {
    embeddingMatrix = tf.variable(tf.randomNormal([vocabSize, embedDim]), true, 'embeddingMatrix');
  }
}

async function loadModel() {
  try {
    model = await tf.loadLayersModel('file://transformer_checkpoint/model.json');
    console.log('Loaded model from checkpoint');
  } catch (err) {
    model = buildTransformerModel(1, embedDim, numHeads, ffDim, vocabSize, 2, 0.1);
    console.log('Built new model');
  }
}

function getPositionalEncoding(seqLen, embedDim) {
  const PE = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const row = [];
    for (let i = 0; i < embedDim; i++) {
      if (i % 2 === 0) {
        row.push(Math.sin(pos / Math.pow(10000, i / embedDim)));
      } else {
        row.push(Math.cos(pos / Math.pow(10000, (i - 1) / embedDim)));
      }
    }
    PE.push(row);
  }
  return tf.tensor(PE, [seqLen, embedDim]);
}

function encodeTokens(tokens) {
  return tokens.map(token => vocab[token] || 0);
}

function prepareInput(text) {
  const tokens = text.split(/\s+/);
  const tokenIds = encodeTokens(tokens);
  // Pad or truncate to windowSize
  let padded = tokenIds.slice(-windowSize);
  while (padded.length < windowSize) padded.unshift(0);
  // Use global embeddingMatrix
  const inputEmb = tf.gather(embeddingMatrix, tf.tensor1d(padded, 'int32')).arraySync();
  let inputTensor = tf.tensor(inputEmb).add(getPositionalEncoding(windowSize, embedDim));
  // Extract last token embedding
  return inputTensor.slice([windowSize - 1, 0], [1, embedDim]).reshape([1, embedDim]);
}

app.post('/chat', async (req, res) => {
  const userText = req.body.text;
  if (!userText) return res.status(400).json({ error: 'Missing text' });
  try {
    const inputTensor = prepareInput(userText);
    const output = model.predict(inputTensor);
    const outputArr = output.arraySync()[0];
    // Get top token and its probability
    const maxProb = Math.max(...outputArr);
    const topIdx = outputArr.indexOf(maxProb);
    let token = '';
    // Only reply if probability > 0.03 (3%)
    if (maxProb > 0.03) {
      token = Object.keys(vocab).find(key => vocab[key] === topIdx) || '';
    }
    res.json({ reply: token });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, async () => {
  loadVocab();
  await loadModel();
  console.log(`Chatbot server running on port ${PORT}`);
});
