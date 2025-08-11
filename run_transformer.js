// Example usage of TransformerBlock with text data
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const { buildTransformerModel } = require('../transformer');

// 1. Read and tokenize text
function readAndTokenize(filePath) {
  const text = fs.readFileSync(filePath, 'utf8');
  // Simple whitespace tokenizer (customize for Myanmar text if needed)
  return text.split(/\s+/);
}

// 2. Build vocabulary and encode tokens
function buildVocab(tokens) {
  const vocab = {};
  let idx = 0;
  tokens.forEach(token => {
    if (!(token in vocab)) {
      vocab[token] = idx++;
    }
  });
  return vocab;
}

function encodeTokens(tokens, vocab) {
  return tokens.map(token => vocab[token]);
}

// 3. Create embeddings
function createEmbeddings(tokenIds, vocabSize, embedDim) {
  // Random embeddings for demonstration
  const embeddingMatrix = tf.randomNormal([vocabSize, embedDim]);
  return tf.gather(embeddingMatrix, tf.tensor1d(tokenIds, 'int32'));
}

// 4. Build, train, save, and load transformer model
async function runTransformerModel(filePath) {
  const embedDim = 32;
  const numHeads = 4;
  const ffDim = 64;

  const tokens = readAndTokenize(filePath);
  const vocab = buildVocab(tokens);
  const tokenIds = encodeTokens(tokens, vocab);
  const seqLen = tokenIds.length - 1; // for next-token prediction
  const vocabSize = Object.keys(vocab).length;

  // Create trainable embedding matrix
  const embeddingMatrix = tf.variable(tf.randomNormal([vocabSize, embedDim]), true, 'embeddingMatrix');

  // Prepare training data: sliding window batching
  const windowSize = 32;
  const xTrain = [];
  const yTrain = [];
  for (let i = 0; i < tokenIds.length - windowSize; i++) {
    const xTokenIds = tokenIds.slice(i, i + windowSize);
    const yTokenId = tokenIds[i + windowSize];
    xTrain.push(tf.gather(embeddingMatrix, tf.tensor1d(xTokenIds, 'int32')).arraySync());
    yTrain.push(yTokenId);
  }
  // Convert to tensors
  // Add positional encoding to input embeddings
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
  const posEncoding = getPositionalEncoding(windowSize, embedDim);
  let xTrainTensor = tf.tensor(xTrain); // [numSamples, windowSize, embedDim]
  xTrainTensor = xTrainTensor.add(posEncoding);

  // Extract last token embedding from each window
  const lastTokenEmbeddings = xTrainTensor.slice([0, windowSize - 1, 0], [-1, 1, -1]).reshape([xTrainTensor.shape[0], embedDim]); // [numSamples, embedDim]
  const yTrainTensor = tf.tensor(yTrain).expandDims(-1).cast('float32'); // [numSamples, 1]


  // Try to load model checkpoint, else build new model
  let model;
  const checkpointPath = 'file://transformer_checkpoint/model.json';
  try {
    model = await tf.loadLayersModel(checkpointPath);
    console.log('Loaded model from checkpoint');
  } catch (err) {
    model = buildTransformerModel(1, embedDim, numHeads, ffDim, vocabSize, 2, 0.1);
    console.log('Built new advanced model');
  }
  model.summary();

  // Compile model
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });

  // Train model (more epochs for better results)
  await model.fit(lastTokenEmbeddings, yTrainTensor, {
    epochs: 30,
    batchSize: 32,
    shuffle: true,
    verbose: 1
  });

  // Save model checkpoint
  await model.save('file://transformer_checkpoint');
  console.log('Model saved to transformer_checkpoint');

  // Load model checkpoint
  const loadedModel = await tf.loadLayersModel('file://transformer_checkpoint/model.json');
  console.log('Model loaded from checkpoint');
  const output = loadedModel.predict(lastTokenEmbeddings);
  output.print();
}

runTransformerModel('input_myanmar.txt');
