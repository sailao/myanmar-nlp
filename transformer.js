
const tf = require('@tensorflow/tfjs');

// Build a simple but effective transformer-inspired model
function buildTransformerModel(seqLen, dModel, numHeads, dff, vocabSize, numBlocks = 3, dropoutRate = 0.1) {
  const model = tf.sequential();
  
  // Input layer
  model.add(tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: dModel,
    inputLength: seqLen
  }));
  
  // Flatten the embedding output to 2D
  model.add(tf.layers.flatten());
  
  // Dense layers for processing
  for (let i = 0; i < numBlocks; i++) {
    model.add(tf.layers.dense({
      units: dff,
      activation: 'relu'
    }));
    
    model.add(tf.layers.dropout({ rate: dropoutRate }));
    
    model.add(tf.layers.dense({
      units: dModel
    }));
    
    model.add(tf.layers.dropout({ rate: dropoutRate }));
    
    // Layer normalization
    model.add(tf.layers.layerNormalization({ axis: -1 }));
  }
  
  // Output layer for language modeling
  model.add(tf.layers.dense({
    units: vocabSize,
    activation: 'softmax'
  }));
  
  return model;
}

// Text generation function with proper input handling
function generateText(model, vocab, startTokens, maxLength = 50, temperature = 1.0) {
  const reverseVocab = {};
  Object.keys(vocab).forEach(key => {
    reverseVocab[vocab[key]] = key;
  });
  
  let currentTokens = [...startTokens];
  const seqLen = model.inputs[0].shape[1];
  
  for (let i = 0; i < maxLength; i++) {
    // Prepare input with proper padding
    let inputTokens = currentTokens.slice(-seqLen);
    
    // Pad or truncate to match sequence length
    while (inputTokens.length < seqLen) {
      inputTokens.unshift(0); // <PAD> token
    }
    if (inputTokens.length > seqLen) {
      inputTokens = inputTokens.slice(-seqLen);
    }
    
    // Create input tensor
    const input = tf.tensor([inputTokens]);
    
    try {
      // Get prediction
      const prediction = model.predict(input);
      const logits = tf.log(prediction).div(tf.scalar(temperature));
      const nextToken = tf.multinomial(tf.softmax(logits, -1), 1).arraySync()[0][0];
      
      currentTokens.push(nextToken);
      
      // Stop if we generate a special token or reach max length
      if (nextToken === vocab['<END>'] || currentTokens.length >= maxLength) {
        break;
      }
      
      // Clean up tensors
      input.dispose();
      prediction.dispose();
      logits.dispose();
      
    } catch (err) {
      console.error('Error in text generation:', err);
      break;
    }
  }
  
  return currentTokens.map(tokenId => reverseVocab[tokenId] || '?').join(' ');
}

module.exports = { 
  buildTransformerModel, 
  generateText
};
