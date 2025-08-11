
const tf = require('@tensorflow/tfjs');

// Build a simple transformer model as a tf.Model



function getPositionalEncoding(seqLen, embedDim) {
  // Standard sinusoidal positional encoding
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

function transformerBlock(x, embedDim, numHeads, ffDim, dropoutRate) {
  // LayerNorm
  let out = tf.layers.layerNormalization({axis: -1}).apply(x);
  // Self-attention (simplified)
  out = tf.layers.dense({units: embedDim, activation: 'relu'}).apply(out);
  // Dropout
  out = tf.layers.dropout({rate: dropoutRate}).apply(out);
  // Feed-forward
  out = tf.layers.layerNormalization({axis: -1}).apply(out);
  out = tf.layers.dense({units: ffDim, activation: 'relu'}).apply(out);
  out = tf.layers.dense({units: embedDim}).apply(out);
  // Dropout
  out = tf.layers.dropout({rate: dropoutRate}).apply(out);
  return out;
}

function buildTransformerModel(seqLen, embedDim, numHeads, ffDim, vocabSize, numBlocks = 2, dropoutRate = 0.1) {
  // Input: embedding for last token in window
  const input = tf.input({shape: [embedDim]});

  let x = input;
  // Stack multiple transformer blocks
  for (let i = 0; i < numBlocks; i++) {
    x = tf.layers.dense({units: embedDim, activation: 'relu'}).apply(x);
    x = tf.layers.dropout({rate: dropoutRate}).apply(x);
    x = tf.layers.dense({units: ffDim, activation: 'relu'}).apply(x);
    x = tf.layers.dense({units: embedDim}).apply(x);
    x = tf.layers.dropout({rate: dropoutRate}).apply(x);
  }

  // Output layer (for language modeling, use vocabSize)
  const output = tf.layers.dense({units: vocabSize, activation: 'softmax'}).apply(x);
  // Build model
  const model = tf.model({inputs: input, outputs: output});
  return model;
}

module.exports = { buildTransformerModel };
