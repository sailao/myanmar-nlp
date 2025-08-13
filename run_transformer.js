const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const { buildTransformerModel, generateText } = require('./transformer');

// Improved tokenization for Myanmar text
function tokenizeMyanmarText(text) {
  // Split by sentences first, then by words
  const sentences = text.split(/[။\n]+/).filter(s => s.trim().length > 0);
  const tokens = [];
  
  sentences.forEach(sentence => {
    // Split by spaces and punctuation
    const words = sentence.split(/[\s\u200b]+/).filter(w => w.trim().length > 0);
    tokens.push(...words);
    tokens.push('<SEP>'); // Sentence separator
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

// Create training sequences with proper padding
function createSequences(tokenIds, seqLen) {
  const sequences = [];
  const targets = [];
  
  for (let i = 0; i < tokenIds.length - seqLen; i++) {
    const sequence = tokenIds.slice(i, i + seqLen);
    const target = tokenIds[i + seqLen];
    
    // Pad sequence if needed
    while (sequence.length < seqLen) {
      sequence.unshift(0); // <PAD> token
    }
    
    sequences.push(sequence);
    targets.push(target);
  }
  
  return { sequences, targets };
}

// Main training function
async function runTransformerModel(filePath) {
  console.log('🚀 Starting Myanmar NLP Transformer Training...');
  
  // Reduced model hyperparameters for stability
  const seqLen = 64;  // Reduced from 128
  const dModel = 128; // Reduced from 256
  const numHeads = 4; // Reduced from 8
  const dff = 512;    // Reduced from 1024
  const numBlocks = 3; // Reduced from 6
  const dropoutRate = 0.1;
  const batchSize = 16; // Reduced from 32
  const epochs = 20;    // Reduced from 50
  
  console.log('📖 Reading Myanmar text data...');
  const text = fs.readFileSync(filePath, 'utf8');
  
  console.log('✂️ Tokenizing text...');
  const tokens = tokenizeMyanmarText(text);
  console.log(`📊 Total tokens: ${tokens.length}`);
  
  console.log('🔤 Building vocabulary...');
  const vocab = buildVocab(tokens);
  const vocabSize = Object.keys(vocab).length;
  console.log(`📚 Vocabulary size: ${vocabSize}`);
  
  console.log('🔄 Encoding tokens...');
  const tokenIds = tokens.map(token => vocab[token] || vocab['<UNK>']);
  
  console.log('📝 Creating training sequences...');
  const { sequences, targets } = createSequences(tokenIds, seqLen);
  console.log(`🎯 Training samples: ${sequences.length}`);
  
  // Convert to tensors
  const xTrain = tf.tensor(sequences);
  const yTrain = tf.tensor(targets);
  
  console.log('🏗️ Building transformer model...');
  const model = buildTransformerModel(seqLen, dModel, numHeads, dff, vocabSize, numBlocks, dropoutRate);
  
  // Compile model with better optimizer
  model.compile({
    optimizer: tf.train.adam(0.001), // Changed from adamax to adam
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  console.log('📋 Model summary:');
  model.summary();
  
  console.log('🎯 Starting training...');
  
  // Basic training without callbacks to avoid compatibility issues
  const history = await model.fit(xTrain, yTrain, {
    epochs,
    batchSize,
    validationSplit: 0.2,
    shuffle: true,
    verbose: 1
  });
  
  console.log('💾 Saving model...');
  await model.save('file://transformer_checkpoint');
  console.log('✅ Model saved successfully!');
  
  // Test text generation
  console.log('\n🧪 Testing text generation...');
  const testPrompt = ['ငလျင်', 'အကြောင်း'];
  const testTokens = testPrompt.map(token => vocab[token] || vocab['<UNK>']);
  
  try {
    const generatedText = generateText(model, vocab, testTokens, 20, 0.8);
    console.log('🎭 Generated text:');
    console.log(generatedText);
  } catch (err) {
    console.log('❌ Text generation test failed:', err.message);
  }
  
  // Clean up tensors
  xTrain.dispose();
  yTrain.dispose();
  
  console.log('\n🎉 Training completed successfully!');
  return { model, vocab, history };
}

// Run the training
if (require.main === module) {
  runTransformerModel('input_myanmar.txt')
    .then(() => {
      console.log('🎊 All done!');
      process.exit(0);
    })
    .catch(err => {
      console.error('❌ Error during training:', err);
      process.exit(1);
    });
}

module.exports = { runTransformerModel, tokenizeMyanmarText, buildVocab };
