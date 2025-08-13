# 🧠 မြန်မာဘာသာစကား NLP Transformer

**Advanced Myanmar Natural Language Processing using Transformer Architecture with TensorFlow.js**

[![Node.js](https://img.shields.io/badge/Node.js-16+-green.svg)](https://nodejs.org/)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22+-orange.svg)](https://tensorflow.org/js)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🌟 အဓိက အင်္ဂါရပ်များ

- **🤖 Real Transformer Architecture** - Multi-head attention, residual connections, layer normalization
- **🇲🇲 Myanmar Language Support** - Optimized for Myanmar text processing
- **💬 Interactive Chatbot** - Web-based chat interface with multiple response options
- **✍️ Text Generation** - Creative text generation with temperature control
- **📊 Advanced Training** - Early stopping, learning rate scheduling, validation
- **💾 Model Persistence** - Save and load trained models
- **🌐 Web API** - RESTful endpoints for integration

## 🚀 စတင်အသုံးပြုခြင်း

### လိုအပ်သော စနစ်များ

- Node.js 16+
- npm သို့မဟုတ် yarn

### ထည့်သွင်းခြင်း

```bash
# Repository ကို clone လုပ်ပါ
git clone https://github.com/your-username/myanmar-nlp.git
cd myanmar-nlp

# Dependencies များကို ထည့်သွင်းပါ
npm install

# Project ကို setup လုပ်ပါ
npm run setup
```

### အသုံးပြုခြင်း

```bash
# Model ကို train လုပ်ပါ (ပထမဆုံး အကြိမ်)
npm run train

# Chatbot server ကို စတင်ပါ
npm start

# Development mode တွင် စတင်ပါ
npm run dev
```

## 📖 အသုံးပြုနည်း

### 1. Model Training

```bash
npm run train
```

ဒီ command က မြန်မာဘာသာစကား text data ကို အသုံးပြုပြီး transformer model ကို train လုပ်ပေးပါတယ်။

### 2. Web Interface

Browser တွင် `http://localhost:3001` ကို ဖွင့်ပါ။ မြန်မာဘာသာစကား chatbot interface ကို တွေ့ရပါမယ်။

### 3. API Endpoints

#### Chat Endpoint

```bash
POST /chat
{
  "text": "ငလျင်အကြောင်း ပြောပြပါ",
  "generateMultiple": true
}
```

#### Text Generation Endpoint

```bash
POST /generate
{
  "prompt": "ငလျင်သည်",
  "maxLength": 100,
  "temperature": 0.8
}
```

#### Health Check

```bash
GET /health
```

## 🏗️ Architecture

### Transformer Model

- **Multi-head Attention**: 8 attention heads
- **Model Dimensions**: 256 (dModel), 1024 (dff)
- **Layers**: 6 transformer blocks
- **Sequence Length**: 128 tokens
- **Dropout**: 0.1

### Training Process

- **Optimizer**: Adamax with learning rate 0.001
- **Loss Function**: Sparse Categorical Crossentropy
- **Callbacks**: Early stopping, learning rate reduction
- **Validation**: 20% split for monitoring

## 📁 Project Structure

```
myanmar-nlp/
├── transformer.js          # Transformer architecture
├── run_transformer.js      # Training script
├── server.js              # Web server & API
├── input_myanmar.txt      # Training data
├── package.json           # Dependencies
├── README.md              # Documentation
└── transformer_checkpoint/ # Saved models
```

## 🔧 Configuration

Model parameters များကို `server.js` တွင် ပြင်ဆင်နိုင်ပါတယ်:

```javascript
let modelConfig = {
  seqLen: 128, // Sequence length
  dModel: 256, // Model dimension
  numHeads: 8, // Number of attention heads
  dff: 1024, // Feed-forward dimension
  numBlocks: 6, // Number of transformer blocks
};
```

## 📊 Performance Tips

1. **GPU Usage**: CUDA-enabled GPU ရှိပါက TensorFlow.js က GPU ကို အသုံးပြုပါမယ်
2. **Batch Size**: Memory ပေါ်မူတည်ပြီး batch size ကို ပြင်ဆင်ပါ
3. **Sequence Length**: ပိုရှည်သော sequence များက ပိုကောင်းသော result များကို ရရှိစေပါတယ်

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Batch size ကို လျှော့ချပါ
2. **Model Not Loading**: `transformer_checkpoint` folder ကို စစ်ဆေးပါ
3. **Poor Quality**: Training epochs ကို တိုးမြှင့်ပါ

### Debug Commands

```bash
# Model ကို ပြန်လည် train လုပ်ပါ
npm run clean
npm run train

# Server logs ကို ကြည့်ရှုပါ
npm start
```

## 🤝 Contributing

မြန်မာဘာသာစကား NLP ကို ပိုကောင်းအောင် ပြုလုပ်ရာတွင် ပါဝင်လိုပါက:

1. Fork လုပ်ပါ
2. Feature branch တစ်ခု ဖန်တီးပါ
3. Changes များကို commit လုပ်ပါ
4. Pull request ပေးပါ

## 📄 License

MIT License - ဒီ project ကို လွတ်လပ်စွာ အသုံးပြုနိုင်ပါတယ်။

## 🙏 Acknowledgments

- TensorFlow.js team
- Myanmar language community
- Open source contributors

## 📞 Contact

အကြံပြုချက်များ သို့မဟုတ် မေးခွန်းများအတွက်:

- GitHub Issues: [Create Issue](https://github.com/your-username/myanmar-nlp/issues)
- Email: your-email@example.com

---

**မြန်မာဘာသာစကား AI ကို အတူတကွ တည်ဆောက်ကြပါစို့! 🇲🇲✨**
