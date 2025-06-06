"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
Updated for modern PyTorch API (2024) - Standalone version without torchtext dependency
"""
from torch.utils.data import DataLoader as TorchDataLoader
from collections import Counter, defaultdict
import torch
import re


def basic_tokenizer(text):
    """基础分词器"""
    # 将文本转为小写并分词
    text = text.lower()
    # 简单的分词：按空格和标点符号分割
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class Vocab:
    """词汇表类"""
    def __init__(self, tokens=None, min_freq=1, specials=None):
        self.min_freq = min_freq
        self.specials = specials or []
        self.itos = []  # index to string
        self.stoi = {}  # string to index
        
        if tokens:
            self.build(tokens)
    
    def build(self, tokens):
        """构建词汇表"""
        # 统计词频
        counter = Counter()
        for token_list in tokens:
            counter.update(token_list)
        
        # 添加特殊token
        self.itos = self.specials.copy()
        
        # 添加满足最小频率要求的token
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.itos:
                self.itos.append(token)
        
        # 构建string to index映射
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        
        # 设置默认索引（通常是<unk>的索引）
        self.default_index = self.stoi.get('<unk>', 0)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)
    
    def __len__(self):
        return len(self.itos)
    
    def get_itos(self):
        return self.itos


class VocabWrapper:
    """提供与旧API兼容的词汇表包装器"""
    def __init__(self, vocab):
        self.vocab = vocab
        
    @property
    def stoi(self):
        """string to index映射，兼容旧API"""
        return self.vocab.stoi
    
    @property  
    def itos(self):
        """index to string映射，兼容旧API"""
        return self.vocab.itos
    
    def __len__(self):
        return len(self.vocab)


class FieldWrapper:
    """提供与旧API兼容的Field包装器"""
    def __init__(self, vocab):
        self.vocab = VocabWrapper(vocab)


class DataLoader:
    source_vocab = None
    target_vocab = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en if tokenize_en else basic_tokenizer
        self.tokenize_de = tokenize_de if tokenize_de else basic_tokenizer
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        print('dataset initializing start')

    def create_sample_data(self):
        """创建示例数据集"""
        # 德语到英语的示例数据
        sample_data_de_en = [
            ("ich bin ein student", "i am a student"),
            ("das ist ein buch", "this is a book"),
            ("wir gehen zur schule", "we go to school"),
            ("er liebt seine arbeit", "he loves his work"),
            ("sie trinkt kaffee", "she drinks coffee"),
            ("das wetter ist schön", "the weather is nice"),
            ("wir essen zu abend", "we eat dinner"),
            ("ich lese ein buch", "i read a book"),
            ("sie spricht deutsch", "she speaks german"),
            ("er fährt nach hause", "he drives home"),
            ("das ist mein freund", "this is my friend"),
            ("wir kaufen lebensmittel", "we buy groceries"),
            ("ich höre musik", "i listen to music"),
            ("sie schreibt einen brief", "she writes a letter"),
            ("das auto ist rot", "the car is red"),
            ("wir besuchen das museum", "we visit the museum"),
            ("er arbeitet im büro", "he works in the office"),
            ("sie kocht das abendessen", "she cooks dinner"),
            ("ich schaue fernsehen", "i watch television"),
            ("das haus ist groß", "the house is big"),
            ("wir spielen fußball", "we play football"),
            ("sie lernt französisch", "she learns french"),
            ("er trinkt wasser", "he drinks water"),
            ("das buch ist interessant", "the book is interesting"),
            ("wir fahren in urlaub", "we go on vacation"),
            ("ich liebe meine familie", "i love my family"),
            ("sie arbeitet als lehrerin", "she works as a teacher"),
            ("das kind spielt im garten", "the child plays in the garden"),
            ("wir gehen ins kino", "we go to the cinema"),
            ("er studiert medizin", "he studies medicine"),
            ("ich trinke tee", "i drink tea"),
            ("sie liest eine zeitung", "she reads a newspaper"),
            ("wir besuchen unsere großeltern", "we visit our grandparents"),
            ("das fenster ist offen", "the window is open"),
            ("er spielt gitarre", "he plays guitar"),
            ("sie tanzt sehr gut", "she dances very well"),
            ("ich kaufe neue kleidung", "i buy new clothes"),
            ("das restaurant ist teuer", "the restaurant is expensive"),
            ("wir lernen zusammen", "we study together"),
            ("sie hilft ihrer mutter", "she helps her mother")
        ]
        
        # 英语到德语的示例数据（相反顺序）
        sample_data_en_de = [(en, de) for de, en in sample_data_de_en]
        
        if self.ext == ('.de', '.en'):
            data = sample_data_de_en
            self.source_tokenizer = self.tokenize_de
            self.target_tokenizer = self.tokenize_en
        elif self.ext == ('.en', '.de'):
            data = sample_data_en_de
            self.source_tokenizer = self.tokenize_en
            self.target_tokenizer = self.tokenize_de
        else:
            raise ValueError(f"Unsupported language pair: {self.ext}")
        
        # 分割数据集
        total_len = len(data)
        train_len = int(0.8 * total_len)
        valid_len = int(0.1 * total_len)
        
        train_data = data[:train_len]
        valid_data = data[train_len:train_len+valid_len]
        test_data = data[train_len+valid_len:]
        
        return train_data, valid_data, test_data

    def make_dataset(self):
        """创建数据集"""
        print("使用内置示例数据集")
        return self.create_sample_data()

    def build_vocab(self, train_data, min_freq=2):
        """构建词汇表"""
        # 收集源语言tokens
        source_tokens = []
        for src_text, _ in train_data:
            tokens = [self.init_token] + self.source_tokenizer(src_text) + [self.eos_token]
            source_tokens.append(tokens)
        
        # 收集目标语言tokens
        target_tokens = []
        for _, tgt_text in train_data:
            tokens = [self.init_token] + self.target_tokenizer(tgt_text) + [self.eos_token]
            target_tokens.append(tokens)
        
        # 构建词汇表
        specials = [self.pad_token, self.unk_token, self.init_token, self.eos_token]
        
        self.source_vocab = Vocab(source_tokens, min_freq=min_freq, specials=specials)
        self.target_vocab = Vocab(target_tokens, min_freq=min_freq, specials=specials)
        
        # 创建兼容的属性以支持旧API
        self.source = FieldWrapper(self.source_vocab)
        self.target = FieldWrapper(self.target_vocab)

    def process_data(self, data):
        """处理数据，将文本转换为数字序列"""
        processed_data = []
        for src_text, tgt_text in data:
            # 分词并添加特殊token
            src_tokens = [self.init_token] + self.source_tokenizer(src_text) + [self.eos_token]
            tgt_tokens = [self.init_token] + self.target_tokenizer(tgt_text) + [self.eos_token]
            
            # 转换为索引
            src_indices = [self.source_vocab[token] for token in src_tokens]
            tgt_indices = [self.target_vocab[token] for token in tgt_tokens]
            
            processed_data.append((src_indices, tgt_indices))
        
        return processed_data

    def collate_fn(self, batch):
        """批处理函数，用于DataLoader"""
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]
        
        # 获取最大长度
        src_max_len = max(len(seq) for seq in src_batch)
        tgt_max_len = max(len(seq) for seq in tgt_batch)
        
        # 填充序列
        src_padded = []
        tgt_padded = []
        
        pad_idx = self.source_vocab[self.pad_token]
        
        for src_seq in src_batch:
            padded = src_seq + [pad_idx] * (src_max_len - len(src_seq))
            src_padded.append(padded)
            
        for tgt_seq in tgt_batch:
            padded = tgt_seq + [pad_idx] * (tgt_max_len - len(tgt_seq))
            tgt_padded.append(padded)
        
        # 创建兼容的批次对象
        class BatchWrapper:
            def __init__(self, src, trg):
                self.src = src
                self.trg = trg
        
        return BatchWrapper(torch.tensor(src_padded), torch.tensor(tgt_padded))

    def make_iter(self, train, validate, test, batch_size, device):
        """创建数据迭代器"""
        # 处理数据
        train_processed = self.process_data(train)
        valid_processed = self.process_data(validate)
        test_processed = self.process_data(test)
        
        # 创建DataLoader
        train_iterator = TorchDataLoader(
            train_processed,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        valid_iterator = TorchDataLoader(
            valid_processed,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        test_iterator = TorchDataLoader(
            test_processed,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator


# 为了兼容性，添加一个辅助函数
def idx_to_word(indices, vocab_wrapper):
    """将索引转换为单词，兼容旧API"""
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    
    words = []
    for idx in indices:
        if idx < len(vocab_wrapper.itos):
            words.append(vocab_wrapper.itos[idx])
        else:
            words.append('<unk>')
    
    return words
