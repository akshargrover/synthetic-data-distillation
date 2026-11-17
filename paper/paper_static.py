"""Static (results-independent) text of the paper: front matter, related work,
references. Results-dependent sections live in build_paper.py."""

TITLE = ("Teacher-to-Student Synthetic Data Distillation for Low-Compute "
         "Domain-Specific Text Classification")

AUTHORS = [
    ["Akshar Grover", "Dept. of Artificial Intelligence and Data Science",
     "Dr. Akhilesh Das Gupta Institute of Professional Studies", "(Affiliated to GGSIPU)",
     "Delhi, India", "[email address]"],
    ["Ritesh Kumar", "Dept. of Artificial Intelligence and Data Science",
     "Dr. Akhilesh Das Gupta Institute of Professional Studies", "(Affiliated to GGSIPU)",
     "Delhi, India", "riteshchandel@adgips.ac.in"],
]

KEYWORDS = ("synthetic data generation, knowledge distillation, large language models, "
            "DistilBERT, semantic deduplication, intent classification, few-shot learning, "
            "low-compute inference")

REFERENCES = [
    'V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019.',
    'G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," arXiv:1503.02531, 2015.',
    'X. Jiao *et al.*, "TinyBERT: Distilling BERT for natural language understanding," in *Findings of the Association for Computational Linguistics: EMNLP 2020*, 2020, pp. 4163–4174.',
    'Y. Wang *et al.*, "Self-Instruct: Aligning language models with self-generated instructions," in *Proc. 61st Annu. Meeting of the Association for Computational Linguistics (ACL)*, 2023, pp. 13484–13508.',
    'Y. Meng, J. Huang, Y. Zhang, and J. Han, "Generating training data with language models: Towards zero-shot language understanding," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 35, 2022.',
    'J. Ye *et al.*, "ZeroGen: Efficient zero-shot learning via dataset generation," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2022, pp. 11653–11669.',
    'T. Schick and H. Schütze, "Generating datasets with pretrained language models," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2021, pp. 6943–6951.',
    'L. Tunstall *et al.*, "Efficient few-shot learning without prompts," arXiv:2209.11055, 2022.',
    'G. Sahu, P. Rodriguez, I. H. Laradji, P. Atighehchian, D. Vazquez, and D. Bahdanau, "Data augmentation for intent classification with off-the-shelf large language models," in *Proc. 4th Workshop on NLP for Conversational AI*, 2022, pp. 47–57.',
    'Y. Yu *et al.*, "Large language model as attributed training data generator: A tale of diversity and bias," in *Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track*, vol. 36, 2023.',
    'I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, and Y. Gal, "AI models collapse when trained on recursively generated data," *Nature*, vol. 631, pp. 755–759, 2024.',
    'K. Lee *et al.*, "Deduplicating training data makes language models better," in *Proc. 60th Annu. Meeting of the Association for Computational Linguistics (ACL)*, 2022, pp. 8424–8445.',
    'A. Abbas, K. Tirumala, D. Simig, S. Ganguli, and A. S. Morcos, "SemDeDup: Data-efficient learning at web-scale through semantic deduplication," arXiv:2303.09540, 2023.',
    'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. EMNLP-IJCNLP*, 2019, pp. 3982–3992.',
    'W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 5776–5788.',
    'I. Casanueva, T. Temčinas, D. Gerz, M. Henderson, and I. Vulić, "Efficient intent detection with dual sentence encoders," in *Proc. 2nd Workshop on NLP for Conversational AI*, 2020, pp. 38–45.',
    'Qwen Team, "Qwen2.5 technical report," arXiv:2412.15115, 2024.',
    'T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.',
    'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.',
    'T. Brown *et al.*, "Language models are few-shot learners," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 1877–1901.',
    'J. Li, M. Galley, C. Brockett, J. Gao, and B. Dolan, "A diversity-promoting objective function for neural conversation models," in *Proc. NAACL-HLT*, 2016, pp. 110–119.',
    'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.',
    'B. Jacob *et al.*, "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 2704–2713.',
    'T. Wolf *et al.*, "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP: System Demonstrations*, 2020, pp. 38–45.',
    'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.',
    'S. Wang, Y. Liu, Y. Xu, C. Zhu, and M. Zeng, "Want to reduce labeling cost? GPT-3 can help," in *Findings of the Association for Computational Linguistics: EMNLP 2021*, 2021, pp. 4195–4205.',
]
