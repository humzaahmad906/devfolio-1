module.exports = {
  siteMetadata: {
    // Site URL for when it goes live
    siteUrl: `https://elastic-meninsky-aa7c74.netlify.app/`,
    // Your Name
    name: 'Humza Ahmad',
    // Main Site Title
    title: `Humza Ahmad | Machine Learning Engineer`,
    // Description that goes under your name in main bio
    description: `Machine Learning Engineer with expertise in Computer Vision and Natural Language Processing`,
    // Optional: Twitter account handle
    author: `@iamhamzu906`,
    // Optional: Github account URL
    github: `https://github.com/humzaahmad906`,
    // Optional: LinkedIn account URL
    linkedin: `https://www.linkedin.com/in/humza-ahmad-7a2830130/`,
    // Content of the About Me section
    about: `Machine Learning Engineer with Computer Vision and NLP expertise with 3+ years experience in data science, machine learning system development and deployment.
    
    Expertise:
    
    - Computer Vision
    - Semantic and Instance Segmentation
    - Object Tracking, Recognition and Detection
    - Image Classification
    - Natural Language Processing
    - Sentiment Textual Similarity
    - Sentiment Analysis
    - Text Classification
    - Statistical Data Analytics
    - Unsupervised Learning Techniques
    - Recommendation system
    - Data visualization and cleaning
    
    Skills:
    
    - Libraries and Frameworks: Scikit-learn, PyTorch, TensorFlow, Hugging Face, NumPy, Pandas, Matplotlib, Seaborn, Jupyter Notebook, Anaconda, Flask, Django, NLTK, Open-CV
    - Algorithms: Deep Neural Networks, Convolutional Neural Networks, Transformers, BERT, T5, Recurrent Neural Networks, Linear Regression, Logistic Regression, Support Vector Machines, Naive Bayes, KNN, Decision Tree, Random Forest, Gradient Boosting, Principal Component Analysis, KMeans
    - Programming languages: Python, C++
    - Work management tools: Jira, Confluence, Trello
    - DevOps Tools: Git, Docker, Google Cloud Platform (GCP), Amazon Web Services (AWS)
    - Database Technologies: PostgreSQL, MySQL, Redis
    - Front-end Technologies: HTML, Bootstrap, CSS, Javascript and ReactJs.`,
    // Optional: List your projects, they must have `name` and `description`. `link` is optional.
    projects: [
      {
        name: 'Hellofact',
        description:
            `Implement Recursive WebCrawler and Search Engine for Legal Casino documents.
            Search Engine has Download, OCR, Name Entity Recognition Tagging and Indexing functionalities.
            ▪ Download pipeline using Requests, ThreadPool, Selenium and Zen-scrape.
            ▪ OCR of Pdf files using Tesseract OCR and Abbyy to convert into raw text and styled
            documents.
            ▪ NER pipeline using HuggingFace to get categories of sub-documents.
            ▪ Indexing pipeline to index text document into MongoDB and Pinecone respectively with GPT3 ADA-002 Embeddings for further use.
            ▪ Information Retrieval and Search pipeline to retrieve best results for search query from indexed documents with GPT3 ADA-002 Embeddings and generate answers using GPT3 Davinci-003 Completion API.`,
        link: 'https://github.com/RyanFitzgerald/devfolio',
      },
      {
        name: 'Person Identification in multi-camera system',
        description:
          'The aim of this project was to track a person on multi-camera setup such that its ID\n' +
            'remains constant. Working of this pipeline is as follows.\n' +
            '▪ Person face and body embeddings will be registered in redis and mysql server when\n' +
            'the person first appears in the frame.\n' +
            '▪ Before Registration the pipeline will check whether the face is registered in current\n' +
            'database by taking cosine similarity of his embeddings with previously added\n' +
            'embeddings.\n' +
            '▪ When the server gets closed, all the face and body embeddings will be reloaded from\n' +
            'mysql server to redis.',
        link: 'https://github.com/RyanFitzgerald/devfolio',
      },
      {
        name: 'Background Remover',
        description:
          'Implement end-to-end pipeline for background removal task. Fine-tune Mask-RCNN and\n' +
            'Yolact for automatic background removal and implement from scratch f-BRS\n' +
            '(Rethinking Background Refinement for Segmentation) and Super-pixel plus grab cut\n' +
            'algorithm for interactive image segmentation.\n' +
            '▪ Collect, clean, and pre-process the data and convert it to universal coco format for\n' +
            'training Image Segmentation models.\n' +
            '▪ Implement interactive segmentation pipeline using Super-pixel to divide each image\n' +
            'into big chunks and used grab cut algorithm for the foreground chunks separation\n' +
            'with Scikit-learn and OpenCV.\n' +
            '▪ Fine-tuned f-BRS model for interactive segmentation with fewer clicks.\n' +
            '▪ Mask-RCNN and Yolact fine-tuning to get foreground mask for automatic\n' +
            'background removal.\n' +
            '▪ Deep alpha matting for smoother foreground masks.',
        link: 'https://chromeextensionkit.com/?ref=devfolio',
      },
      {
        name: 'Health-app',
        description:
          'Prediction of meal plan and exercise required to gain or lose weight.\n' +
            '▪ Train XGBoost classifier to predict duration of exercise from previous exercise\n' +
            'duration, original weight, required weight and height.\n' +
            '▪ Train classifier to predict intake calories from the same features.\n' +
            '▪ After prediction of calories intake we use Nearest Neighbor algorithm to get the\n' +
            'meal-plan from fit-bit data-set of the closest intake calories.',
        link: 'https://github.com/RyanFitzgerald/devfolio',
      },
      {
        name: 'Semantic Textual Similarity',
        description:
            `Implement training and prediction BERT base and Roberta using HuggingFace and Pytorch for Semantic Similarity and match it to categorize FAQ answers
            ▪ Data preprocessing and post-processing using Pandas and Numpy.
            ▪ Fine-tune and Validate both the models using HuggingFace and Pytorch.
            ▪ Train both the models and compared the Accuracy, F1-score, Recall, and Precision
            metrics.`,
        link: 'https://github.com/RyanFitzgerald/devfolio',
      },
      {
        name: 'FNIR based activity segmentation',
        description:
            `Classification on different difficulty level tasks using machine learning and statistical
              techniques on data-set got from 12 near-infrared sensors that are used to measure
              oxidized and deoxidized hemoglobin. Use the classification report to get the best model.
              ▪ Clean time steps that contained no information about the difficulty levels.
              ▪ Select windows of variable lengths to derive features like mean, median, max, min,
              skewness to convert 24 features to 120 features.
              ▪ Use feature selection to extract the best 50 features from 120.
              ▪ Implement SVM, Polynomial Regression, and Artificial Neural Networks in Scikit-
              learn and Keras. PCA and LDA implementation for dimensionality reduction.
              ▪ Train and validate the data-set using Accuracy, F1-score, Recall, and Precision
              metrics. Select the best window length using the scores on validation data.`,
        link: 'https://github.com/RyanFitzgerald/devfolio',
      },
      {
        name: 'Tumor Segmentation',
        description:
            `Implement training and prediction pipeline of vanilla U-Net and V3 Inception based U-
              Net for image segmentation of benign and malignant tumors.
              ▪ Data pre-processing and post-processing using Pandas and Numpy.
              ▪ Implement U-Net with V3 inception modules and vanilla U-Net using python andTensorFlow.
              ▪ Train both the models and compared the Accuracy, MIOU, F1-score, Recall,
              Precision, and Dice Coefficient metrics.
              ▪ Validate both models and hyper-parameter tuning to get better results.`,
        link: 'https://github.com/RyanFitzgerald/devfolio',
      },
    ],
    // Optional: List your experience, they must have `name` and `description`. `link` is optional.
    experience: [
      {
        name: 'Acme Corp',
        description: 'Full-Stack Developer, February 2020 - Present',
        link: 'https://github.com/RyanFitzgerald/devfolio',
      },
      {
        name: 'Globex Corp',
        description: 'Full-Stack Developer, December 2017 - February 2020',
        link: 'https://github.com/RyanFitzgerald/devfolio',
      },
      {
        name: 'Hooli',
        description: 'Full-Stack Developer, May 2015 - December 2017',
        link: 'https://github.com/RyanFitzgerald/devfolio',
      },
    ],
    // Optional: List your skills, they must have `name` and `description`.
    skills: [
      {
        name: 'Languages & Frameworks',
        description:
          'JavaScript (ES6+), Golang, Node.js, Express.js, React, Ruby on Rails, PHP',
      },
      {
        name: 'Databases',
        description: 'MongoDB, PostreSQL, MySQL',
      },
      {
        name: 'Other',
        description:
          'Docker, Amazon Web Services (AWS), CI / CD, Microservices, API design, Agile / Scrum',
      },
    ],
  },
  plugins: [
    `gatsby-plugin-react-helmet`,
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `images`,
        path: `${__dirname}/src/images`,
      },
    },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        path: `${__dirname}/content/blog`,
        name: `blog`,
      },
    },
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          {
            resolve: `gatsby-remark-images`,
            options: {
              maxWidth: 590,
              wrapperStyle: `margin: 0 0 30px;`,
            },
          },
          {
            resolve: `gatsby-remark-responsive-iframe`,
            options: {
              wrapperStyle: `margin-bottom: 1.0725rem`,
            },
          },
          `gatsby-remark-prismjs`,
          `gatsby-remark-copy-linked-files`,
          `gatsby-remark-smartypants`,
        ],
      },
    },
    `gatsby-transformer-sharp`,
    `gatsby-plugin-sharp`,
    `gatsby-plugin-postcss`,
    `gatsby-plugin-feed`,
    {
      resolve: `gatsby-plugin-google-analytics`,
      options: {
        trackingId: `ADD YOUR TRACKING ID HERE`, // Optional Google Analytics
      },
    },
    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        name: `devfolio`,
        short_name: `devfolio`,
        start_url: `/`,
        background_color: `#663399`,
        theme_color: `#663399`, // This color appears on mobile
        display: `minimal-ui`,
        icon: `src/images/icon.png`,
      },
    },
  ],
};
