{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPQSJu68wwzWhBe8hdSmCLB",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ADRIAN02-CODE/telco-churn-prediction/blob/main/telco_churn_prediction.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "jAY5MRiQNb-5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e7f58f54-3125-451e-cd65-6d48235a1592"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Libraries imported!\n"
          ]
        }
      ],
      "source": [
        "\n",
        "import os\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve\n",
        "\n",
        "from sklearn.cluster import KMeans\n",
        "from sklearn.decomposition import PCA\n",
        "\n",
        "import joblib\n",
        "\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Dropout\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "\n",
        "try:\n",
        "    from imblearn.over_sampling import SMOTE\n",
        "    IMBLEARN_AVAILABLE = True\n",
        "except Exception:\n",
        "    IMBLEARN_AVAILABLE = False\n",
        "\n",
        "RANDOM_STATE = 42\n",
        "TEST_SIZE = 0.2\n",
        "\n",
        "print('Libraries imported!')\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "!git config --global user.name \"Adrian02-CODE\"\n",
        "!git config --global user.email \"adrianvethanayagam@gmail.com\"\n"
      ],
      "metadata": {
        "id": "V5uDPcJmOa9C"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "iBkxQCFBOzJe"
      }
    }
  ]
}