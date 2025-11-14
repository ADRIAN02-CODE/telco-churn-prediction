{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO0ZRRlhwWd+Q//IQdvBGkN",
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
      "execution_count": null,
      "metadata": {
        "id": "jAY5MRiQNb-5",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a4b9758d-5679-40a5-c2f0-3a421a5e98ff"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Libraries imported successfully.\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import warnings\n",
        "warnings.filterwarnings(\"ignore\")\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score, classification_report, confusion_matrix,\n",
        "    roc_auc_score, roc_curve\n",
        ")\n",
        "import joblib\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Dropout\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "\n",
        "# Optional SMOTE\n",
        "try:\n",
        "    from imblearn.over_sampling import SMOTE\n",
        "    IMBLEARN_AVAILABLE = True\n",
        "except:\n",
        "    IMBLEARN_AVAILABLE = False\n",
        "\n",
        "print(\"✅ Libraries imported successfully.\")\n"
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