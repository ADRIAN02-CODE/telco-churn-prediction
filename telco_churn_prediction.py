{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNOCj3D2jfRnZZB4oY1pd6L",
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
        "outputId": "c86e354e-d6c6-4a2e-888b-b427564ceb6a"
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
      "cell_type": "code",
      "source": [
        "# [Beginner Note] Let's look for the data file in a few common places, so anyone with the file can load it.\n",
        "possible_paths = ['Telecom Dataset.xlsx', '/content/Telecom Dataset.xlsx', 'WA_Fn-UseC_-Telco-Customer-Churn.csv']\n",
        "data_loaded = False\n",
        "for path in possible_paths:\n",
        "    if os.path.exists(path):\n",
        "        if path.lower().endswith('.csv'):\n",
        "            df = pd.read_csv(path)\n",
        "        else:\n",
        "            df = pd.read_excel(path)\n",
        "        print(f'Data loaded from {path}')\n",
        "        data_loaded = True\n",
        "        break\n",
        "if not data_loaded:\n",
        "    raise FileNotFoundError('Dataset not found. Please upload it and update possible_paths.')\n",
        "print(df.head())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "j2reC3jHVWf0",
        "outputId": "d9e338e3-e1c3-4894-a23b-8146afd6da5b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Data loaded from WA_Fn-UseC_-Telco-Customer-Churn.csv\n",
            "   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService  \\\n",
            "0  7590-VHVEG  Female              0     Yes         No       1           No   \n",
            "1  5575-GNVDE    Male              0      No         No      34          Yes   \n",
            "2  3668-QPYBK    Male              0      No         No       2          Yes   \n",
            "3  7795-CFOCW    Male              0      No         No      45           No   \n",
            "4  9237-HQITU  Female              0      No         No       2          Yes   \n",
            "\n",
            "      MultipleLines InternetService OnlineSecurity  ... DeviceProtection  \\\n",
            "0  No phone service             DSL             No  ...               No   \n",
            "1                No             DSL            Yes  ...              Yes   \n",
            "2                No             DSL            Yes  ...               No   \n",
            "3  No phone service             DSL            Yes  ...              Yes   \n",
            "4                No     Fiber optic             No  ...               No   \n",
            "\n",
            "  TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling  \\\n",
            "0          No          No              No  Month-to-month              Yes   \n",
            "1          No          No              No        One year               No   \n",
            "2          No          No              No  Month-to-month              Yes   \n",
            "3         Yes          No              No        One year               No   \n",
            "4          No          No              No  Month-to-month              Yes   \n",
            "\n",
            "               PaymentMethod MonthlyCharges  TotalCharges Churn  \n",
            "0           Electronic check          29.85         29.85    No  \n",
            "1               Mailed check          56.95        1889.5    No  \n",
            "2               Mailed check          53.85        108.15   Yes  \n",
            "3  Bank transfer (automatic)          42.30       1840.75    No  \n",
            "4           Electronic check          70.70        151.65   Yes  \n",
            "\n",
            "[5 rows x 21 columns]\n"
          ]
        }
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