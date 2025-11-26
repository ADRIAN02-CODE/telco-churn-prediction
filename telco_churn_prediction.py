{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNAjC4hGfD0Vp6JH2nJmfe7",
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
        "outputId": "c0cb92db-c434-4a72-e234-603141301024"
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
        "outputId": "ab41d04f-0eb1-4646-a0c6-d4bfdb4ecdac"
      },
      "execution_count": 4,
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
      "cell_type": "code",
      "source": [
        "# [Beginner Note] EDA: We check the dataset for missing values, outliers, and see how the columns work.\n",
        "print(df.info())\n",
        "print(df.describe())\n",
        "\n",
        "# [Why?] It helps spot problems (like missing data), see class balance, and get to know feature distributions.\n",
        "plt.figure(figsize=(6,4))\n",
        "sns.countplot(x='Churn', data=df)\n",
        "plt.title('Churn Distribution')\n",
        "plt.show()\n",
        "# [Beginner Note] This shows how many customers left (churned) vs. stayed.\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "V28RqNy3fccG",
        "outputId": "dc27f246-c35f-400b-e60a-a9b443cce676"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 7043 entries, 0 to 7042\n",
            "Data columns (total 21 columns):\n",
            " #   Column            Non-Null Count  Dtype  \n",
            "---  ------            --------------  -----  \n",
            " 0   customerID        7043 non-null   object \n",
            " 1   gender            7043 non-null   object \n",
            " 2   SeniorCitizen     7043 non-null   int64  \n",
            " 3   Partner           7043 non-null   object \n",
            " 4   Dependents        7043 non-null   object \n",
            " 5   tenure            7043 non-null   int64  \n",
            " 6   PhoneService      7043 non-null   object \n",
            " 7   MultipleLines     7043 non-null   object \n",
            " 8   InternetService   7043 non-null   object \n",
            " 9   OnlineSecurity    7043 non-null   object \n",
            " 10  OnlineBackup      7043 non-null   object \n",
            " 11  DeviceProtection  7043 non-null   object \n",
            " 12  TechSupport       7043 non-null   object \n",
            " 13  StreamingTV       7043 non-null   object \n",
            " 14  StreamingMovies   7043 non-null   object \n",
            " 15  Contract          7043 non-null   object \n",
            " 16  PaperlessBilling  7043 non-null   object \n",
            " 17  PaymentMethod     7043 non-null   object \n",
            " 18  MonthlyCharges    7043 non-null   float64\n",
            " 19  TotalCharges      7043 non-null   object \n",
            " 20  Churn             7043 non-null   object \n",
            "dtypes: float64(1), int64(2), object(18)\n",
            "memory usage: 1.1+ MB\n",
            "None\n",
            "       SeniorCitizen       tenure  MonthlyCharges\n",
            "count    7043.000000  7043.000000     7043.000000\n",
            "mean        0.162147    32.371149       64.761692\n",
            "std         0.368612    24.559481       30.090047\n",
            "min         0.000000     0.000000       18.250000\n",
            "25%         0.000000     9.000000       35.500000\n",
            "50%         0.000000    29.000000       70.350000\n",
            "75%         0.000000    55.000000       89.850000\n",
            "max         1.000000    72.000000      118.750000\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x400 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiUAAAGJCAYAAABVW0PjAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAALiFJREFUeJzt3XtcVPW+//H3gDKiOOAV9IhKmhcUNa2tcyzzQrINu0mWbreXvJSGlVLqcT/cZrZPlm7zmressJPu0kxLKZCDijvFS3Qorxw1DEsBy2DUFBDm90eH+TnhDQLnu+P1fDzmsZ3v+sx3fZbFnndrfdfC4nQ6nQIAAPAwL083AAAAIBFKAACAIQglAADACIQSAABgBEIJAAAwAqEEAAAYgVACAACMQCgBAABGIJQAAAAjEEqA3xGLxaLx48d7uo1KM2LECDVv3vyW7Kt58+YaMWKE631sbKwsFou++OKLW7L/nj17qmfPnrdkX4ApCCXAv4Djx4/rqaee0m233aYaNWrIZrOpe/fuWrBggS5evOjp9splxowZslgsrlfNmjXVtGlTPfDAA3rnnXeUn59fIfs5dOiQZsyYoRMnTlTIfBXJ5N4AT6jm6QYAXF9cXJwGDhwoq9WqYcOGqX379iooKNDnn3+uSZMm6eDBg1qxYoWn2yy3pUuXys/PT/n5+fr++++VkJCgkSNHav78+dq8ebOCg4NdtW+++aaKi4vLNP+hQ4f00ksvqWfPnmU6y5Keni4vr8r977br9bZly5ZK3TdgIkIJYLCMjAwNGjRIzZo109atW9WoUSPXtujoaB07dkxxcXG3tKfi4mIVFBSoRo0aFTLfo48+qvr167veT58+XatXr9awYcM0cOBA7d6927WtevXqFbLPa3E6nbp06ZJ8fX1ltVordV834uPj49H9A57A5RvAYLNnz9b58+f11ltvuQWSEi1bttRzzz1Xanzjxo1q3769rFar2rVrp/j4eLft11qbUXJJ5Uol61RWr16tdu3ayWq1Kj4+3rXGYufOnYqJiVGDBg1Uq1YtPfLIIzpz5sxvOu4hQ4Zo9OjR2rNnjxITE6/b9/vvv68uXbqodu3astlsCgsL04IFCyT9sg5k4MCBkqRevXq5LhVt375d0i/rRvr376+EhATdeeed8vX11fLly13brlxTUuLnn3/WU089pXr16slms2nYsGH66aefSv2dzZgxo9Rnr5zzRr1dbU1JTk6ORo0apcDAQNWoUUMdO3bUqlWr3GpOnDghi8Wiv//971qxYoVatGghq9Wqu+66S/v27bvq3zdgCs6UAAbbtGmTbrvtNv37v//7TX/m888/10cffaSnn35atWvX1sKFCxUVFaXMzEzVq1evXH1s3bpVa9eu1fjx41W/fn01b95caWlpkqRnnnlGderU0YsvvqgTJ05o/vz5Gj9+vD744INy7avE0KFDtWLFCm3ZskX33XffVWsSExM1ePBg9enTR6+99pok6fDhw9q5c6eee+459ejRQ88++6wWLlyov/zlL2rbtq0kuf5X+uUyzeDBg/XUU09pzJgxat269XX7Gj9+vAICAjRjxgylp6dr6dKl+vbbb7V9+/ZSge56bqa3K128eFE9e/bUsWPHNH78eIWEhGjdunUaMWKEcnNzS4XTNWvW6Ny5c3rqqadksVg0e/ZsDRgwQN98802ln3ECyotQAhjK4XDo+++/10MPPVSmzx0+fFiHDh1SixYtJP3yX+EdO3bUP/7xj3LfmZOenq79+/crNDTUNVYSSurVq6ctW7a4vpCLi4u1cOFC5eXlyd/fv1z7k6T27dtL+mWR77XExcXJZrMpISFB3t7epbbfdtttuueee7Rw4ULdd999V72b5dixY4qPj1dERMRN9eXj46OkpCTXF3uzZs00efJkbdq0SQ8++OBNzXGzvV1pxYoVOnz4sN577z0NGTJEkjR27Fjde++9mjZtmkaOHKnatWu76jMzM3X06FHVqVNHktS6dWs99NBDSkhIUP/+/W+6T+BW4vINYCiHwyFJbl80NyM8PNwVSCSpQ4cOstls+uabb8rdy7333usWSK705JNPup0huOeee1RUVKRvv/223PuTJD8/P0nSuXPnrlkTEBCgCxcuuF3iKauQkJCbDiTSL8d75ZmGcePGqVq1avr000/L3cPN+PTTTxUUFKTBgwe7xqpXr65nn31W58+fV3Jyslv9448/7gok0i//XCT9pn8PgMpGKAEMZbPZJF3/S/lqmjZtWmqsTp06pdY9lEVISMhN76/ki/C37E+Szp8/L+n6oezpp59Wq1at1K9fPzVp0kQjR44stX7mRq53bFdz++23u7338/NTo0aNKv223m+//Va33357qTuCSi73/DoEVtY/F6AyEUoA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Cirr57NkXqfTecPaKhFKfvjhBxUVFSkwMNBtPDAwUEeOHClVP2vWLL300kulxoODgyutR6Aq8F801tMtALiGyv75PHfunPz9/a9bUyVCSVlNnTpVMTExrvfFxcU6e/as6tWrJ4vF4sHOUFEcDoeCg4N18uRJ2Ww2T7cD4Ar8fP6+OJ1OnTt3To0bN75hbZUIJfXr15e3t7eys7PdxrOzsxUUFFSq3mq1ymq1uo0FBARUZovwEJvNxv/pAYbi5/P340ZnSEpUiYWuPj4+6tKli5KSklxjxcXFSkpKkt1u92BnAACgRJU4UyJJMTExGj58uO6880794Q9/0Pz583XhwgU98cQTnm4NAACoCoWSxx9/XGfOnNH06dOVlZWlTp06KT4+vtTiV1QNVqtVL774YqnLdAA8j5/PqsvivJl7dAAAACpZlVhTAgAAzEcoAQAARiCUAAAAIxBKAACAEQgl+N0aMWKELBaLXn31VbfxjRs38mRewAOcTqfCw8MVERFRatuSJUsUEBCg7777zgOdwRSEEvyu1ahRQ6+99pp++uknT7cCVHkWi0XvvPOO9uzZo+XLl7vGMzIyNHnyZC1atEhNmjTxYIfwNEIJftfCw8MVFBSkWbNmXbNm/fr1ateunaxWq5o3b665c+fewg6BqiU4OFgLFizQCy+8oIyMDDmdTo0aNUp9+/bVHXfcoX79+snPz0+BgYEaOnSofvjhB9dnP/zwQ4WFhcnX11f16tVTeHi4Lly44MGjQUUjlOB3zdvbW6+88ooWLVp01dPCqampeuyxxzRo0CDt379fM2bM0F//+lfFxsbe+maBKmL48OHq06ePRo4cqcWLF+vAgQNavny5evfurTvuuENffPGF4uPjlZ2drccee0ySdPr0aQ0ePFgjR47U4cOHtX37dg0YMEA8auv3hYen4XdrxIgRys3N1caNG2W32xUaGqq33npLGzdu1COPPCKn06khQ4bozJkz2rJli+tzkydPVlxcnA4ePOjB7oHft5ycHLVr105nz57V+vXrdeDAAf3zn/9UQkKCq+a7775TcHCw0tPTdf78eXXp0kUnTpxQs2bNPNg5KhNnSlAlvPbaa1q1apUOHz7sNn748GF1797dbax79+46evSoioqKbmWLQJXSsGFDPfXUU2rbtq0efvhhffXVV9q2bZv8/PxcrzZt2kiSjh8/ro4dO6pPnz4KCwvTwIED9eabb7JW7HeIUIIqoUePHoqIiNDUqVM93QqA/1OtWjVVq/bLr2A7f/68HnjgAaWlpbm9jh49qh49esjb21uJiYn67LPPFBoaqkWLFql169bKyMjw8FGgIlWZX8gHvPrqq+rUqZNat27tGmvbtq127tzpVrdz5061atVK3t7et7pFoMrq3Lmz1q9fr+bNm7uCyq9ZLBZ1795d3bt31/Tp09WsWTNt2LBBMTExt7hbVBbOlKDKCAsL05AhQ7Rw4ULX2PPPP6+kpCS9/PLL+t///V+tWrVKixcv1gsvvODBToGqJzo6WmfPntXgwYO1b98+HT9+XAkJCXriiSdUVFSkPXv26JVXXtEXX3yhzMxMffTRRzpz5ozatm3r6dZRgQglqFJmzpyp4uJi1/vOnTtr7dq1ev/999W+fXtNnz5dM2fO1IgRIzzXJFAFNW7cWDt37lRRUZH69u2rsLAwTZgwQQEBAfLy8pLNZtOOHTt0//33q1WrVpo2bZrmzp2rfv36ebp1VCDuvgEAAEbgTAkAADACoQQAABiBUAIAAIxAKAEAAEYglAAAACMQSgAAgBEIJQAAwAiEEgAAYARCCQCjWSwWbdy40dNtALgFCCUAPCorK0vPPPOMbrvtNlmtVgUHB+uBBx5QUlKSp1sDcIvxW4IBeMyJEyfUvXt3BQQEaM6cOQoLC1NhYaESEhIUHR2tI0eOVMp+CwoK5OPjUylzAyg/zpQA8Jinn35aFotFe/fuVVRUlFq1aqV27dopJiZGu3fvdtX98MMPeuSRR1SzZk3dfvvt+uSTT1zbYmNjFRAQ4Dbvxo0bZbFYXO9nzJihTp06aeXKlQoJCVGNGjUk/XJpaOXKldecG8CtRSgB4BFnz55VfHy8oqOjVatWrVLbrwwaL730kh577DF9/fXXuv/++zVkyBCdPXu2TPs7duyY1q9fr48++khpaWkVOjeAikEoAeARx44dk9PpVJs2bW5YO2LECA0ePFgtW7bUK6+8ovPnz2vv3r1l2l9BQYHeffdd3XHHHerQoUOFzg2gYhBKAHiE0+m86dorQ0StWrVks9mUk5NTpv01a9ZMDRo0qJS5AVQMQgkAj7j99ttlsVhuajFr9erV3d5bLBYVFxdLkry8vEoFnMLCwlJzXO0S0Y3mBnBrEUoAeETdunUVERGhN954QxcuXCi1PTc396bmadCggc6dO+c2x5VrRgD86yCUAPCYN954Q0VFRfrDH/6g9evX6+jRozp8+LAWLlwou91+U3N07dpVNWvW1F/+8hcdP35ca9asUWxsbOU2DqBSEEoAeMxtt92mL7/8Ur169dLzzz+v9u3b67777lNSUpKWLl16U3PUrVtX7733nj799FOFhYXpH//4h2bMmFG5jQOoFBZnWVabAQAAVBLOlAAAACMQSgAAgBEIJQAAwAiEEgAAYARCCQAAMAKhBAAAGIFQAgAAjEAoAQAARiCUAAAAIxBKAACAEQglAADACP8P908fltobizoAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
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